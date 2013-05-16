__author__ = "Mikael Mortensen <mikaem@math.uio.no>"
__date__ = "2011-12-19"
__copyright__ = "Copyright (C) 2011 " + __author__
__license__  = "GNU Lesser GPL version 3 or any later version"
"""
This is a highly tuned and stripped down Navier-Stokes solver optimized
for both speed and memory.  

To use this solver for a new problem you have to modify two sections.
    'Problem dependent parameters':
        add a mesh and set problem specific parameters. 
        Parameters that are recognized are already given 
        default values.

    'Boundary conditions':
        Simply create the dictionary bcs, with keys 
        'u0', 'u1', 'u2' and 'p' and value a list of 
        boundary conditions. 
        If necessary enable normalization of the pressure.
        
The algorithm used is a second order in time fractional step method
(incremental pressure correction).

We use a Crank-Nicolson discretization in time of the Laplacian and 
the convected velocity. The convecting velocity is computed with an 
Adams-Bashforth projection. The fractional step method can be used
both non-iteratively or with iterations over the pressure velocity 
system.

We assemble a single coefficient matrix used by all velocity componenets
and build it by preassembling as much as possible. The matrices used are:

    A  = Coefficient (lhs) matrix or rhs matrix
    Ac = Convection matrix
    K  = Diffusion matrix
    M  = Mass matrix

For the lhs A is computed as:

    A  = 1/dt*M + 0.5*Ac + 0.5*K

However, we start by assembling a coefficient matrix (Ar) that is used 
to compute the rhs:

    Ar = 1/dt*M - 0.5*Ac - 0.5*K
    b  = A*u_1.vector()

Ac needs to be reassembled each new timestep into A to save memory.
A and Ar are recreated each timestep by assembling Ac, setting up Ar 
and then using the following to create the lhs A:

   A = - Ar + 2/dt*M

"""
from cbc.cfd.oasis import *
from numpy import arctan, array, sin, linspace, mod
import random
from os import getpid, path, makedirs, getcwd, listdir
from scipy.interpolate import interp1d
import time
from cbc.cfd.tools.Probe import StructuredGrid

parameters["form_compiler"]["optimize"]     = True   # I sometimes get memory access error with True here (MM)
parameters["form_compiler"]["cpp_optimize"] = True
parameters['mesh_partitioner'] = "ParMETIS"

# Check how much memory is actually used by dolfin before we allocate anything
#dolfin_memory_use = getMyMemoryUsage()
#info_red('Memory use of plain dolfin = ' + dolfin_memory_use)

################### Problem dependent parameters ####################

mesh = Mesh("/usit/abel/u1/mikaem/create_mesh/Mesh_Tet_H.xml")
normal = FacetNormal(mesh)
### HACK!!
ds = ds[mesh.domains().facet_domains()]

info_red('Finished reading mesh {}'.format(mesh.num_vertices()))
    
# Set parameters
nu = Constant(1.0e-6)           # Viscosity
t = 0.0                       # time
tstep = 0                     # Timestep
T = 0.78e-4                        # End time
max_iter = 1                  # Pressure velocity iterations on given timestep
iters_on_first_timestep = 1   # Pressure velocity iterations on first timestep
max_error = 1e-6
check = 1000                    # print out info and save solution every check timestep 
save_restart_file = 2000       # Saves two previous timesteps needed for a clean restart
probe_interval = 1
    
# Specify body force
dim = mesh.geometry().dim()
f = Constant((0,)*dim)

# Set the timestep
#dt =  0.2*(h / U)
#n  = int(T / dt + 1.0)
#dt = Constant(T / n)
dt = Constant(0.78e-4)
n = int(T / dt(0))

# Give a folder for storing the results
folder = "csf_bryn"

# To avoid writing over old data create a new folder for each run                              
# Create a new folder for each run
folder = path.join(folder, 'data', 'dt={0:2.4e}'.format(dt(0)))
if not path.exists(folder):
    folder = path.join(folder, '1')
else:
    previous = listdir(folder)
    folder = path.join(folder, str(max(map(eval, previous)) + 1))

MPI.barrier()
vtkfolder = path.join(folder, "VTK")
statsfolder = path.join(folder, "Stats")
if MPI.process_number() == 0:
    try:
        makedirs(statsfolder)
    except:
        pass
if MPI.process_number() == 0:
    try:
        makedirs(vtkfolder)
    except:
        pass

#### Set a folder that contains xml.gz files of the solution. 
#restart_folder = None        
restart_folder = "/usit/abel/u1/mikaem/Fenics/cbcpdesys/cbc/cfd/csf_bryn/data/dt=7.8000e-05/5/timestep=24000/"    
#### Use for initialization if not None
    
#####################################################################

# Declare solution Functions and FunctionSpaces
V = FunctionSpace(mesh, 'CG', 1)
Q = FunctionSpace(mesh, 'CG', 1)
Vv = VectorFunctionSpace(mesh, 'CG', V.ufl_element().degree())
u = TrialFunction(V)
v = TestFunction(V)
p = TrialFunction(Q)
q = TestFunction(Q)

u_components = ['u0', 'u1', 'u2']
sys_comp =  u_components + ['p']

# Use dictionaries to hold all Functions and FunctionSpaces
VV = dict((ui, V) for ui in u_components); VV['p'] = Q

def strain(u):
    return 0.5*(grad(u)+ grad(u).T)

def omega(u):
    return 0.5*(grad(u) - grad(u).T)

def Omega(u):
    return inner(omega(u), omega(u))

def Strain(u):
    return inner(strain(u), strain(u))

# Start from previous solution if restart_folder is given
if restart_folder:
    q_  = dict((ui, Function(VV[ui], path.join(restart_folder, ui + '.xml.gz'))) for ui in sys_comp)
    q_1 = dict((ui, Function(V, path.join(restart_folder, ui + '.xml.gz'))) for ui in u_components)
    q_2 = dict((ui, Function(V, path.join(restart_folder, ui + '_1.xml.gz'))) for ui in u_components)
else:
    q_  = dict((ui, Function(VV[ui])) for ui in sys_comp)
    q_1 = dict((ui, Function(V)) for ui in u_components)
    q_2 = dict((ui, Function(V)) for ui in u_components)

u_  = as_vector([q_[ui]  for ui in u_components]) # Velocity vector at t
u_1 = as_vector([q_1[ui] for ui in u_components]) # Velocity vector at t - dt
u_2 = as_vector([q_2[ui] for ui in u_components]) # Velocity vector at t - 2*dt
p_ = q_['p']                # pressure at t - dt/2
dp_ = Function(Q)               # pressure correction

# Slices
#sl = [StructuredGrid(V, [250, 250], [0.003, 0.01, 0.022], [[1., 0., 0.], [0., 0., 1.]], [0.021, 0.017], statistics=True),
#      StructuredGrid(V, [250, 250], [0.006, 0.02, 0.016], [[1., 0., 0.], [0., 0., 1.]], [0.016, 0.015], statistics=True),
#      StructuredGrid(V, [250, 250], [0.004, 0.03, 0.01], [[1., 0., 0.], [0., 0., 1.]], [0.02, 0.015], statistics=True),
#      StructuredGrid(V, [250, 250], [0.007, 0.04, 0.005], [[1., 0., 0.], [0., 0., 1.]], [0.017, 0.015], statistics=True),
#      StructuredGrid(V, [250, 250], [0.004, 0.05, 0.002], [[1., 0., 0.], [0., 0., 1.]], [0.023, 0.014], statistics=True),
#      StructuredGrid(V, [250, 250], [0.006, 0.06, 0.0], [[1., 0., 0.], [0., 0., 1.]], [0.021, 0.013], statistics=True),
#      StructuredGrid(V, [250, 250], [0.005, 0.07, 0.0], [[1., 0., 0.], [0., 0., 1.]], [0.024, 0.013], statistics=True),
#      StructuredGrid(V, [250, 250], [0.007, 0.08, 0.001], [[1., 0., 0.], [0., 0., 1.]], [0.022, 0.014], statistics=True),
#      StructuredGrid(V, [250, 250], [0.007, 0.09, 0.002], [[1., 0., 0.], [0., 0., 1.]], [0.021, 0.015], statistics=True),
#      StructuredGrid(V, [250, 250], [0.006, 0.10, 0.003], [[1., 0., 0.], [0., 0., 1.]], [0.023, 0.015], statistics=True),
#      StructuredGrid(V, [250, 250], [0.008, 0.11, 0.005], [[1., 0., 0.], [0., 0., 1.]], [0.020, 0.016], statistics=True),
#      StructuredGrid(V, [250, 250], [0.008, 0.12, 0.007], [[1., 0., 0.], [0., 0., 1.]], [0.021, 0.016], statistics=True),
#      StructuredGrid(V, [250, 250], [0.008, 0.13, 0.01], [[1., 0., 0.], [0., 0., 1.]], [0.022, 0.015], statistics=True),
#      StructuredGrid(V, [250, 250], [0.008, 0.14, 0.013], [[1., 0., 0.], [0., 0., 1.]], [0.022, 0.017], statistics=True),
#      StructuredGrid(V, [250, 250], [0.008, 0.15, 0.016], [[1., 0., 0.], [0., 0., 1.]], [0.023, 0.020], statistics=True),
#      StructuredGrid(V, [250, 250], [0.008, 0.16, 0.02], [[1., 0., 0.], [0., 0., 1.]], [0.023, 0.022], statistics=True),
#      StructuredGrid(V, [250, 250], [0.003, 0.17, 0.018], [[1., 0., 0.], [0., 0., 1.]], [0.031, 0.033], statistics=True)]

EnstrophyBox = StructuredGrid(V, [100, 300, 100], [0.004, 0.05, 0.0], [[1., 0., 0.], [0., 1., 0.], [0., 0., 1.]], [0.025, 0.075, 0.025])
EnstrophyLumpedBox = StructuredGrid(V, [100, 300, 100], [0.004, 0.05, 0.0], [[1., 0., 0.], [0., 1., 0.], [0., 0., 1.]], [0.025, 0.075, 0.025])

#ypos = range(10, 180, 10)

class RandomStreamFunction(Expression):
    def __init__(self):
        random.seed(2 + MPI.process_number())
    def eval(self, values, x):
        values[0] = 0.000001*random.random()

class RandomStreamVector(Expression):
    def __init__(self):
        random.seed(2 + MPI.process_number())
    def eval(self, values, x):
        values[0] = 0.0000001*random.random()
        values[1] = 0.0000001*random.random()
        values[2] = 0.0000001*random.random()
    def value_shape(self):
        return (3,)
        
##psi = interpolate(RandomStreamFunction(), V)
if restart_folder == None:
    psi = interpolate(RandomStreamVector(), Vv)
    u0 = project(curl(psi), Vv)
    u0x = project(u0[0], V)
    u1x = project(u0[1], V)
    u2x = project(u0[2], V)
    #u0 = project(psi.dx(0), V)
    q_['u0'].vector()[:] = u0x.vector()[:]
    q_['u1'].vector()[:] = u1x.vector()[:]
    q_['u2'].vector()[:] = u2x.vector()[:]
    q_1['u0'].vector()[:] = q_['u0'].vector()[:]
    q_2['u0'].vector()[:] = q_['u0'].vector()[:]
    q_1['u1'].vector()[:] = q_['u1'].vector()[:]
    q_2['u1'].vector()[:] = q_['u1'].vector()[:]
    q_1['u2'].vector()[:] = q_['u2'].vector()[:]
    q_2['u2'].vector()[:] = q_['u2'].vector()[:]

###################  Boundary conditions  ###########################

bcs = dict((ui, []) for ui in sys_comp)

# Fra Fluent:
#(0 "Zone Sections")
#(39 (16 fluid BODY)())
#(39 (17 interior int_BODY)())
#(39 (18 pressure-outlet TOP)())
#(39 (19 velocity-inlet BOTTOM)())
#(39 (20 wall DURA)())
#(39 (21 wall PIA)())
#(39 (22 wall VENTRAL)())
#(39 (23 wall DORSAL)())
#(39 (24 wall DL)())
#(39 (25 wall DL-TOPS)())

p_top = Constant(0)

# Find distance to wall
#Walls.type = lambda : 'Wall'
#Walls.mf = mf
#Walls.bid = 1 # Boundary indicator for wall
#Eikonal.solver_parameters['max_err'] = 1e-6
#Eikonal.solver_parameters['linear_solver'] = 'gmres'
#Eikonal.solver_parameters['precond'] = 'ml_amg'
#eikonal = Eikonal.Eikonal(mesh, [Walls], parameters=Eikonal.solver_parameters)
#distance_to_wall = eikonal.y_
#u_in = Expression("-sin(2*t*pi)*64.*(sqrt(x[0]*x[0]+x[1]*x[1])-0.5)*(0.75-sqrt(x[0]*x[0]+x[1]*x[1]))", t=0)
#ff = File('distance.xml.gz')
#ff << distance_to_wall

one = Constant(1.)
n = FacetNormal(mesh)
V0 = assemble(one*dx, mesh=mesh)
A1 = assemble(one*ds(18), mesh=mesh)
A2 = assemble(one*ds(19), mesh=mesh)
A3 = assemble(one*n[0]*ds(19), mesh=mesh)
A4 = assemble(one*n[1]*ds(19), mesh=mesh)
A5 = assemble(one*n[2]*ds(19), mesh=mesh)

print "Volume = ", V0
print "Area of outlet = ", A1, " and inlet = ", A2, A3, A4, A5

#tid = array([0.00, 0.05, 0.11, 0.16, 0.21, 0.27, 0.32, 0.37, 0.42, 0.48, 0.5])
#U = array([1.57252, 0.43811, -3.73425, -5.8157, -3.89452, -1.1597, 1.59744, 2.89147, 2.10766, 1.68838, 1.57252]) * 1e-6 / A2
#Uut = array([1.57252, 0.43811, -3.73425, -5.8157, -3.89452, -1.1597, 1.59744, 2.89147, 2.10766, 1.68838, 1.57252]) * 1e-6 / A1
tid = array([0.00, 0.07, 0.13, 0.20, 0.26, 0.33, 0.39, 0.45, 0.52, 0.59, 0.65, 0.72, 0.78, 0.8])
U = array([1.74, 1.05, -1.61, -3.43, -2.84, -2.05, -0.75, 0.16, 0.28, 0.56, 0.92, 1.41, 1.69, 1.74]) * 1e-6 / A2
Uut = array([1.74, 1.05, -1.61, -3.43, -2.84, -2.05, -0.75, 0.16, 0.28, 0.56, 0.92, 1.41, 1.69, 1.74]) * 1e-6 / A1
u_in = interp1d(tid, U)
u_ut = interp1d(tid, Uut)

class UV(Expression):

    def __init__(self, mesh=None, d=None, y_=None, y0=None, **kwargs):
        self.mesh = mesh
        self.d = d
        self.y_ = y_
        self.y0 = y0
 
    def eval_cell(self, value, x, ufc_cell):
        # Find normal
        cell = Cell(self.mesh, ufc_cell.index)
        normal = cell.normal(ufc_cell.local_facet)
        #value[0] = -self.d*self.y_(x)/self.y0*normal.x()
        #value[1] = -self.d*self.y_(x)/self.y0*normal.y()
        #value[2] = -self.d*self.y_(x)/self.y0*normal.z()
        value[0] = -self.d*normal.x()
        value[1] = -self.d*normal.y()
        value[2] = -self.d*normal.z()

    def value_shape(self):
        return (3,)

class U0(Expression):

    def __init__(self, mesh=None, d=None, y_=None, y0=None, **kwargs):
        self.mesh = mesh
        self.d = d
        self.y_ = y_
        self.y0 = y0
 
    def eval_cell(self, value, x, ufc_cell):
        # Find normal
        cell = Cell(self.mesh, ufc_cell.index)
        normal = cell.normal(ufc_cell.local_facet)
        #value[0] = -self.d*self.y_(x)/self.y0*normal.x()
        value[0] = -self.d*normal.x()

class U1(Expression):

    def __init__(self, mesh=None, d=None, y_=None, y0=None, **kwargs):
        self.mesh = mesh
        self.d = d
        self.y_ = y_
        self.y0 = y0
 
    def eval_cell(self, value, x, ufc_cell):
        # Find normal
        cell = Cell(self.mesh, ufc_cell.index)
        normal = cell.normal(ufc_cell.local_facet)
        #value[0] = -self.d*self.y_(x)/self.y0*normal.y()
        value[0] = -self.d*normal.y()

class U2(Expression):

    def __init__(self, mesh=None, d=None, y_=None, y0=None, **kwargs):
        self.mesh = mesh
        self.d = d
        self.y_ = y_
        self.y0 = y0
 
    def eval_cell(self, value, x, ufc_cell):
        # Find normal
        cell = Cell(self.mesh, ufc_cell.index)
        normal = cell.normal(ufc_cell.local_facet)
        #value[0] = -self.d*self.y_(x)/self.y0*normal.z()
        value[0] = -self.d*normal.z()

# First contract all wall boundaries into the same number
mvc = mesh.domains().markers(2)
wall_ids = [21, 22, 23, 24, 25]
for key, val in mvc.values().iteritems():
    if val in wall_ids:
        mvc.set_value(key[0], key[1], 20)

bcs['u0'] = [DirichletBC(V, Constant(0), 20)]

# Find distance to wall
#F1 = inner(grad(u), grad(v))*dx - Constant(1)*v*dx
#y_ = Function(V)
#solve(lhs(F1) == rhs(F1), y_, bcs=bcs['u0'], solver_parameters={'linear_solver': 'bicgstab', 'preconditioner': 'hypre_euclid'})
#print 'Norm y_ = ', y_.vector().norm('l2')
#F2 = sqrt(inner(grad(y_), grad(y_)))*v*dx  -  Constant(1)*v*dx + \
#               Constant(0.01)*inner(grad(y_), grad(v))*dx
#solve(F2 == 0, y_, bcs=bcs['u0'], solver_parameters={'linear_solver': 'bicgstab', 'preconditioner': 'hypre_euclid'})
#y0 = assemble(y_*ds(19))
#print 'Norm y0 = ', y0

#u0_in = U0(mesh=mesh, d=0, y_=y_, y0=y0)
#u1_in = U1(mesh=mesh, d=0, y_=y_, y0=y0)
#u2_in = U2(mesh=mesh, d=0, y_=y_, y0=y0)
#uv_in = UV(mesh=mesh, d=0, y_=y_, y0=y0)
#u0_ut = U0(mesh=mesh, d=0, y_=y_, y0=y0)
#u1_ut = U1(mesh=mesh, d=0, y_=y_, y0=y0)
#u2_ut = U2(mesh=mesh, d=0, y_=y_, y0=y0)
#uv_ut = UV(mesh=mesh, d=0, y_=y_, y0=y0)

u1_code = '''
class U1 : public Expression
{
  public:

    double d;
    boost::shared_ptr< Mesh > mesh;

    U1() : Expression() {}

  void eval(Array<double>& values, const Array<double>& x,
            const ufc::cell& c) const
    {
      const Cell cell((*mesh), c.index);   
      values[0] = -d * cell.normal(c.local_facet, 1);
    }
};'''

u1_in = Expression(u1_code)
u1_out = Expression(u1_code)
u1_in.mesh = mesh
u1_out.mesh = mesh
u1_in.d = float(0.)
u1_out.d = float(0.)

bcs['u0'].insert(0, DirichletBC(V, Constant(0), 19))
bcs['u0'].insert(0, DirichletBC(V, Constant(0), 18))

bcs['u1'] = [DirichletBC(V, u1_in, 19),
             DirichletBC(V, u1_out, 18),
             DirichletBC(V, Constant(0), 20)]

bcs['u2'] = [DirichletBC(V, Constant(0), 19),
             DirichletBC(V, Constant(0), 18),
             DirichletBC(V, Constant(0), 20)]

bcs['p']  = [DirichletBC(Q, Constant(0), 18)]

[bc.apply(q_['u0'].vector()) for bc in bcs['u0']]
[bc.apply(q_['u1'].vector()) for bc in bcs['u1']]
[bc.apply(q_['u2'].vector()) for bc in bcs['u2']]

# Normalize pressure or not?
normalize = False

#####################################################################

# Preassemble some constant in time matrices
M = assemble(inner(u, v)*dx)                    # Mass matrix
K = assemble(nu*inner(grad(u), grad(v))*dx)     # Diffusion matrix
Ap = assemble(inner(grad(q), grad(p))*dx)       # Pressure Laplacian
A = Matrix()                                    # Coefficient matrix (needs reassembling)

###########
# Declare some variables used for lumping of the mass matrix
ones = Vector(q_['u0'].vector())
ones[:] = 1.

### Enstrophy will be probed using lumped mass matrix ###
lumped_inverse = Function(V)
LV = lumped_inverse.vector()
LV[:] = M * ones
LV.set_local(1. / LV.array())
enstrophy_form = 0.5*dot(curl(u_), curl(u_))*v*dx
enstrophy = Function(V)
enstrophy_vec = enstrophy.vector()
################################

# Apply boundary conditions on M and Ap that are used directly in solve
[bc.apply(Ap) for bc in bcs['p']]
[bc.apply(M)  for bc in bcs['u0']]

# Lumping of mass matrix after applied boundary conditions
ML = M * ones
MP = Vector(ML)
ML.set_local(1. / ML.array())

# Adams Bashforth projection of velocity at t - dt/2
U_ = 1.5*u_1 - 0.5*u_2

# Convection form
a  = 0.5*inner(v, dot(U_, nabla_grad(u)))*dx

# Preassemble constant pressure gradient matrix
P = dict((ui, assemble(v*p.dx(i)*dx)) for i, ui in enumerate(u_components))

# Preassemble velocity divergence matrix
Rc = P    
#Rc = dict((ui, assemble(q*u.dx(i)*dx)) for i, ui in  enumerate(u_components))

reset_sparsity = True

u_sol = KrylovSolver('bicgstab', 'jacobi')
u_sol.parameters['error_on_nonconvergence'] = False
u_sol.parameters['nonzero_initial_guess'] = True
u_sol.parameters['monitor_convergence'] = True
u_sol.parameters['maximum_iterations'] = 50
#u_sol.parameters['relative_tolerance'] = 1e-7
#u_sol.parameters['absolute_tolerance'] = 1e-10
    
p_sol = KrylovSolver('gmres', 'hypre_amg')
p_sol.parameters['error_on_nonconvergence'] = False
p_sol.parameters['nonzero_initial_guess'] = True
p_sol.parameters['preconditioner']['reuse'] = True
p_sol.parameters['monitor_convergence'] = True
p_sol.parameters['maximum_iterations'] = 50
#p_sol.parameters['relative_tolerance'] = 1e-7
#p_sol.parameters['absolute_tolerance'] = 1e-10

x_  = dict((ui, q_ [ui].vector()) for ui in sys_comp)     # Solution vectors t
x_1 = dict((ui, q_1[ui].vector()) for ui in u_components) # Solution vectors t - dt
x_2 = dict((ui, q_2[ui].vector()) for ui in u_components) # Solution vectors t - 2*dt
b   = dict((ui, Vector(x_[ui])) for ui in sys_comp)       # rhs vectors
bold= dict((ui, Vector(x_[ui])) for ui in sys_comp)       # rhs temp storage vectors
work = Vector(x_['u0'])

t0 = t1 = time.time()
dt_ = dt(0)
total_iters = 0
ttot = time.time()
probe_time = time.time()
while t < (T - tstep*DOLFIN_EPS):
    t += dt_
    tstep += 1
    j = 0
    err = 1e8
    total_iters += 1
    
    ### prepare ###
    uu = u_in(mod(t, 0.8))
    uu_ut = u_ut(mod(t, 0.8))
    u1_in.d = float(uu)
    u1_out.d = float(-uu_ut)
    ### prepare ###
    
    if tstep == 1:
        num_iter = max(iters_on_first_timestep, max_iter)
    else:
        num_iter = max_iter
                
    while err > max_error and j < num_iter:
        err = 0
        j += 1
        ### Start by solving for an intermediate velocity ###
        if j == 1:
            # Only on the first iteration because nothing here is changing in time
            # Set up coefficient matrix for computing the rhs:
            A = assemble(a, tensor=A, reset_sparsity=reset_sparsity) 
            reset_sparsity = False
            A._scale(-1.)            # Negative convection on the rhs 
            A.axpy(1./dt_, M, True)  # Add mass
            A.axpy(-0.5, K, True)    # Add diffusion                
            # Compute rhs for all velocity components
            for ui in u_components:
                b[ui][:] = A*x_1[ui]
            # Reset matrix for lhs
            A._scale(-1.)
            A.axpy(2./dt_, M, True)
            [bc.apply(A) for bc in bcs['u0']]
            
        for ui in u_components:
            bold[ui][:] = b[ui][:]
            b[ui].axpy(-1., P[ui]*x_['p'])
            [bc.apply(b[ui]) for bc in bcs[ui]]
            work[:] = x_[ui][:]
            #if u_sol.parameters['monitor_convergence'] and MPI.process_number() == 0:
            if MPI.process_number() == 0:
                print 'Solving tentative ', ui
            u_sol.solve(A, x_[ui], b[ui])
            err += norm(work - x_[ui])
            b[ui][:] = bold[ui][:] # In case of inner iterations
            
        ### Solve pressure ###
        dp_.vector()[:] = x_['p'][:]
        b['p'][:] = Ap * x_['p']
        for ui in u_components:
            b['p'].axpy(-1./dt_, Rc[ui]*x_[ui]) # Divergence of u_
        [bc.apply(b['p']) for bc in bcs['p']]
        rp = residual(Ap, x_['p'], b['p'])
        #if p_sol.parameters['monitor_convergence'] and MPI.process_number() == 0:
        if MPI.process_number() == 0:
            print 'Solving p' 
            
        p_sol.solve(Ap, x_['p'], b['p'])
        dp_.vector()[:] = x_['p'][:] - dp_.vector()[:]
        if tstep % check == 0:
            if num_iter > 1:
                if j == 1: info_blue('                 error u  error p')
                info_blue('    Iter = {0:4d}, {1:2.2e} {2:2.2e}'.format(j, err, rp))

    ### Update ################################################################
    ### Update velocity ###
    # Lumping
    for ui in u_components:
        MP[:] = (P[ui] * dp_.vector()) * ML
        x_[ui].axpy(-dt_, MP)
        [bc.apply(x_[ui]) for bc in bcs[ui]]

    # Update to a new timestep
    for ui in u_components:
        x_2[ui][:] = x_1[ui][:]
        x_1[ui][:] = x_ [ui][:]

    if tstep % probe_interval == 0:
        tp = time.time()
        enstrophy_vec[:] = assemble(enstrophy_form, tensor=enstrophy_vec)
        enstrophy_vec[:] = enstrophy_vec * LV
        EnstrophyLumpedBox(enstrophy)

        enstrophy2 = project(0.5*dot(curl(u_), curl(u_)), V)
        EnstrophyBox(enstrophy2)

        #EnstrophyBox.tovtk(1, filename=vtkfolder+"/snapshot_box_{}.vtk".format(tstep))
        EnstrophyLumpedBox.toh5_lowmem(0, tstep, filename=vtkfolder+"/enstrophyLumped.h5")
        EnstrophyLumpedBox.probes.clear()
        EnstrophyBox.toh5_lowmem(0, tstep, filename=vtkfolder+"/enstrophy.h5")
        EnstrophyBox.probes.clear()

        #for yp, sli in zip(ypos, sl):
        #    sli(q_['u0'], q_['u1'], q_['u2'])
        #    sli.tovtk(1, filename=vtkfolder+"/snapshot_{}_{}.vtk".format(yp, tstep))

        probe_time += time.time() - tp

    # Print some information and save intermediate solution
    if tstep % check == 0:
        list_timings(True)
        if MPI.process_number()==0:
            tottime= time.time() - t1
            info_red('Total computing time on previous {0:d} timesteps = {1:f}'.format(check, tottime))
            info_red('Probe time {}'.format(probe_time))
            info_green('Time = {0:2.4e}, timestep = {1:6d}, End time = {2:2.4e}'.format(t, tstep, T)) 
        probe_time = 0
        newfolder = path.join(folder, 'timestep='+str(tstep))        
        #plot(u_[0], rescale=True)
        u1 = assemble(dot(u_, normal)*ds(18), mesh=mesh)
        u2 = assemble(dot(u_, normal)*ds(19), mesh=mesh)
        if MPI.process_number() == 0:
            print 'flux [cm/s] = ', u1/A1, u2/A2, u1, u2
        MPI.barrier()

        if MPI.process_number() == 0:
           try:
               makedirs(newfolder)
           except OSError:
               pass
        for ui in sys_comp:
           newfile = File(path.join(newfolder, ui + '.xml.gz'))
           newfile << q_[ui]        
        if tstep % save_restart_file == 0:
           for ui in u_components:
               newfile_1 = File(path.join(newfolder, ui + '_1.xml.gz'))
               newfile_1 << q_1[ui]

        if MPI.process_number()==0:
            ff = open(newfolder+"/timeprstep_{}.txt".format(tottime/check), "w")
            ff.close()
        t1 = time.time()

### end of time loop ###
list_timings()
info_red('Total computing time = {0:f}'.format(time.time()- ttot))


