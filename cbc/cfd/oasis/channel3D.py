__author__ = "Mikael Mortensen <mikaem@math.uio.no>"
__date__ = "2011-12-19"
__copyright__ = "Copyright (C) 2011 " + __author__
__license__  = "GNU GPL version 3 or any later version"
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
from numpy import arctan, array
import random

#parameters["linear_algebra_backend"] = "Epetra"
parameters["linear_algebra_backend"] = "PETSc"
parameters["form_compiler"]["optimize"]     = False   # I sometimes get memory access error with True here (MM)
parameters["form_compiler"]["cpp_optimize"] = True
set_log_active(True)

# Check how much memory is actually used by dolfin before we allocate anything
dolfin_memory_use = getMyMemoryUsage()
info_red('Memory use of plain dolfin = ' + dolfin_memory_use)

################### Problem dependent parameters ####################

Lx = 4.
Ly = 2.
Lz = 2.
Nx = 40
Ny = 50
Nz = 30
mesh = Box(0., -Ly/2., -Lz/2., Lx, Ly/2., Lz/2., Nx, Ny, Nz)
# Create stretched mesh in y-direction
x = mesh.coordinates()        
x[:, 1] = arctan(pi*(x[:, 1]))/arctan(pi) 
normal = FacetNormal(mesh)
    
# Set parameters
Re_tau = 395.
nu = Constant(2.e-5)           # Viscosity
utau = nu(0) * Re_tau
t = 0.0                        # time
tstep = 0                      # Timestep
T = 10.0                        # End time
max_iter = 1                  # Pressure velocity iterations on given timestep
iters_on_first_timestep = 2   # Pressure velocity iterations on first timestep
max_error = 1e-6
check = 10                    # print out info and save solution every check timestep 
save_vtk = 10000
save_restart_file = 50000       # Saves two previous timesteps needed for a clean restart
    
# Specify body force
dim = mesh.geometry().dim()
#f = Constant((0,)*dim)

# Set the timestep
#dt =  0.2*(h / U)
#n  = int(T / dt + 1.0)
#dt = Constant(T / n)
dt = Constant(0.1)
n = int(T / dt(0))

# Give a folder for storing the results
folder = "/home/mikael/Fenics/cbcpdesys/cbc/cfd/oasis/channel395/"
#folder = "/home-4/mikaelmo/cbcpdesys/cbc/cfd/oasis/csf_results/dt=2.0000e-01/13"
vtk_file = File("/home/mikael/Fenics/cbcpdesys/cbc/cfd/oasis/channel395/VTK/u/u.pvd")
u_stats_file = File("/home/mikael/Fenics/cbcpdesys/cbc/cfd/oasis/channel395/Stats/umean.xml.gz")
k_stats_file = File("/home/mikael/Fenics/cbcpdesys/cbc/cfd/oasis/channel395/Stats/kmean.xml.gz")
p_stats_file = File("/home/mikael/Fenics/cbcpdesys/cbc/cfd/oasis/channel395/Stats/pmean.xml.gz")

if not folder is None:
    if MPI.process_number() == 0:
        if not path.exists(folder):
            raise IOError("Folder {0} does not exist".format(folder))
        
else:
    # Create a new folder for each run
    folder = path.join(getcwd(), 'csf_results', 
                                'dt={0:2.4e}'.format(dt(0)))

    # To avoid writing over old data create a new folder for each run                              
    if not path.exists(folder):
        folder = path.join(folder, '1')
    else:
        previous = listdir(folder)
        folder = path.join(folder, str(max(map(eval, previous)) + 1) )
    if MPI.process_number() == 0:
        makedirs(folder)
    
#### Set a folder that contains xml.gz files of the solution. 
restart_folder = None        
#restart_folder = '/home/mikaelmo/cbcpdesys/cbc/cfd/oasis/csf_results/dt=0.001/1'
#restart_folder = "/home-4/mikaelmo/cbcpdesys/cbc/cfd/oasis/csf_results/dt=2.0000e-01/13/timestep=12500"
#### Use for initialization if not None
    
#####################################################################
# Declare solution Functions and FunctionSpaces
V = FunctionSpace(mesh, 'CG', 1)
Q = FunctionSpace(mesh, 'CG', 1)
Vv = VectorFunctionSpace(mesh, 'CG', V.ufl_element().degree())
R = FunctionSpace(mesh, 'R', 0)
QR = Q * R
u = TrialFunction(V)
v = TestFunction(V)
p, c = TrialFunctions(QR)
q, d = TestFunctions(QR)

if dim == 2:
    u_components = ['u0', 'u1']
else:
    u_components = ['u0', 'u1', 'u2']
sys_comp =  u_components + ['pc']

# Use dictionaries to hold all Functions and FunctionSpaces
VV = dict((ui, V) for ui in u_components); VV['pc'] = QR

# Start from previous solution if restart_folder is given
if restart_folder:
    q_  = dict((ui, Function(VV[ui], path.join(restart_folder, ui + '.xml.gz'))) for ui in sys_comp)
    q_1 = dict((ui, Function(V, path.join(restart_folder, ui + '.xml.gz'))) for ui in u_components)
    q_2 = dict((ui, Function(V, path.join(restart_folder, ui + '_1.xml.gz'))) for ui in u_components)
else:
    q_  = dict((ui, Function(VV[ui])) for ui in sys_comp)
    q_1 = dict((ui, Function(V)) for ui in u_components)
    q_2 = dict((ui, Function(V)) for ui in u_components)

class RandomStreamFunction(Expression):
    def __init__(self):
        random.seed(2 + MPI.process_number())
    def eval(self, values, x):
        values[0] = 0.01*random.random()

class RandomStreamVector(Expression):
    def __init__(self):
        random.seed(2 + MPI.process_number())
    def eval(self, values, x):
        values[0] = 0.002*random.random()
        values[1] = 0.002*random.random()
        values[2] = 0.002*random.random()
    def value_shape(self):
        return (3,)
        
psi = interpolate(RandomStreamFunction(), V)
psi = interpolate(RandomStreamVector(), Vv)
u0 = project(curl(psi), Vv)
u0x = project(u0[0], V)
u1x = project(u0[1], V)
u2x = project(u0[2], V)
#dy = interpolate(Expression("x[1] > 0. ? 1. - x[1] : x[1] + 1."), V)
y = interpolate(Expression("2*0.1335*((1+x[1])*(1-x[1]))"), V)

if restart_folder == None:    
   #u0 = project(psi.dx(0), V)
   q_['u0'].vector()[:] = y.vector()[:] 
   q_['u0'].vector().axpy(1.0, u0x.vector())
   q_['u1'].vector()[:] = u1x.vector()[:]
   q_['u2'].vector()[:] = u2x.vector()[:]
   #u1 = project(-psi.dx(1), V)
   #q_['u1'].vector()[:] = u1.vector()[:]
   q_1['u0'].vector()[:] = q_['u0'].vector()[:]
   q_2['u0'].vector()[:] = q_['u0'].vector()[:]
   q_1['u1'].vector()[:] = q_['u1'].vector()[:]
   q_2['u1'].vector()[:] = q_['u1'].vector()[:]
   q_1['u2'].vector()[:] = q_['u2'].vector()[:]
   q_2['u2'].vector()[:] = q_['u2'].vector()[:]
else:
   # Add a random field to jumpstart the turbulence
   q_['u0'].vector().axpy(1.0, u0x.vector())
   q_['u1'].vector().axpy(1.0, u1x.vector())
   q_['u2'].vector().axpy(1.0, u2x.vector())
   q_1['u0'].vector().axpy(1.0, u0x.vector())
   q_1['u1'].vector().axpy(1.0, u1x.vector())
   q_1['u2'].vector().axpy(1.0, u2x.vector())
   q_2['u0'].vector().axpy(1.0, u0x.vector())
   q_2['u1'].vector().axpy(1.0, u1x.vector())
   q_2['u2'].vector().axpy(1.0, u2x.vector())

u_  = as_vector([q_[ui]  for ui in u_components]) # Velocity vector at t
u_1 = as_vector([q_1[ui] for ui in u_components]) # Velocity vector at t - dt
u_2 = as_vector([q_2[ui] for ui in u_components]) # Velocity vector at t - 2*dt
pc_ = q_['pc']
q_['p'], q_['c'] = p_, c_ = pc_.split()
dpc_ = Function(QR)      # pressure correction
dp_, dc_ = dpc_.split()

###################  Boundary conditions  ###########################

bcs = dict((ui, []) for ui in sys_comp)

def walls(x, on_bnd):
    return on_bnd and (near(x[1], -Ly/2.) or near(x[1], Ly/2.))

def outlet(x, on_bnd):
    return on_bnd and near(x[0], Lx)

def outletz(x, on_bnd):
    return on_bnd and near(x[2], Lz/2.)

class PeriodicBoundaryX(SubDomain):

    # Left boundary is "target domain" G
    def inside(self, x, on_boundary):
        return near(x[0], 0.) and on_boundary

    # Map right boundary (H) to left boundary (G)
    def map(self, x, y):
        y[0] = x[0] - Lx
        y[1] = x[1]
        y[2] = x[2]

class PeriodicBoundaryZ(SubDomain):

    # Left boundary is "target domain" G
    def inside(self, x, on_boundary):
        return near(x[2], -Lz/2.) and on_boundary

    # Map right boundary (H) to left boundary (G)
    def map(self, x, y):
        y[0] = x[0]
        y[1] = x[1]
        y[2] = x[2] - Lz
                
pbx = PeriodicBoundaryX()
pbz = PeriodicBoundaryZ()
# Create FacetFunction for computing intermediate results
mf = FacetFunction("uint", mesh) # Facets
mf.set_all(0)
Walls = AutoSubDomain(walls)
Walls.mark(mf, 1)
pbx.mark(mf, 2)
pbz.mark(mf, 3)
Outlet = AutoSubDomain(outlet)
Outletz = AutoSubDomain(outletz)
Outlet.mark(mf, 4)
Outletz.mark(mf, 5)

one = Constant(1)
V0 = assemble(one*dx, mesh=mesh)
A1 = assemble(one*ds(1), mesh=mesh, exterior_facet_domains=mf)
A2 = assemble(one*ds(2), mesh=mesh, exterior_facet_domains=mf)
A3 = assemble(one*ds(3), mesh=mesh, exterior_facet_domains=mf)

# Preassemble constant pressure gradient matrix
# Before creating PeriodicBC to avoid master/slave sparsity issues
P = dict((ui, assemble(v*p.dx(i)*dx)) for i, ui in enumerate(u_components))

# Preassemble velocity divergence matrix
Rx = dict((ui, assemble(q*u.dx(i)*dx)) for i, ui in  enumerate(u_components))
#for ui in  u_components:
#    bcs['p'][0].pre_solve_elimination(QR._periodic_master_slave_dofs, Rx[ui])

bc = [PeriodicBC(V, pbx), PeriodicBC(V, pbz), DirichletBC(V, Constant(0), walls)]
bcs['u0'] = bc
bcs['u1'] = bc
bcs['u2'] = bc
bcs['pc'] = [PeriodicBC(QR, pbx), PeriodicBC(QR, pbz)]

#####################################################################

# Preassemble some constant in time matrices
M = assemble(inner(u, v)*dx)                    # Mass matrix
K = assemble(nu*inner(grad(u), grad(v))*dx)     # Diffusion matrix
Ap = assemble(inner(grad(q), grad(p))*dx + 0*(inner(d, p) + inner(q, c)) * dx)    # Pressure Laplacian
A = Matrix()                                    # Coefficient matrix (needs reassembling)

NN = Ap.size(0)
if (NN-1 >= Ap.local_range(0)[0] and NN-1 < Ap.local_range(0)[1]):
    a1 = Ap.getrow(NN-1)
    a2 = Ap.getrow(NN-2)
    a1[1][-2] = 1.
    a2[1][-1] = 1.
    #Ap.setrow(NN-2, array(a2[0], 'I'), a2[1])
    #Ap.setrow(NN-1, array(a1[0], 'I'), a1[1])
    Ap.setrow(NN-2, array([NN-2], 'I'), array([1.]))
    Ap.setrow(NN-1, array([NN-2, NN-1], 'I'), array([1., 1.]))
Ap.apply("insert")

# Apply boundary conditions on M and Ap that are used directly in solve
App = Ap.copy()
MM = M.copy()
[bc.apply(MM) for bc in bcs['u0']]
[bc.apply(App) for bc in bcs['pc']]
bcs['pc'][0].pre_solve_elimination(QR._periodic_master_slave_dofs, App)
bcs['u0'][0].pre_solve_elimination(V._periodic_master_slave_dofs, MM)
MM.compress()
App.compress()

# Adams Bashforth projection of velocity at t - dt/2
U_ = 1.5*u_1 - 0.5*u_2

# Convection form
a  = 0.5*inner(v, dot(U_, nabla_grad(u)))*dx

#u_sol = LUSolver()
u_sol = KrylovSolver('bicgstab', 'jacobi')
u_sol.parameters['error_on_nonconvergence'] = False
u_sol.parameters['nonzero_initial_guess'] = True
u_sol.parameters['preconditioner']['reuse'] = False
u_sol.parameters['monitor_convergence'] = False
u_sol.parameters['maximum_iterations'] = 50
#u_sol.parameters['relative_tolerance'] = 1e-9
#u_sol.parameters['absolute_tolerance'] = 1e-10
u_sol.t = 0
reset_sparsity = True

#du_sol = LUSolver()
du_sol = KrylovSolver('bicgstab', 'hypre_euclid')
du_sol.parameters['error_on_nonconvergence'] = False
du_sol.parameters['nonzero_initial_guess'] = True
du_sol.parameters['preconditioner']['reuse'] = True
du_sol.parameters['monitor_convergence'] = False
du_sol.parameters['maximum_iterations'] = 50
#du_sol.parameters['relative_tolerance'] = 1e-9
#du_sol.parameters['absolute_tolerance'] = 1e-10
du_sol.t = 0

#p_sol = LUSolver()
p_sol = KrylovSolver('gmres', 'hypre_amg')
p_sol.parameters['error_on_nonconvergence'] = False
p_sol.parameters['nonzero_initial_guess'] = True
p_sol.parameters['preconditioner']['reuse'] = True
p_sol.parameters['monitor_convergence'] = False
p_sol.parameters['maximum_iterations'] = 50
#p_sol.parameters['relative_tolerance'] = 1e-9
#p_sol.parameters['absolute_tolerance'] = 1e-10
p_sol.t = 0

x_  = dict((ui, q_ [ui].vector()) for ui in sys_comp)     # Solution vectors t
x_1 = dict((ui, q_1[ui].vector()) for ui in u_components) # Solution vectors t - dt
x_2 = dict((ui, q_2[ui].vector()) for ui in u_components) # Solution vectors t - 2*dt
b   = dict((ui, Vector(x_[ui])) for ui in sys_comp)       # rhs vectors
bold= dict((ui, Vector(x_[ui])) for ui in sys_comp)       # rhs temp storage vectors
work = Vector(x_['u0'])

dpdx = (utau**2, 0., 0.)
b0 = dict((ui, assemble(v*Constant(dpdx[i])*dx)) for i, ui in enumerate(u_components))

u0mean = Function(V)
kmean = Function(V)
pmean = Function(QR)

# To project solution on Vv more efficiently:
uv = TrialFunction(Vv)
vv = TestFunction(Vv)
AV = assemble(inner(uv, vv)*dx)
dv = DirichletBC(Vv, Constant((0, 0, 0)), walls)
dv.apply(AV)
AV.compress()
uvtk = Function(Vv)
vtk_time = 0
project_sol = KrylovSolver('gmres', 'hypre_amg')
project_sol.parameters['error_on_nonconvergence'] = False
project_sol.parameters['nonzero_initial_guess'] = True
project_sol.parameters['preconditioner']['reuse'] = True
project_sol.parameters['monitor_convergence'] = True
project_sol.parameters['maximum_iterations'] = 50

t0 = t1 = time.time()
dt_ = dt(0)
total_iters = 0
tstep0 = tstep
while t < T + DOLFIN_EPS:
    t += dt_
    tstep += 1
    j = 0
    err = 1e8
    total_iters += 1
    
    ### prepare ###    
    #b0 = dict((ui, assemble(v*Constant(dpdx[i])*dx)) for i, ui in enumerate(u_components))
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
            reset_sparsity = False   # Warning! Must be true for periodic boundary conditions
            A._scale(-1.)            # Negative convection on the rhs 
            A.axpy(1./dt_, M, True)  # Add mass
            A.axpy(-0.5, K, True)    # Add diffusion
            # Compute rhs for all velocity components
            for i, ui in enumerate(u_components):
                b[ui][:] = b0[ui][:]
                b[ui].axpy(1., A*x_1[ui])
            # Reset matrix for lhs
            A._scale(-1.)
            A.axpy(2./dt_, M, True)
            [bc.apply(A) for bc in bcs['u0']]
            bcs['u0'][0].pre_solve_elimination(V._periodic_master_slave_dofs, A)
            
        for ui in u_components:
            bold[ui][:] = b[ui][:]
            b[ui].axpy(-1., P[ui]*x_['pc'])
            [bc.apply(b[ui]) for bc in bcs[ui]]
            work[:] = x_[ui][:]
            #if u_sol.parameters['monitor_convergence'] and MPI.process_number() == 0:
            if MPI.process_number() == 0:
                print 'Solving tentative ', ui
            if ui == "u0":
                u_sol.parameters['preconditioner']['reuse'] = False
            else:
                u_sol.parameters['preconditioner']['reuse'] = True
            t0 = time.time()
            u_sol.solve(A, x_[ui], b[ui])
            u_sol.t += (time.time()-t0)
            err += norm(work - x_[ui])
            b[ui][:] = bold[ui][:]
            bcs[ui][0].post_solve(V._periodic_master_slave_dofs, x_[ui])
            
        ### Solve pressure ###
        dpc_.vector()[:] = x_['pc'][:]
        b['pc'][:] = Ap*x_['pc']
        for ui in u_components:
            b['pc'].axpy(-1./dt_, Rx[ui]*x_[ui]) # Divergence of u_
        [bc.apply(b['pc']) for bc in bcs['pc']]
        rp = residual(Ap, x_['pc'], b['pc'])
        #if p_sol.parameters['monitor_convergence'] and MPI.process_number() == 0:
        if MPI.process_number() == 0:
            print 'Solving p'        
        t0 = time.time()
        p_sol.solve(App, x_['pc'], b['pc'])
        p_sol.t += (time.time()-t0)
        bcs['pc'][0].post_solve(QR._periodic_master_slave_dofs, x_['pc'])
        dpc_.vector()[:] = x_['pc'][:] - dpc_.vector()[:]
        if tstep % check == 0:
            if num_iter > 1:
                if j == 1: info_blue('                 error u  error p')
                info_blue('    Iter = {0:4d}, {1:2.2e} {2:2.2e}'.format(j, err, rp))

    ### Update ################################################################
    ### Update velocity ###
    for ui in u_components:
        b[ui][:] = M*x_[ui][:]        
        b[ui].axpy(-dt_, P[ui]*dpc_.vector())
        [bc.apply(b[ui]) for bc in bcs[ui]]
        #if du_sol.parameters['monitor_convergence'] and MPI.process_number() == 0:
        if MPI.process_number() == 0:  
            print 'Solving ', ui
        t0 = time.time()
        du_sol.solve(MM, x_[ui], b[ui])
        du_sol.t += (time.time()-t0)
        bcs[ui][0].post_solve(V._periodic_master_slave_dofs, x_[ui])
    
    # Update to a new timestep
    for ui in u_components:
        x_2[ui][:] = x_1[ui][:]
        x_1[ui][:] = x_ [ui][:]

    ## Update statistics
    #u0mean.vector().axpy(1.0, x_['u0'])
    #kmean.vector().axpy(1.0, 0.5 * (x_['u0'] * x_['u0'] + x_['u1'] * x_['u1'] + x_['u2'] * x_['u2']))
    #pmean.vector().axpy(1.0, x_['pc'])
    #################################################################################################
        
    # Print some information and save intermediate solution
    vtk0 = time.time()
    if tstep % save_vtk == 0:
        uvtk = project(u_, Vv)
        #bvtk = assemble(dot(u_, vv)*dx)
        #dv.apply(bvtk)
        if MPI.process_number() == 0:
            print 'Projecting velocity'
        #project_sol.solve(AV, uvtk.vector(), bvtk)
        vtk_file << uvtk
    vtk_time += time.time() - vtk0
    
    if tstep % check == 0:
        list_timings(True)
        info_red('Total computing time on previous {0:d} timesteps = {1:f}'.format(check, time.time() - t1))
        t1 = time.time()
        info_green('Time = {0:2.4e}, timestep = {1:6d}, End time = {2:2.4e}'.format(t, tstep, T)) 
        #newfolder = path.join(folder, 'timestep='+str(tstep))
        #u2 = assemble(dot(u_, normal)*ds(2), mesh=mesh, exterior_facet_domains=mf)
        #u3 = assemble(dot(u_, normal)*ds(3), mesh=mesh, exterior_facet_domains=mf)
        #u4 = assemble(dot(u_, normal)*ds(4), mesh=mesh, exterior_facet_domains=mf)
        #u5 = assemble(dot(u_, normal)*ds(5), mesh=mesh, exterior_facet_domains=mf)
        
        #plot(u_[0], rescale=True)
        #if MPI.process_number()==0:
           #print 'flux [m/s] = ', (u2+u4)/u4, (u3+u5)/u5
           #try:
               #makedirs(newfolder)
           #except OSError:
               #pass
        #for ui in sys_comp:
           #newfile = File(path.join(newfolder, ui + '.xml.gz'))
           #newfile << q_[ui]
        
        #if tstep % save_restart_file == 0:
           #for ui in u_components:
               #newfile_1 = File(path.join(newfolder, ui + '_1.xml.gz'))
               #newfile_1 << q_1[ui]
    ### Update ################################################################            
u0mean.vector()._scale(1./(tstep-tstep0))
kmean.vector()._scale(1./(tstep-tstep0))
pmean.vector()._scale(1./(tstep-tstep0))
u_stats_file << u0mean
k_stats_file << kmean
p_stats_file << pmean

info_red('Additional memory use of solver = {0}'.format(eval(getMyMemoryUsage()) - eval(dolfin_memory_use)))
info_red('Total memory use = ' + getMyMemoryUsage())
list_timings()
info_red('Total computing time = {0:f}'.format(time.time()- t0))
plot(u_[0])
print 'u_sol ', u_sol.t
print 'du_sol ', du_sol.t
print 'p_sol ', p_sol.t
#plot(project(u_, Vv))
print 'vtk time ', vtk_time
interactive()


