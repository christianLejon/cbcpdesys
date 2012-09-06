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

parameters["linear_algebra_backend"] = "Epetra"
#parameters["linear_algebra_backend"] = "PETSc"
parameters["form_compiler"]["optimize"]     = False   # I sometimes get memory access error with True here (MM)
parameters["form_compiler"]["cpp_optimize"] = True
set_log_active(True)

# Check how much memory is actually used by dolfin before we allocate anything
dolfin_memory_use = getMyMemoryUsage()
info_red('Memory use of plain dolfin = ' + dolfin_memory_use)

################### Problem dependent parameters ####################

from csf_formula4 import smooth_flow, create_spline, splev

a = 2.5     
b = 10.0 
c = 0.7
dt = 0.001
m = 2 
smooth_func = smooth_flow(a, b, c, dt, m)
spline_func = create_spline(smooth_func, m, c, dt)

#mesh = Mesh("/home/mikaelmo/cbcpdesys/cbc/cfd/data/straight_nerves_refined.xml")
mesh = Mesh("/home/mikael/Fenics/cbcpdesys/cbc/cfd/data/csf_block80_refined.xml")
#mesh = Mesh("/home/mikael/Fenics/cbcpdesys/cbc/cfd/data/straight_nerves_refined.xml")
#mesh = UnitCube(20, 20, 25)
#xx = mesh.coordinates()
#xx[:, 2] = 10.*(xx[:, 2] - 0.5)
#xx[:, 1] = 0.25*xx[:, 1]
#xx[:, 0] = 0.25*xx[:, 0]
normal = FacetNormal(mesh)
    
# Set parameters
nu = Constant(0.007)           # Viscosity
t = 0.0                         # time
tstep = 0                     # Timestep
T = 1.                        # End time
max_iter = 1                  # Pressure velocity iterations on given timestep
iters_on_first_timestep = 2   # Pressure velocity iterations on first timestep
max_error = 1e-6
check = 100                    # print out info and save solution every check timestep 
save_restart_file = 1000       # Saves two previous timesteps needed for a clean restart
    
# Specify body force
dim = mesh.geometry().dim()
f = Constant((0,)*dim)

# Set the timestep
#dt =  0.2*(h / U)
#n  = int(T / dt + 1.0)
#dt = Constant(T / n)
dt = Constant(1.e-4)
n = int(T / dt(0))

# Give a folder for storing the results
folder = None

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
        folder = path.join(folder, str(max(map(eval, previous)) + 1))
    if MPI.process_number() == 0:
        makedirs(folder)
    
#### Set a folder that contains xml.gz files of the solution. 
restart_folder = None        
#restart_folder = '/home/mikaelmo/cbcpdesys/cbc/cfd/oasis/csf_results/dt=0.001/1'
#### Use for initialization if not None
    
#####################################################################

# Declare solution Functions and FunctionSpaces
V = FunctionSpace(mesh, 'CG', 1)
Q = FunctionSpace(mesh, 'CG', 1)
R = FunctionSpace(mesh, 'R', 0)
Vv = VectorFunctionSpace(mesh, 'CG', V.ufl_element().degree())
QRR = MixedFunctionSpace([Q, R, R])
u = TrialFunction(V)
v = TestFunction(V)
p, pa, pb = TrialFunctions(QRR)
q, qa, qb = TestFunctions(QRR)

if dim == 2:
    u_components = ['u0', 'u1']
else:
    u_components = ['u0', 'u1', 'u2']
sys_comp =  u_components + ['pcc']

# Use dictionaries to hold all Functions and FunctionSpaces
VV = dict((ui, V) for ui in u_components); VV['pcc'] = QRR

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
dpdx = splev(0, spline_func)/10.
q_['pcc'] = interpolate(Expression(("dpdx*(x[2]+5)", "0", "0"), dpdx=dpdx), QRR)
pcc_ = q_['pcc']                # pressure at t - dt/2
dpcc_ = Function(QRR)               # pressure correction
p_, ca_, cb_ = pcc_.split()
dp_, dca_, dcb_ = dpcc_.split()
###################  Boundary conditions  ###########################

bcs = dict((ui, []) for ui in sys_comp)

tol = 10.*DOLFIN_EPS
# Just mark all boundaries as walls, overwrite with top and bottom
def walls(x, on_bnd):
    return on_bnd
    
#def walls(x, on_bnd):
    #return on_bnd and (x[0] < tol or x[0] > 0.25 - tol or x[1] < tol or x[1] > 0.25 - tol) 
    
def top(x, on_bnd):
    return abs(x[2] - 5.) < 1.e-12 and on_bnd 
    
def bottom(x, on_bnd):
    return abs(x[2] + 5.) < 1.e-12 and on_bnd 

# Create FacetFunction for computing intermediate results
mf = FacetFunction("uint", mesh) # Facets
mf.set_all(0)
Walls = AutoSubDomain(walls)
Walls.mark(mf, 1)
Top = AutoSubDomain(top)
Top.mark(mf, 2)
Bottom = AutoSubDomain(bottom)
Bottom.mark(mf, 3)
one = Constant(1)
V0 = assemble(one*dx, mesh=mesh)
A1 = assemble(one*ds(1), mesh=mesh, exterior_facet_domains=mf)
A2 = assemble(one*ds(2), mesh=mesh, exterior_facet_domains=mf)
A3 = assemble(one*ds(3), mesh=mesh, exterior_facet_domains=mf)
    
p_top = Constant(0)
p_bottom = Constant(0)
bcs['u0'] = [DirichletBC(V, Constant(0), mf, 1)]
bcs['u1'] = [DirichletBC(V, Constant(0), mf, 1)]
bcs['u2'] = [DirichletBC(V, Constant(0), mf, 1)]
#bcs['p']  = [DirichletBC(Q, p_top, top),
             #DirichletBC(Q, p_bottom, bottom)]
bcs['p'] = []
# Normalize pressure or not?
normalize = False

#####################################################################

# Preassemble some constant in time matrices
M = assemble(inner(u, v)*dx)                    # Mass matrix
K = assemble(nu*inner(grad(u), grad(v))*dx)     # Diffusion matrix
App = assemble(inner(grad(q), grad(p))*dx)      # Pressure Laplacian
Ap = assemble(inner(grad(q), grad(p))*dx + pa*q*ds(2) + p*qa*ds(2) + pb*q*ds(3) + p*qb*ds(3),
              exterior_facet_domains=mf)        # Pressure Laplacian plus boundary terms
A = Matrix()                                    # Coefficient matrix (needs reassembling)

dp0 = p_top*qa*ds(2)

# Apply boundary conditions on M that are used directly in solve
[bc.apply(M)  for bc in bcs['u0']]

# Adams Bashforth projection of velocity at t - dt/2
U_ = 1.5*u_1 - 0.5*u_2

# Convection form
a  = 0.5*inner(v, dot(U_, nabla_grad(u)))*dx

# Preassemble constant pressure gradient matrix
P = dict((ui, assemble(v*p.dx(i)*dx)) for i, ui in enumerate(u_components))

# Preassemble velocity divergence matrix
#if V.ufl_element().degree() == Q.ufl_element().degree():
    #R = P
#else:
    
Rc = dict((ui, assemble(q*u.dx(i)*dx)) for i, ui in  enumerate(u_components))

reset_sparsity = True

if True:
    u_sol = KrylovSolver('bicgstab', 'jacobi')
    u_sol.parameters['error_on_nonconvergence'] = False
    u_sol.parameters['nonzero_initial_guess'] = True
    u_sol.parameters['monitor_convergence'] = True
    u_sol.parameters['relative_tolerance'] = 1e-7
    u_sol.parameters['absolute_tolerance'] = 1e-10
    
    du_sol = KrylovSolver('bicgstab', 'ilu')
    du_sol.parameters['error_on_nonconvergence'] = False
    du_sol.parameters['nonzero_initial_guess'] = True
    du_sol.parameters['preconditioner']['reuse'] = True
    du_sol.parameters['monitor_convergence'] = True
    du_sol.parameters['relative_tolerance'] = 1e-7
    du_sol.parameters['absolute_tolerance'] = 1e-10

    p_sol = KrylovSolver('gmres', 'ml_amg')
    p_sol.parameters['error_on_nonconvergence'] = False
    p_sol.parameters['nonzero_initial_guess'] = True
    p_sol.parameters['preconditioner']['reuse'] = True
    p_sol.parameters['monitor_convergence'] = True
    p_sol.parameters['relative_tolerance'] = 1e-7
    p_sol.parameters['absolute_tolerance'] = 1e-10
else:
    u_sol = LUSolver()
    u_sol.parameters['reuse_factorization'] = False

    p_sol = LUSolver()
    p_sol.parameters['reuse_factorization'] = True

    du_sol = LUSolver()
    du_sol.parameters['reuse_factorization'] = True

x_  = dict((ui, q_ [ui].vector()) for ui in sys_comp)     # Solution vectors t
x_1 = dict((ui, q_1[ui].vector()) for ui in u_components) # Solution vectors t - dt
x_2 = dict((ui, q_2[ui].vector()) for ui in u_components) # Solution vectors t - 2*dt
b   = dict((ui, Vector(x_[ui])) for ui in sys_comp)       # rhs vectors
bold= dict((ui, Vector(x_[ui])) for ui in sys_comp)       # rhs temp storage vectors
work = Vector(x_['u0'])

t0 = t1 = time.time()
dt_ = dt(0)
total_iters = 0
while t < (T - tstep*DOLFIN_EPS):
    t += dt_
    tstep += 1
    j = 0
    err = 1e8
    total_iters += 1
    
    ### prepare ###
    p_top.assign(splev(t, spline_func))
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
            for ui in u_components:
                b[ui][:] = A*x_1[ui]
            # Reset matrix for lhs
            A._scale(-1.)
            A.axpy(2./dt_, M, True)
            [bc.apply(A) for bc in bcs['u0']]
            
        for ui in u_components:
            bold[ui][:] = b[ui][:]
            b[ui].axpy(-1., P[ui]*x_['pcc'])
            [bc.apply(b[ui]) for bc in bcs[ui]]
            work[:] = x_[ui][:]
            #if u_sol.parameters['monitor_convergence'] and MPI.process_number() == 0:
            if MPI.process_number() == 0:
                print 'Solving tentative ', ui
            u_sol.solve(A, x_[ui], b[ui])
            err += norm(work - x_[ui])
            b[ui][:] = bold[ui][:] # In case of inner iterations
            
        ### Solve pressure ###
        dpcc_.vector()[:] = x_['pcc'][:]
        b['pcc'] = assemble(dp0, exterior_facet_domains=mf)
        b['pcc'].axpy(1., App*x_['pcc'])
        for ui in u_components:
            b['pcc'].axpy(-1./dt_, Rc[ui]*x_[ui]) # Divergence of u_
        rp = residual(Ap, x_['pcc'], b['pcc'])
        #if p_sol.parameters['monitor_convergence'] and MPI.process_number() == 0:
        if MPI.process_number() == 0:
            print 'Solving p' 
            
        p_sol.solve(Ap, x_['pcc'], b['pcc'])
        dpcc_.vector()[:] = x_['pcc'][:] - dpcc_.vector()[:]
        if tstep % check == 0:
            if num_iter > 1:
                if j == 1: info_blue('                 error u  error p')
                info_blue('    Iter = {0:4d}, {1:2.2e} {2:2.2e}'.format(j, err, rp))

    ### Update ################################################################
    ### Update velocity ###
    for ui in u_components:
        b[ui][:] = M*x_[ui][:]        
        b[ui].axpy(-dt_, P[ui]*dpcc_.vector())
        [bc.apply(b[ui]) for bc in bcs[ui]]
        #if du_sol.parameters['monitor_convergence'] and MPI.process_number() == 0:
        if MPI.process_number() == 0:
            print 'Solving ', ui
        du_sol.solve(M, x_[ui], b[ui])

    # Update to a new timestep
    for ui in u_components:
        x_2[ui][:] = x_1[ui][:]
        x_1[ui][:] = x_ [ui][:]
        
    # Print some information and save intermediate solution
    if tstep % check == 0:
        info_red('Total computing time on previous {0:d} timesteps = {1:f}'.format(check, time.time() - t1))
        t1 = time.time()
        info_green('Time = {0:2.4e}, timestep = {1:6d}, End time = {2:2.4e}'.format(t, tstep, T)) 
        newfolder = path.join(folder, 'timestep='+str(tstep))
        u1 = assemble(dot(u_, normal)*ds(2), mesh=mesh, exterior_facet_domains=mf)
        u2 = assemble(dot(u_, normal)*ds(3), mesh=mesh, exterior_facet_domains=mf)
        if MPI.process_number() == 0:
            print 'flux [cm/s] = ', u1/A2, u2/A3, u1, u2
        try:
            makedirs(newfolder)
        except OSError:
            pass
        for ui in sys_comp:
            newfile = File(path.join(newfolder, ui + '.xml.gz'))
            print 'Writing result file ', ui
            newfile << q_[ui]
        
        if tstep % save_restart_file == 0:
            for ui in u_components:
                newfile_1 = File(path.join(newfolder, ui + '_1.xml.gz'))
                newfile_1 << q_1[ui]
    ### Update ################################################################            
info_red('Additional memory use of solver = {0}'.format(eval(getMyMemoryUsage()) - eval(dolfin_memory_use)))
info_red('Total memory use = ' + getMyMemoryUsage())
list_timings()
info_red('Total computing time = {0:f}'.format(time.time()- t0))
#plot(project(u_, Vv))

