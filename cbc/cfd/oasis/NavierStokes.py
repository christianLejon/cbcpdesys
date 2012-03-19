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

#parameters["linear_algebra_backend"] = "Epetra"
parameters["linear_algebra_backend"] = "PETSc"
parameters["form_compiler"]["optimize"]     = False
parameters["form_compiler"]["cpp_optimize"] = True
set_log_active(True)

# Check how much memory is actually used by dolfin before we allocate anything
dolfin_memory_use = getMyMemoryUsage()
info_red('Memory use of plain dolfin = ' + dolfin_memory_use)

################### Problem dependent parameters ####################

mesh = UnitSquare(100, 100)
nu = Constant(0.001)          # Viscosity
t = 0                         # time
tstep = 0                     # Timestep
T = 0.1                       # End time
#T = 2*dt(0)
max_iter = 1                  # Pressure velocity iterations on given timestep
iters_on_first_timestep = 2   # Pressure velocity iterations on first timestep
max_error = 1e-6
#dt = Constant(0.012254901960784314)
dt = Constant(0.25*T/ceil(T/0.2/mesh.hmin())) # timestep
check = 1                     # print out info every check timestep 

# Specify body force
dim = mesh.geometry().dim()
f = Constant((0,)*dim)

#####################################################################

# Declare solution Functions and FunctionSpaces
V = FunctionSpace(mesh, 'CG', 2)
Q = FunctionSpace(mesh, 'CG', 1)
Vv = VectorFunctionSpace(mesh, 'CG', V.ufl_element().degree())
u = TrialFunction(V)
v = TestFunction(V)
p = TrialFunction(Q)
q = TestFunction(Q)

if dim == 2:
    u_components = ['u0', 'u1']
else:
    u_components = ['u0', 'u1', 'u2']
sys_comp =  u_components + ['p']

# Use dictionaries to hold all Functions
q_  = dict((ui, Function(V)) for ui in u_components)
q_1 = dict((ui, Function(V)) for ui in u_components)
q_2 = dict((ui, Function(V)) for ui in u_components)
u_  = as_vector([q_[ui]  for ui in u_components]) # Velocity vector at t
u_1 = as_vector([q_1[ui] for ui in u_components]) # Velocity vector at t - dt
u_2 = as_vector([q_2[ui] for ui in u_components]) # Velocity vector at t - 2*dt

q_['p'] = p_ = Function(Q)  # pressure at t - dt/2
dp_ = Function(Q)      # pressure correction

###################  Boundary conditions  ###########################

bcs = dict((ui, []) for ui in sys_comp)

# Driven cavity example:
def lid(x, on_boundary):
    return (on_boundary and near(x[1], 1.0))
    
def stationary_walls(x, on_boundary):
    return on_boundary and (near(x[0], 0.) or near(x[0], 1.) or near(x[1], 0.))

bc0  = DirichletBC(V, 0., stationary_walls)
bc00 = DirichletBC(V, 1., lid)
bc01 = DirichletBC(V, 0., lid)
bcs['u0'] = [bc00, bc0]
bcs['u1'] = [bc01, bc0]

# Normalize pressure or not?
#normalize = False

#####################################################################

# Preassemble some constant in time matrices
M = assemble(inner(u, v)*dx)                    # Mass matrix
K = assemble(nu*inner(grad(u), grad(v))*dx)     # Diffusion matrix
Ap = assemble(inner(grad(q), dt*grad(p))*dx)    # Pressure Laplacian
A = Matrix()                                    # Coefficient matrix (needs reassembling)

# Apply boundary conditions on M and Ap that are used directly in solve
[bc.apply(M)  for bc in bcs['u0']]
[bc.apply(Ap) for bc in bcs['p']]

# Adams Bashforth projection of velocity at t - dt/2
U_ = 1.5*u_1 - 0.5*u_2

# Convection form
a  = 0.5*inner(v, dot(U_, nabla_grad(u)))*dx

# Preassemble constant body force
assert(isinstance(f, Constant))
b0 = dict((ui, assemble(v*f[i]*dx)) for i, ui in enumerate(u_components))

# Preassemble constant pressure gradient matrix
P = dict((ui, assemble(v*p.dx(i)*dx)) for i, ui in enumerate(u_components))

# Preassemble velocity divergence matrix
if V.ufl_element().degree() == Q.ufl_element().degree():
    R = P
else:
    R = dict((ui, assemble(q*u.dx(i)*dx)) for i, ui in  enumerate(u_components))

# Set up linear solvers
#u_sol = LUSolver()

#du_sol = LUSolver()
#du_sol.parameters['reuse_factorization'] = True

#p_sol = LUSolver()
#p_sol.parameters['reuse_factorization'] = True
list_timings()

u_sol = KrylovSolver('gmres', 'jacobi')
u_sol.parameters['error_on_nonconvergence'] = False
u_sol.parameters['nonzero_initial_guess'] = True
#u_sol.parameters['monitor_convergence'] = True
reset_sparsity = True

du_sol = KrylovSolver('gmres', 'jacobi')
du_sol.parameters['error_on_nonconvergence'] = False
du_sol.parameters['nonzero_initial_guess'] = True
du_sol.parameters['preconditioner']['reuse'] = True

p_sol = KrylovSolver('gmres', 'hypre_amg')
p_sol.parameters['error_on_nonconvergence'] = False
p_sol.parameters['nonzero_initial_guess'] = True
p_sol.parameters['preconditioner']['reuse'] = True

x_  = dict((ui, q_ [ui].vector()) for ui in sys_comp)     # Solution vectors t
x_1 = dict((ui, q_1[ui].vector()) for ui in u_components) # Solution vectors t - dt
x_2 = dict((ui, q_2[ui].vector()) for ui in u_components) # Solution vectors t - 2*dt
b   = dict((ui, Vector(x_[ui])) for ui in sys_comp)       # rhs vectors
bold= dict((ui, Vector(x_[ui])) for ui in sys_comp)       # rhs temp storage vectors
work = Vector(x_['u0'])

t0 = time.time()
dt_ = dt(0)
total_iters = 0
while t < (T - tstep*DOLFIN_EPS):
    t += dt_
    tstep += 1
    j = 0
    err = 1e8
    total_iters += 1
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
                b[ui][:] = b0[ui][:]
                b[ui].axpy(1., A*x_1[ui])

            # Reset matrix for lhs
            A._scale(-1.)
            A.axpy(2./dt_, M, True)
            [bc.apply(A) for bc in bcs['u0']]
        
        for ui in u_components:
            bold[ui][:] = b[ui][:] 
            b[ui].axpy(-1., P[ui]*x_['p'])
            [bc.apply(b[ui]) for bc in bcs[ui]]
            work[:] = x_[ui][:]
            u_sol.solve(A, x_[ui], b[ui])
            b[ui][:] = bold[ui][:]  # preassemble part
            err += norm(work - x_[ui])
            
        ### Solve pressure ###
        dp_.vector()[:] = x_['p'][:]
        b['p'][:] = Ap*x_['p']
        for ui in u_components:
            b['p'].axpy(-1., R[ui]*x_[ui]) # Divergence of u_
        [bc.apply(b['p']) for bc in bcs['p']]
        rp = residual(Ap, x_['p'], b['p'])
        p_sol.solve(Ap, x_['p'], b['p'])
        if normalize: normalize(x_['p'])
        dp_.vector()[:] = x_['p'][:] - dp_.vector()[:]
        if tstep % check == 0:
            if num_iter > 1:
                if j == 1: info_blue('                 error u  error p')
                info_blue('    Iter = {0:4d}, {1:2.2e} {2:2.2e}'.format(j, err, rp))

    ### Update velocity ###
    for ui in u_components:
        b[ui][:] = M*x_[ui][:]        
        b[ui].axpy(-dt_, P[ui]*dp_.vector())
        [bc.apply(b[ui]) for bc in bcs[ui]]        
        du_sol.solve(M, x_[ui], b[ui])

    # Update to a new timestep
    for ui in u_components:
        x_2[ui][:] = x_1[ui][:]
        x_1[ui][:] = x_ [ui][:]
        
    # Print some information
    if tstep % check == 0:
        info_green('Time = {0:2.4e}, timestep = {1:6d}, End time = {2:2.4e}'.format(t, tstep, T)) 

info_red('Total computing time = {0:f}'.format(time.time()- t0))
print 'Additional memory use of processor = {0}'.format(eval(getMyMemoryUsage()) - eval(dolfin_memory_use))
mymem = eval(getMyMemoryUsage())-eval(dolfin_memory_use)
info_red('Total memory use of solver = ' + str(comm.reduce(mymem, root=0)))
list_timings()
#plot(project(u_, Vv)) 
plot(p_, interactive=True)


