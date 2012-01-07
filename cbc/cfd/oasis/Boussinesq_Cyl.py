__author__ = "Mikael Mortensen <mikaem@math.uio.no>"
__date__ = "2011-12-19"
__copyright__ = "Copyright (C) 2011 " + __author__
__license__  = "GNU GPL version 3 or any later version"

"""
This is a highly tuned and stripped down Navier-Stokes solver for
natural convection problems. The Boussinesq model is used as a body 
force in the momentum equation and we solve for the temperature.

To use this solver for a new problem you have to modify two sections.

    'Problem dependent parameters':
        add a mesh and set problem specific parameters. 
        Parameters that are recognized are already given 
        default values.

    'Boundary conditions':
        Simply create the dictionary bcs, with keys 
        'u0', 'u1', 'u2', 'p' and 'c' and value a list of 
        boundary conditions. 
        If necessary enable normalization of the pressure.
        
The algorithm used is a second order in time fractional step method
(incremental pressure correction).

We use a Crank-Nicolson discretization in time of the Laplacian and 
the convected velocity. The convecting velocity is computed with an
iterative midpoint 0.5*(u_ + u_1). 

    inner(v, dot(0.5*(u_+u_1), grad(u)))*dx

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
   
For temperature we use
    C = 0.5*(c + c_1)
    U_ = 0.5*(u_ + u_1)
    F = (1./dt)*inner(c - c_1, v_c)*dx + inner(dot(U_, grad(C)), v_c)*dx \
        + nu/Pr*inner(grad(C), grad(v_c))*dx 

    T = 1/dt*M + 0.5*Ac + 0.5/Pr*K  = lhs matrix
    
    Tr = 1/dt*M - 0.5*Ac - 0.5/Pr*K = rhs matrix
    
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

mesh = Mesh('../data/11116_fenics_2d.xml')
nu = Constant(1.e-6)          # Viscosity
t = 0                         # time
tstep = 0                     # Timestep
T = 50.                       # End time
#T = 2*dt(0)
max_iter = 10                  # Iterations on timestep
max_error = 1e-6
dt = Constant(0.05)
#dt = Constant(T/ceil(T/0.2/mesh.hmin())) # timestep
check = 1                     # print out info every check timestep 
T0 = Constant(294.)
T1 = Constant(317.)
rho0 = Constant(1000.)
beta = Constant(200.e-5)      # beta*g
Pr = Constant(7.)
restart_files = '.xml.gz'     # if None then we initialize with zero
#restart_files = False
save_solution = '.xml.gz'

#####################################################################

# Declare solution Functions and FunctionSpaces
V = FunctionSpace(mesh, 'CG', 2)
Q = FunctionSpace(mesh, 'CG', 1)
Vv = VectorFunctionSpace(mesh, 'CG', V.ufl_element().degree())
u = TrialFunction(V)
v = TestFunction(V)
p = TrialFunction(Q)
q = TestFunction(Q)
dim = mesh.geometry().dim()
n = FacetNormal(mesh)

if dim == 2:
    u_components = ['u0', 'u1']
else:
    u_components = ['u0', 'u1', 'u2']
sys_comp =  u_components + ['p'] + ['c']
uc_comp  =  u_components + ['c']

# Use dictionaries to hold all Functions
if restart_files:
    q_  = dict((ui, Function(V, ui + restart_files)) for ui in uc_comp)
    q_1 = dict((ui, Function(V, ui + '_1' + restart_files)) for ui in uc_comp)
else:
    q_  = dict((ui, Function(V)) for ui in uc_comp)
    q_1 = dict((ui, Function(V)) for ui in uc_comp)
u_  = as_vector([q_[ui]  for ui in u_components]) # Velocity vector at t
u_1 = as_vector([q_1[ui] for ui in u_components]) # Velocity vector at t - dt

# pressure at t - dt/2
if restart_files:
    q_['p'] = p_ = Function(Q, 'p' + restart_files)  
else:
    q_['p'] = p_ = Function(Q)  # pressure at t - dt/2
    
dp_ = Function(Q)           # pressure correction

c_  = q_ ['c']              # Short forms
c_1 = q_1['c']

###################  Boundary conditions  ###########################

bcs = dict((ui, []) for ui in sys_comp)

def on_cylinder(x, on_boundary):
    return (not (near(x[0], -0.5) or near(x[1], -0.5) 
            or near(x[0], 0.5) or near(x[1], 1.))
            and on_boundary)

def walls(x, on_boundary):
    return ((near(x[0], -0.5) or near(x[1], -0.5)
            or near(x[0], 0.5) or near(x[1], 1.))
            and on_boundary)

Cylinder = AutoSubDomain(on_cylinder)
Walls = AutoSubDomain(walls)

bc0 = DirichletBC(V, 0., Walls)
bc1 = DirichletBC(V, 0., Cylinder)
bcc = DirichletBC(V, T1, Cylinder)
bcc2 = DirichletBC(V, T0, Walls)
bcs['u0'] = [bc0, bc1]
bcs['u1'] = [bc0, bc1]
bcs['c'] = [bcc]
# Normalize pressure or not?
#normalize = False
normalize = dolfin_normalize(Q)

# Specify body force. 
C_ = 0.5*(c_ + c_1)
f  = dict(u0=Constant(0), u1=beta*(C_ - T0))

# Initialize temperature
if not restart_files:
    c_ .vector()[:] = T0(0)
    c_1.vector()[:] = T0(0)
    bcc.apply(c_ .vector())   # Apply wall temperature on walls
    bcc.apply(c_1.vector())

#####################################################################

# Preassemble some constant in time matrices
M = assemble(inner(u, v)*dx)                    # Mass matrix
K = assemble(nu*inner(grad(u), grad(v))*dx)     # Diffusion matrix
Ap = assemble(inner(grad(q), dt*grad(p))*dx)    # Pressure Laplacian

Ac = Matrix()                                   # Convection
A = Matrix(K)                                   # Coefficient matrix (needs reassembling)
Ta = Matrix(K)                                  # Coefficient matrix (needs reassembling)
Mu = Matrix(M)

# Apply boundary conditions on Mu and Ap that are used directly in solve
[bc.apply(Mu) for bc in bcs['u0']]
[bc.apply(Ap) for bc in bcs['p']]

# Midpoint velocity at t - dt/2
U_ = 0.5*(u_ + u_1)

# Convection form
a  = 0.5*inner(v, dot(U_, nabla_grad(u)))*dx

# Preassemble constant pressure gradient matrix
P = dict((ui, assemble(v*p.dx(i)*dx)) for i, ui in enumerate(u_components))

# Preassemble velocity divergence matrix
R = dict((ui, assemble(q*u.dx(i)*dx)) for i, ui in  enumerate(u_components))

# Set up linear solvers
#u_sol = LUSolver()

#du_sol = LUSolver()
#du_sol.parameters['reuse_factorization'] = True

#p_sol = LUSolver()
#p_sol.parameters['reuse_factorization'] = True

u_sol = KrylovSolver('bicgstab', 'jacobi')
u_sol.parameters['error_on_nonconvergence'] = False
u_sol.parameters['nonzero_initial_guess'] = True
#u_sol.parameters['monitor_convergence'] = True
u_sol.parameters['relative_tolerance'] = 1e-13
u_sol.parameters['absolute_tolerance'] = 1e-20
reset_sparsity = True

du_sol = KrylovSolver('bicgstab', 'ilu')
du_sol.parameters['error_on_nonconvergence'] = False
du_sol.parameters['nonzero_initial_guess'] = True
du_sol.parameters['preconditioner']['reuse'] = True
#du_sol.parameters['monitor_convergence'] = True
du_sol.parameters['relative_tolerance'] = 1e-13
du_sol.parameters['absolute_tolerance'] = 1e-20

p_sol = KrylovSolver('gmres', 'hypre_amg')
p_sol.parameters['error_on_nonconvergence'] = False
p_sol.parameters['nonzero_initial_guess'] = True
p_sol.parameters['preconditioner']['reuse'] = True
#p_sol.parameters['monitor_convergence'] = True
p_sol.parameters['relative_tolerance'] = 1e-13
p_sol.parameters['absolute_tolerance'] = 1e-20

c_sol = KrylovSolver('bicgstab', 'ilu')
c_sol.parameters['error_on_nonconvergence'] = False
c_sol.parameters['nonzero_initial_guess'] = True
#c_sol.parameters['monitor_convergence'] = True
c_sol.parameters['relative_tolerance'] = 1e-13
c_sol.parameters['absolute_tolerance'] = 1e-20

x_  = dict((ui, q_ [ui].vector()) for ui in sys_comp)     # Solution vectors t
x_1 = dict((ui, q_1[ui].vector()) for ui in uc_comp)      # Solution vectors t - dt
b   = dict((ui, Vector(x_[ui])) for ui in sys_comp)       # rhs vectors
bold= dict((ui, Vector(x_[ui])) for ui in sys_comp)       # rhs temp storage vectors
work = Vector(x_['u0'])
error = dict((ui, 0.) for ui in sys_comp)

t0 = time.time()
dt_ = dt(0)
total_iters = 0
while t < (T - tstep*DOLFIN_EPS):
    t += dt_
    tstep += 1
    j = 0
    err = 1e8
    total_iters += 1
    while err > max_error and j < max_iter:
        err = 0
        j += 1
        ### Start by solving for an intermediate velocity ###
        # Set up coefficient matrix for computing the rhs:
        Ac = assemble(a, tensor=Ac, reset_sparsity=reset_sparsity) 
        reset_sparsity = False   # Warning! Must be true for periodic boundary conditions
        A._scale(0.)
        A.axpy(-rho0(0), Ac, True)    # Negative convection on the rhs
        A.axpy(rho0(0)/dt_, M, True)  # Add mass
        A.axpy(-0.5*rho0(0), K, True)    # Add diffusion      
        
        # Compute rhs for all velocity components
        for ui in u_components:
            b[ui][:] = A*x_1[ui]
            if ui == 'u1':
                fv = assemble(inner(v, rho0*f[ui])*dx)
                b[ui].axpy(1., fv)

        # Reset matrix for lhs
        A._scale(-1.)
        A.axpy(2.*rho0(0)/dt_, M, True)
        [bc.apply(A) for bc in bcs['u0']]
        
        for ui in u_components:
            #bold[ui][:] = b[ui][:] 
            b[ui].axpy(-1., P[ui]*x_['p'])
            [bc.apply(b[ui]) for bc in bcs[ui]]
            work[:] = x_[ui][:]
            u_sol.solve(A, x_[ui], b[ui])
            #b[ui][:] = bold[ui][:]  # preassemble part
            error[ui] = norm(work - x_[ui])
            err += error[ui]             
            
        ### Solve pressure ###
        dp_.vector()[:] = x_['p'][:]
        b['p'][:] = Ap*x_['p']
        for ui in u_components:
            b['p'].axpy(-rho0(0), R[ui]*x_[ui]) # Divergence of u_
        [bc.apply(b['p']) for bc in bcs['p']]
        error['p'] = residual(Ap, x_['p'], b['p'])
        p_sol.solve(Ap, x_['p'], b['p'])
        if normalize: normalize(x_['p'])
        dp_.vector()[:] = x_['p'][:] - dp_.vector()[:]
        err += error['p']

        ### Update velocity ###
        for ui in u_components:
            b[ui][:] = Mu*x_[ui][:]        
            b[ui].axpy(-dt_/rho0(0), P[ui]*dp_.vector())
            [bc.apply(b[ui]) for bc in bcs[ui]]        
            du_sol.solve(Mu, x_[ui], b[ui])
        
        Ta._scale(0.)
        Ta.axpy(-1., Ac, True)       # Negative convection on the rhs
        Ta.axpy(1./dt_, M, True)     # Add mass
        Ta.axpy(-0.5/Pr(0), K, True) # Add diffusion   
        b['c'][:] = Ta*x_1['c']      # Compute rhs
        Ta._scale(-1.)
        Ta.axpy(2./dt_, M, True)
        [bc.apply(Ta, b['c']) for bc in bcs['c']]
        work[:] = x_['c']
        c_sol.solve(Ta, x_['c'], b['c'])
        error['c'] = norm(x_['c']- work)
        err += error['c']
        
        if tstep % check == 0:
            if j == 1: info_blue('                 error u  error p  error c')
            info_blue('    Iter = {0:4d}, {1:2.2e} {2:2.2e} {3:2.2e}'.format(j, err, error['p'], error['c']))
        
    # Update to a new timestep
    for ui in uc_comp:
        x_1[ui][:] = x_ [ui][:]
        
    if tstep % check == 0:
        #plot(project(u_, Vv))
        plot(c_)
        
    # Print some information
    if tstep % check == 0:
        info_green('Time = {0:2.4e}, timestep = {1:6d}, End time = {2:2.4e}'.format(t, tstep, T)) 

info_red('Additional memory use of solver = {0}'.format(eval(getMyMemoryUsage()) - eval(dolfin_memory_use)))
info_red('Total memory use = ' + getMyMemoryUsage())
list_timings()
#plot(project(u_, Vv))    
info_red('Total computing time = {0:f}'.format(time.time()- t0))

if save_solution:
    for ui in sys_comp:
        f0 = File(ui + save_solution)        
        f0 << q_[ui]
        if not ui == 'p':
            f_1 = File(ui + '_1' + save_solution)
            f_1 << q_1[ui]
