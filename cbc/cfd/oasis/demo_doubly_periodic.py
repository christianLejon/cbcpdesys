# Copyright (C) 2013 Mikael Mortensen
#
# This file is part of DOLFIN.
#
# DOLFIN is free software: you can redistribute it and/or modify
# it under the terms of the GNU Lesser General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# DOLFIN is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
# GNU Lesser General Public License for more details.
#
# You should have received a copy of the GNU Lesser General Public License
# along with DOLFIN. If not, see <http://www.gnu.org/licenses/>.
#
# First added:  
# Last changed: 
"""
This is a highly tuned and stripped down Navier-Stokes solver optimized
for both speed and memory. The algorithm used is a second order in time 
fractional step method (incremental pressure correction).

Crank-Nicolson discretization is used in time of the Laplacian and 
the convected velocity. The convecting velocity is computed with an 
Adams-Bashforth projection. The fractional step method can be used
both non-iteratively or with iterations over the pressure velocity 
system.

The velocity vector is segregated, and we use three velocity components

V = FunctionSpace(mesh, 'CG', 1)
u0, u1, u2 = Function(V), Function(V), Function(V)

A single coefficient matrix is assembled and used by all velocity 
componenets. It is built by preassembling as much as possible. 

The form for the tentative velocity is
U = 0.5*(u+u_1)
U1 = 1.5*u_1 - 0.5*u_2
F = (1/dt)*inner(u - u_1, v)*dx + inner(grad(U)*U1, v)*dx + inner(grad(p_), v)*dx \
     nu*inner(grad(U), grad(v))*dx - inner(dpdx, v)*dx
     
where u_1 and u_2 are velocities at time steps k-1 and k-2. We are 
solving for u, which is the velocity at time step k. p_ is the latest 
approximation for the pressure.

The matrix corresponding to assemble(lhs(F)) is computed as:

    A  = 1/dt*M + 0.5*Ac + 0.5*K
    
where

    M  = assemble(inner(u, v)*dx)
    Ac = assemble(inner(grad(u)*U1, v)*dx)
    K  = assemble(nu*inner(grad(u), grad(v))*dx)

However, we start by assembling a coefficient matrix (Ar) that is used 
to compute parts of the rhs vector corresponding to mass, convection
and diffusion:

    Ar = 1/dt*M - 0.5*Ac - 0.5*K
    b  = A*u_1.vector()

The pressure gradient and body force needs to be added to b as well. Three
matrices are preassembled for the computation of the pressure gradient:

u_components = ["u0", "u1", "u2"]
P = dict((ui, assemble(v*p.dx(i)*dx)) for i, ui in enumerate(u_components))

and the pressure gradient for each component of the momentum equation is 
then computed as

dpdx = P["u0"] * p_.vector()
dpdy = P["u1"] * p_.vector()
dpdz = P["u2"] * p_.vector()

Ac needs to be reassembled each new timestep. Ac is assembled into A to 
save memory. A and Ar are recreated each timestep by assembling Ac, setting 
up Ar and then using the following to create A:

   A = - Ar + 2/dt*M

"""
from cbc.cfd.oasis import *
from numpy import arctan, array
import random
from os import getpid, path, makedirs, getcwd, listdir
import time
from cbc.cfd.tools.Probe import StructuredGrid

parameters["form_compiler"]["optimize"]     = True   # I sometimes get memory access error with True here (MM)
parameters["form_compiler"]["cpp_optimize"] = True
parameters['mesh_partitioner'] = "ParMETIS"
#parameters["std_out_all_processes"] = False

#dolfin_memory_use = getMyMemoryUsage()
#info_red('Memory use of plain dolfin = ' + dolfin_memory_use)

################### Problem dependent parameters ####################

Lx = 4.
Ly = 2.
Lz = 2.
Nx = 250
Ny = 200
Nz = 125
mesh = BoxMesh(0., -Ly/2., -Lz/2., Lx, Ly/2., Lz/2., Nx, Ny, Nz)
# Create stretched mesh in y-direction
x = mesh.coordinates()        
x[:, 1] = arctan(pi*(x[:, 1]))/arctan(pi) 
normal = FacetNormal(mesh)

class PeriodicDomain(SubDomain):

    def inside(self, x, on_boundary):
        # return True if on left or bottom boundary AND NOT on one of the two slave edges
        return bool((near(x[0], 0) or near(x[2], -Lz/2.)) and 
                (not ((near(x[0], Lx) and near(x[2], -Lz/2.)) or 
                      (near(x[0], 0) and near(x[2], Lz/2.)))) and on_boundary)
                      
    def map(self, x, y):
        if near(x[0], Lx) and near(x[2], Lz/2.):
            y[0] = x[0] - Lx
            y[1] = x[1] 
            y[2] = x[2] - Lz
        elif near(x[0], Lx):
            y[0] = x[0] - Lx
            y[1] = x[1]
            y[2] = x[2]
        elif near(x[2], Lz/2.):
            y[0] = x[0]
            y[1] = x[1]
            y[2] = x[2] - Lz
        else:
            y[0] = -1000
            y[1] = -1000
            y[2] = -1000
            
pd = PeriodicDomain()

# Set parameters
Re_tau = 395.
nu = Constant(2.e-5)           # Viscosity
utau = nu(0) * Re_tau
t = 0.0                        # time
tstep = 0                      # Timestep
T = 500.                        # End time
max_iter = 1                   # Pressure velocity iterations on given timestep
iters_on_first_timestep = 2    # Pressure velocity iterations on first timestep
max_error = 1e-6
check = 500                     # print out info and save solution every check timestep 
save_vtk = 100000
save_restart_file = 2000        # Saves two previous timesteps needed for a clean restart
    
# Specify body force
dim = mesh.geometry().dim()

# Set the timestep
dt = Constant(0.05)
n = int(T / dt(0))

# Give a folder for storing the results
folder = "channel_result"

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
        makedirs(vtkfolder)
    except:
        pass
    try:
        makedirs(statsfolder)
    except:
        pass
vtk_file = File(path.join(vtkfolder, "u.pvd"))
u_stats_file = File(path.join(statsfolder, "umean.xml.gz"))
k_stats_file = File(path.join(statsfolder, "kmean.xml.gz"))

#### Set a folder that contains xml.gz files of the solution. 
#restart_folder = None        
restart_folder = "/usit/abel/u1/mikaem/Fenics/cbcpdesys/cbc/cfd/channel_result/data/dt=5.0000e-02/4/timestep=12000/"    
#####################################################################
# Declare solution Functions and FunctionSpaces
V = FunctionSpace(mesh, 'CG', 1, constrained_domain=pd)
Q = FunctionSpace(mesh, 'CG', 1, constrained_domain=pd)
Vv = VectorFunctionSpace(mesh, 'CG', V.ufl_element().degree(), constrained_domain=pd)
u = TrialFunction(V)
v = TestFunction(V)
p = TrialFunction(Q)
q = TestFunction(Q)

u_components = ['u0', 'u1', 'u2']
sys_comp =  u_components + ['p']

# Compute average values for some planes
class ChannelGrid(StructuredGrid):
    def create_grid(self):
        """Create grid skewed towards the walls"""
        x = StructuredGrid.create_grid(self)
        x[:, 1] = arctan(0.5*pi*(x[:, 1]))/arctan(0.5*pi)  
        return x
        
tol = 1e-12
slx = ChannelGrid([250, 200], [tol, -Ly/2.+tol, tol],     [[1., 0., 0.], [0., 1., 0.]], [Lx-2*tol, Ly-2*tol], V, statistics=True)
slz = ChannelGrid([200, 125], [2., -Ly/2.+tol, -Lz/2.+tol], [[0., 1., 0.], [0., 0., 1.]], [Ly-2*tol, Lz-2*tol], V, statistics=True)

# Use dictionaries to hold all Functions and FunctionSpaces
VV = dict((ui, V) for ui in u_components); VV['p'] = Q

# Start from previous solution if restart_folder is given
if restart_folder:
    q_  = dict((ui, Function(VV[ui], path.join(restart_folder, ui + '.xml.gz'))) for ui in sys_comp)
    q_1 = dict((ui, Function(V, path.join(restart_folder, ui + '.xml.gz'))) for ui in u_components)
    q_2 = dict((ui, Function(V, path.join(restart_folder, ui + '_1.xml.gz'))) for ui in u_components)
else:
    q_  = dict((ui, Function(VV[ui])) for ui in sys_comp)
    q_1 = dict((ui, Function(V)) for ui in u_components)
    q_2 = dict((ui, Function(V)) for ui in u_components)

class RandomStreamVector(Expression):
    def __init__(self):
        random.seed(2 + MPI.process_number())
    def eval(self, values, x):
        values[0] = 0.002*random.random()
        values[1] = 0.002*random.random()
        values[2] = 0.002*random.random()
    def value_shape(self):
        return (3,)        

if restart_folder == None:    
    psi = interpolate(RandomStreamVector(), Vv)
    u0 = project(curl(psi), Vv)
    u0x = project(u0[0], V)
    u1x = project(u0[1], V)
    u2x = project(u0[2], V)
    y = interpolate(Expression("0.1335*((1+x[1])*(1-x[1]))"), V)
    q_['u0'].vector()[:] = y.vector()[:] 
    q_['u0'].vector().axpy(1.0, u0x.vector())
    q_['u1'].vector()[:] = u1x.vector()[:]
    q_['u2'].vector()[:] = u2x.vector()[:]
    q_1['u0'].vector()[:] = q_['u0'].vector()[:]
    q_2['u0'].vector()[:] = q_['u0'].vector()[:]
    q_1['u1'].vector()[:] = q_['u1'].vector()[:]
    q_2['u1'].vector()[:] = q_['u1'].vector()[:]
    q_1['u2'].vector()[:] = q_['u2'].vector()[:]
    q_2['u2'].vector()[:] = q_['u2'].vector()[:]

u_  = as_vector([q_[ui]  for ui in u_components]) # Velocity vector at t
u_1 = as_vector([q_1[ui] for ui in u_components]) # Velocity vector at t - dt
u_2 = as_vector([q_2[ui] for ui in u_components]) # Velocity vector at t - 2*dt
p_ = q_['p']
dp_ = Function(Q)      # pressure correction

###################  Boundary conditions  ###########################

bcs = dict((ui, []) for ui in sys_comp)

def walls(x, on_bnd):
    return on_bnd and (near(x[1], -Ly/2.) or near(x[1], Ly/2.))
    
# Preassemble constant pressure gradient matrix
P = dict((ui, assemble(v*p.dx(i)*dx)) for i, ui in enumerate(u_components))

# Preassemble velocity divergence matrix
#Rx = dict((ui, assemble(q*u.dx(i)*dx)) for i, ui in  enumerate(u_components))
Rx = P

bc = [DirichletBC(V, Constant(0), walls)]
bcs['u0'] = bc
bcs['u1'] = bc
bcs['u2'] = bc
bcs['p'] = []

#####################################################################

# Preassemble some constant in time matrices
M = assemble(inner(u, v)*dx)                    # Mass matrix
K = assemble(nu*inner(grad(u), grad(v))*dx)     # Diffusion matrix
Ap = assemble(inner(grad(q), grad(p))*dx)    # Pressure Laplacian
A = Matrix()                                    # Coefficient matrix (needs reassembling)

# Apply boundary conditions on M and Ap that are used directly in solve
[bc.apply(M) for bc in bcs['u0']]
Ap.compress()

# Create vectors used for lumping mass matrix
ones = Vector(q_['u0'].vector())
ones[:] = 1.
ML = M * ones
MP = Vector(ML)
ML.set_local(1. / ML.array())

# Adams Bashforth projection of velocity at t - dt/2
U_ = 1.5*u_1 - 0.5*u_2

# Convection form
a  = 0.5*inner(v, dot(U_, nabla_grad(u)))*dx

#u_sol = LUSolver()
u_sol = KrylovSolver('bicgstab', 'jacobi')
u_sol.parameters['error_on_nonconvergence'] = False
u_sol.parameters['nonzero_initial_guess'] = True
u_sol.parameters['preconditioner']['reuse'] = False
u_sol.parameters['monitor_convergence'] = True
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
du_sol.parameters['monitor_convergence'] = True
du_sol.parameters['maximum_iterations'] = 50
#du_sol.parameters['relative_tolerance'] = 1e-9
#du_sol.parameters['absolute_tolerance'] = 1e-10
du_sol.t = 0

#p_sol = LUSolver()
p_sol = KrylovSolver('gmres', 'hypre_amg')
p_sol.parameters['error_on_nonconvergence'] = True
p_sol.parameters['nonzero_initial_guess'] = True
p_sol.parameters['preconditioner']['reuse'] = True
p_sol.parameters['monitor_convergence'] = True
p_sol.parameters['maximum_iterations'] = 50
p_sol.parameters['relative_tolerance'] = 1e-7*dt(0)
p_sol.parameters['absolute_tolerance'] = 1e-7*dt(0)
p_sol.t = 0

x_  = dict((ui, q_ [ui].vector()) for ui in sys_comp)     # Solution vectors t
x_1 = dict((ui, q_1[ui].vector()) for ui in u_components) # Solution vectors t - dt
x_2 = dict((ui, q_2[ui].vector()) for ui in u_components) # Solution vectors t - 2*dt
b   = dict((ui, Vector(x_[ui])) for ui in sys_comp)       # rhs vectors
bold= dict((ui, Vector(x_[ui])) for ui in sys_comp)       # rhs temp storage vectors
work = Vector(x_['u0'])

# Preassemble constant pressure gradient
dpdx = (utau**2, 0., 0.)
b0 = dict((ui, assemble(v*Constant(dpdx[i])*dx)) for i, ui in enumerate(u_components))

u0mean = Function(V)
kmean = Function(V)

#ff = open(path.join(folder, "{}_memoryuse{}.txt".format(MPI.process_number(), getMyMemoryUsage())), "w")
#ff.close()

t0 = t1 = time.time()
dt_ = dt(0)
total_iters = 0
tstep0 = tstep
slx.t = 0
ttot=time.time()
while t < T + DOLFIN_EPS:
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
        t0 = time.time()
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
            
        for ui in u_components:
            bold[ui][:] = b[ui][:]
            b[ui].axpy(-1., P[ui]*x_['p'])
            [bc.apply(b[ui]) for bc in bcs[ui]]
            work[:] = x_[ui][:]
            if MPI.process_number() == 0:
                print 'Solving tentative ', ui
            if ui == "u0":
                u_sol.parameters['preconditioner']['reuse'] = False
            else:
                u_sol.parameters['preconditioner']['reuse'] = True
            u_sol.solve(A, x_[ui], b[ui])
            err += norm(work - x_[ui])
            b[ui][:] = bold[ui][:]
        u_sol.t += (time.time()-t0)
            
        ### Solve pressure ###
        t0 = time.time()
        dp_.vector()[:] = x_['p'][:]
        b['p'][:] = 0.
        for ui in u_components:
            b['p'].axpy(-1./dt_, Rx[ui]*x_[ui]) # Divergence of u_
        b['p'].axpy(1., Ap*x_['p'])
        rp = residual(Ap, x_['p'], b['p'])
        if MPI.process_number() == 0:
            print 'Solving p'        
        p_sol.solve(Ap, x_['p'], b['p'])
        normalize(p_.vector())
        dp_.vector()[:] = x_['p'][:] - dp_.vector()[:]
        if tstep % check == 0:
            if num_iter > 1:
                if MPI.process_number()==0: 
                    if j == 1:
                        info_blue('                 error u  error p')
                    info_blue('    Iter = {0:4d}, {1:2.2e} {2:2.2e}'.format(j, err, rp))
        p_sol.t += (time.time()-t0)

    ### Update ################################################################
    ### Update velocity ###
    #for ui in u_components:
        #b[ui][:] = M*x_[ui][:]        
        #b[ui].axpy(-dt_, P[ui]*dpc_.vector())
        #[bc.apply(b[ui]) for bc in bcs[ui]]
        ##if du_sol.parameters['monitor_convergence'] and MPI.process_number() == 0:
        #if MPI.process_number() == 0:  
            #print 'Solving ', ui
        #t0 = time.time()
        #du_sol.solve(M, x_[ui], b[ui])
        #du_sol.t += (time.time()-t0)

    # Update velocity using lumped mass matrix
    t0 = time.time()
    for ui in u_components:
        MP[:] = (P[ui] * dp_.vector()) * ML
        x_[ui].axpy(-dt_, MP)
        bcs[ui][0].apply(x_[ui])
    du_sol.t += (time.time()-t0)
    
    t0 = time.time()
    # Update solution to a new timestep
    for ui in u_components:
        x_2[ui][:] = x_1[ui][:]
        x_1[ui][:] = x_ [ui][:]

    # Update statistics
    u0mean.vector().axpy(1.0, x_['u0'])
    kmean.vector().axpy(1.0, 0.5*(x_['u0']*x_['u0'] + x_['u1']*x_['u1'] + x_['u2']*x_['u2']))
    slx(q_['u0'], q_['u1'], q_['u2'])
    slz(q_['u0'], q_['u1'], q_['u2'])
    slx.t += time.time() - t0
    #################################################################################################
    
    # Print some information and save intermediate solution
    if tstep % check == 0:
        list_timings(True)
        if MPI.process_number()==0:
            tottime= time.time() - t1
            info_red('Total computing time on previous {0:d} timesteps = {1:f}'.format(check, tottime))
        if MPI.process_number()==0:
            info_green('Time = {0:2.4e}, timestep = {1:6d}, End time = {2:2.4e}'.format(t, tstep, T)) 
        newfolder = path.join(folder, 'timestep='+str(tstep))        
        #plot(u_[0], rescale=True)
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

        slx.tovtk(0, filename=statsfolder+"/dump_mean_x_{}.vtk".format(tstep))
        slz.tovtk(0, filename=statsfolder+"/dump_mean_z_{}.vtk".format(tstep))
        slx.tovtk(1, filename=newfolder+"/snapshot_x_{}.vtk".format(tstep))
        slz.tovtk(1, filename=newfolder+"/snapshot_z_{}.vtk".format(tstep))
        t1 = time.time()
        if MPI.process_number()==0:
            ff = open(newfolder+"/timeprstep_{}.txt".format(tottime/check), "w")
            ff.close()

    ### Update ################################################################            
u0mean.vector()._scale(1./(tstep-tstep0))
kmean.vector()._scale(1./(tstep-tstep0))
u_stats_file << u0mean
k_stats_file << kmean
slx.tovtk(0, filename=statsfolder+"/dump_mean_x.vtk")
slz.tovtk(0, filename=statsfolder+"/dump_mean_z.vtk")

list_timings()
if MPI.process_number()==0:
    info_red('Total computing time = {0:f}'.format(time.time()- ttot))
    print 'u_sol ', u_sol.t
    print 'du_sol ', du_sol.t
    print 'p_sol ', p_sol.t

#plot(u_[0])
#interactive()
