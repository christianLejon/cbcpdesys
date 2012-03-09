#from dolfin import *
from cbc.pdesys import *
from LagrangianParticles import zalesak, LagrangianParticles
from numpy import zeros
import time
from copy import deepcopy
from mpi4py import MPI as nMPI
from numpy.linalg import norm as nnorm
from numpy import sum, where, array, sign, sqrt, dot as ndot
from sets import Set

if MPI.num_processes() > 1:
    raise RuntimeError("Not working in parallel yet")

comm = nMPI.COMM_WORLD
parameters["form_compiler"]["optimize"]     = True
parameters["form_compiler"]["cpp_optimize"] = True

mesh = Rectangle(0, 0, 100, 100, 100, 100)
V = VectorFunctionSpace(mesh, 'CG', 2)
u = interpolate(Expression(('pi/314.*(50.-x[1])', 
                            'pi/314.*(x[0]-50.)')), V)
u.gather() # Required for parallel

# Initialize particles
# Note, with a random function one must compute points on 0 and bcast to get the same values on all procs
x = []
nn = []    
if comm.Get_rank() == 0:
    x, nn = zalesak(center=(50, 75), N=1000, normal=True)    
    
x = comm.bcast(x, root=0) 
nn = comm.bcast(nn, root=0) 
lp = LagrangianParticles(V)
lp.add_particles(x, {'normal': nn})

Q = FunctionSpace(mesh, 'CG', 2)
phi = Function(Q)
phiv = phi.vector()
dim = Q.cell().d
x, xv = [], []
for i in range(dim):
    x.append(interpolate(Expression('x[{}]'.format(i)), Q))
    xv.append(x[i].vector())

# Find the closest particle to each dof in cells that contains particles
pos_d = zeros(dim)
# Create a list of particles independent of cells
list_of_particles = []
p_pos = zeros((lp.cellparticles.total_number_of_particles(), dim))
for cell in lp.cellparticles.itervalues():
    list_of_particles += cell.particles
for i, particle in enumerate(list_of_particles):
    p_pos[i, :] = particle.position[:]

for dof in range(phiv.local_range(0)[0], phiv.local_range(0)[1]):
    xd = zeros(dim)
    for i in range(dim):
        xd[i] = xv[i][dof]     # The position (x,y,z) of the dof
    p = sum((xd - p_pos)**2, axis=1).argmin()    
    dn = xd - p_pos[p, :]
    phiv[dof] = ndot(dn, list_of_particles[p].prm['normal'])

# Just a few test using variable density
R = FunctionSpace(mesh, 'CG', 1)
rho = project(phi, R)
rhom = 1.2
rhop = 0.8
for i in range(rho.vector().size()):
    if rho.vector()[i] < 0:
        rho.vector()[i] = rhom
    else:
        rho.vector()[i] = rhop

#plot(rho)
#DG = FunctionSpace(mesh, 'DG', 0)
#dg = project(rho, DG)
#p = TrialFunction(Q)
#q = TestFunction(Q)
#F = dg*inner(grad(p), grad(q))*dx + inner(dot(grad(rho), grad(rho)), q)*dx
#s = Function(Q)
#solve(lhs(F) == rhs(F), s)
#normalize(s.vector())
#plot(s, title='S')

cf = CellFunction('uint', mesh)
cf.set_all(0)
for cell in cells(mesh):
    mp = cell.midpoint()
    dn = array([mp.x(), mp.y()])
    p = sum((dn - p_pos)**2, axis=1).argmin()    
    dn = dn - p_pos[p, :]
    dn = ndot(dn, list_of_particles[p].prm['normal'])
    if dn > 0:
        cf[cell] = 1
    else:
        cf[cell] = 2
    
for cell in lp.cellparticles.iterkeys():
    cf[cell] = 3

problem_parameters['time_integration'] = 'Steady'
problem = Problem(mesh, problem_parameters)

solver_parameters['degree']['p'] = 2
solver_parameters['degree']['c'] = 0
solver_parameters['family']['c'] = 'R'
solver_parameters['iteration_type'] = 'Newton'
Potential = PDESystem([['p', 'c']], problem, solver_parameters)
n = FacetNormal(mesh)
df = n
pointsources = []
for cell in lp.cellparticles.itervalues():
    for particle in cell.particles:
        pointsources.append(PointSource(Potential.V['p'], Point(*particle.position), -0.01))

PointSource.apply1 = PointSource.apply 
def myapply(self, *args):
    if len(args) == 2:
        self.apply1(args[1])
    elif len(args) == 1:
        self.apply1(args[0])
    elif len(args) == 3:
        self.apply1(args[1])
PointSource.apply = myapply

class potential(PDESubSystem):
    def form(self, p, v_p, c, v_c, **kwargs):
        return inner(grad(p), grad(v_p))*dx + \
            ((c)*v_p + v_c*(p))*dx

class Eikonal(PDESubSystem):        
    def form(self, p, p_, v_p, eps, **kwargs):
        return sqrt(inner(grad(p_), grad(p_)))*v_p*dx + \
        eps*inner(grad(p_), grad(v_p))*dx
            
#Potential.add_pdesubsystem(potential, ['p', 'c'], normalize=normalize)
Potential.add_pdesubsystem(potential, ['p', 'c'], bcs=pointsources)
#Potential.eps = Constant(0.01)
#Potential.add_pdesubsystem(Eikonal, ['p'], bcs=pointsources)

problem.solve()

ff = File("test.pvd")
ff << Potential.p_