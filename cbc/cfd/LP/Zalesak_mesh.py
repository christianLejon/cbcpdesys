from dolfin import *
from LagrangianParticles import zalesak, LagrangianParticlesPosition
import time
from copy import deepcopy

# Create a mesh of the Zalesak disc with topology 1.
mesh = BoundaryMesh()
editor = MeshEditor()
editor.open(mesh, 1, 2)
N = 100
z = zalesak(center=(50, 75), N=N)
editor.init_vertices(N)
editor.init_cells(N)
for i, (x, y) in enumerate(z):
    editor.add_vertex(i, x, y)
for i in range(N):
    editor.add_cell(i, i, (i+1) % (N))
plot(mesh)

f = File("zalesak_{}.xml.gz".format(N))
f << mesh

# Create rectangular mesh  
rect = Rectangle(0, 0, 100, 100, 100, 100)
V = FunctionSpace(rect, 'CG', 1)
x0 = interpolate(Expression('x[0]'), V)
x1 = interpolate(Expression('x[1]'), V)
a0 = x0.vector().array()
a1 = x1.vector().array()

# For all nodes in V find the signed distance to the disc 
point_outside = Point(10, 10)  # A point outside the disc
Np = x0.vector().size()
dummymesh = Mesh()
dummyeditor = MeshEditor()
dummyeditor.open(dummymesh, 1, 2)
dummyeditor.init_vertices(Np + 1)
dummyeditor.init_cells(Np)
dummyeditor.add_vertex(0, point_outside.x(), point_outside.y())
inside = Function(V)
inside.vector()[:] = 1
# Use ray casting to determine if the point is inside (-1) or outside (+1)
# There's probably a better way to do this than to create a dummy mesh
for i in range(Np):
    dummyeditor.add_vertex(i + 1, a0[i], a1[i])
    dummyeditor.add_cell(i, 0, i + 1)
    ei = Edge(dummymesh, i) # Edge (ray) from vertex 0 to i + 1
    ix = mesh.intersected_cells(ei) # Odd number means you're inside
    if ix.shape[0] % 2 == 1:
        inside.vector()[i] = -1
      
phi_ = Function(V)
i = 0
# Compute the signed distance
for xx, yy in zip(a0, a1):
    p = Point(xx, yy)
    zz = mesh.distance(p)
    phi_.vector()[i] = zz*inside.vector()[i]
    i += 1

plot(phi_)

ff = File("zalesak_ls.pvd")
ff << phi_, 0

V2 = VectorFunctionSpace(rect, 'CG', 1)
u_ = interpolate(Expression(('pi/314.*(50.-x[1])', 
                             'pi/314.*(x[0]-50.)')), V2)

phi = TrialFunction(V)
v = TestFunction(V)
dt = Constant(0.25)
n = FacetNormal(rect)
eps = Constant(0.01)

vv = v + dot(u_, nabla_grad(v)) # Stabilized testfunction
#F = 1./dt*inner(phi - phi_, v)*dx + 0.5*inner(v, dot(u_, nabla_grad(phi)))*dx + 0.5*inner(v, div(u_*phi))*dx
#F = 1./dt*inner(phi - phi_, v)*dx + inner(v, dot(u_, nabla_grad(phi)))*dx
#F = 1./dt*inner(phi - phi_, v)*dx + inner(v, div(u_*phi))*dx
#F = 1./dt*inner(phi - phi_, v)*dx - inner(grad(v), u_*phi)*dx + inner(v*n, u_*phi_)*ds
#F = 1./dt*inner(phi - phi_, v)*dx + inner(v, dot(u_, nabla_grad(phi)))*dx + eps*inner(grad(v), grad(phi))*dx - eps*inner(v, dot(grad(phi), n))*ds
#A, L = lhs(F), rhs(F)
# Or with a minimum amount of optimization
a0 = 1./dt*inner(vv, phi)*dx
a1 = inner(vv, dot(u_, nabla_grad(phi)))*dx
A0 = assemble(a0) # Split up because A0 can be used to compute rhs
A = assemble(a1)
A.axpy(1., A0, True)
t = 0
step = 0
pt = 1
xx = mesh.coordinates()
dx = deepcopy(xx)
# Solve the levelset function AND independently move the zalesak mesh
solver = LUSolver()
solver.parameters['reuse_factorization'] = True # Since u_ is constant    

lp = LagrangianParticlesPosition(V2)
lp.add_particles(z)

t0 = time.time()
while t <= 628*2:
    step += 1
    t = t + dt(0)
    #b = assemble(L)
    b = A0*phi_.vector()
    solver.solve(A, phi_.vector(), b)
    
    # Move particles
    #lp.step(u_, dt(0))
    
    # Move zalesak mesh
    for i in range(xx.shape[0]):
        du = u_(xx[i, :])
        xx[i, :] = xx[i, :] + du[:] * dt(0)
        dx[i, :] = du[:]
    
    if step % 20 == 0:
        ff << phi_, pt
        pt += 1
        plot(phi_)
        plot(mesh)

print 'Total time ', time.time() - t0
list_timings()
