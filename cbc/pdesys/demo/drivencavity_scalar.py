from cbc.pdesys import *

class NavierStokes(PDESubSystem):
    """Transient form with Crank-Nicholson (CN) diffusion and where convection 
    is computed using AB-projection for convecting and CN for convected 
    velocity."""    
    def form(self, u, v_u, p, v_p, u_, nu, dt, u_1, u_2, f, **kwargs):
        U = 0.5*(u + u_1)
        U1 = 1.5*u_1 - 0.5*u_2
        F = (1./dt)*inner(u - u_1, v_u)*dx + self.conv(v_u, U, U1)*dx \
            + nu*inner(grad(v_u), grad(U) + grad(U).T)*dx \
            - inner(div(v_u), p)*dx - inner(v_p, div(u))*dx - inner(v_u, f)*dx
        return F
        
class Scalar(PDESubSystem):
    def form(self, c, v_c, c_, c_1, u_, u_1, nu, Pr, dt, **kwargs):
        U_ = 0.5*(u_ + u_1)
        C = 0.5*(c + c_1)
        F = (1./dt)*inner(c - c_1, v_c)*dx + inner(dot(U_, grad(C)), v_c)*dx \
            + nu/Pr*inner(grad(v_c), grad(C))*dx
        return F

problem_parameters = recursive_update(problem_parameters, {
    'viscosity': 0.01,
    'dt': 0.01,
    'T': 0.1,
    'time_integration': 'Transient'
})
mesh = UnitSquare(20, 20)
problem = Problem(mesh, problem_parameters)

solver_parameters = recursive_update(solver_parameters, {
    'degree': {'u':2, 'c': 2},
    'space': {'u': VectorFunctionSpace},
    'familyname': 'Navier-Stokes'
})
NS = PDESystem([['u', 'p']], problem, solver_parameters)
NS.nu = Constant(problem.prm['viscosity'])
NS.dt = Constant(problem.prm['dt'])
NS.f = Constant((0, 0))
nup = extended_normalize(NS.V['up'], 2)
bcs = [DirichletBC(NS.V['u'], (0., 0.), "on_boundary"),
       DirichletBC(NS.V['u'], (1., 0.), "on_boundary && x[1] > 1. - DOLFIN_EPS")]
NS.pdesubsystems['up'] = NavierStokes(vars(NS), ['u', 'p'], 
                                      bcs=bcs, normalize=nup)
# Add method to plot intermediate results
def update(self):
    NS = self.pdesystems['Navier-Stokes']
    plot(NS.u_, rescale=True)
    if len(self.pdesystemlist) > 1:
        for i, name in enumerate(self.pdesystemlist[1:]):
            sol = self.pdesystems[name]
            plot(sol.c_, rescale=True)
        info_red('        {}: Total amount of c = {}'.format(name, assemble(sol.c_*dx)))
        
Problem.update = update

problem.solve()

# Set up two scalar fields
solver_parameters['familyname'] = 'Scalar1'
Scalar1 = PDESystem([['c']], problem, solver_parameters)
solver_parameters['familyname'] = 'Scalar2'
Scalar2 = PDESystem([['c']], problem, solver_parameters)
bcs1 = [DirichletBC(Scalar1.V['c'], (1.), "on_boundary && x[1] > 1. - DOLFIN_EPS")]
bcs2 = [DirichletBC(Scalar2.V['c'], (1.), "on_boundary && x[1] < DOLFIN_EPS")]
Scalar1.nu = Scalar2.nu = Constant(problem.prm['viscosity'])
Scalar1.dt = Scalar2.dt = Constant(problem.prm['dt'])
Scalar1.Pr = Constant(10.)
Scalar2.Pr = Constant(1.)
Scalar1.u_  = Scalar2.u_  = NS.u_
Scalar1.u_1 = Scalar2.u_1 = NS.u_1
Scalar1.pdesubsystems['c'] = Scalar(vars(Scalar1), ['c'], bcs=bcs1)
Scalar2.pdesubsystems['c'] = Scalar(vars(Scalar2), ['c'], bcs=bcs2)

problem.prm['T'] = 2.
problem.solve()
