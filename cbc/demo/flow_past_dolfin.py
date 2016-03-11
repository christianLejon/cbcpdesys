from cbc.pdesys import *
set_log_active(False)

# Set up problem by loading mesh from file
mesh = Mesh('dolfin_fine.xml')

# problem_parameters are defined in Problem.py
problem_parameters['time_integration'] = "Transient"    # default='Steady'
problem = Problem(mesh, problem_parameters)

# Set up first PDESystem
solver_parameters['space']['u'] = VectorFunctionSpace   # default=FunctionSpace
solver_parameters['degree']['u'] = 2                    # default=1
NStokes = PDESystem([['u', 'p']], problem, solver_parameters)

# Use a constant forcing field to drive the flow from right to left
NStokes.f = Constant((-1., 0.))

# No-slip boundary condition for velocity on the dolfin
dolfin = AutoSubDomain(lambda x, on_boundary: on_boundary and not
                       (near(x[0], 0) or near(x[0], 1.) or near(x[1], 0.) or near(x[1], 1.)))

bc = [DirichletBC(NStokes.V['up'].sub(0), Constant((0.0, 0.0)), dolfin)]

# Set up variational form.
# u_, u_1 are the solution Functions at time steps N and N-1.
# v_u/v_p are the TestFunctions for velocity/pressure in the MixedFunctionSpace for u and p

NStokes.nu = Constant(0.01)
class NavierStokes(PDESubSystem):
    def form(self, u, v_u, u_, u_1, p, v_p, nu, dt, f, **kwargs):
        U = 0.5*(u + u_1)
        return ((1./dt)*inner(u - u_1, v_u)*dx + 
               inner(dot(u_1, nabla_grad(u_1)), v_u)*dx + 
               nu*inner(grad(U), grad(v_u))*dx -
               inner(p, div(v_u))*dx + inner(div(U), v_p)*dx - 
               inner(f, v_u)*dx)

NStokes.pdesubsystems['up'] = NavierStokes(vars(NStokes), ['u', 'p'], bcs=bc,
                                           reassemble_lhs=False)

# Integrate the solution from t=0 to t=0.5
problem.prm['T'] = 0.5
problem.solve()

# Define a new nonlinear PDESystem for a scalar c
scalar = PDESystem([['c']], problem, solver_parameters)

class Scalar(PDESubSystem):
    def form(self, c, v_c, c_, c_1, U_, dt, nu, **kwargs):
        C = 0.5*(c + c_1)
        return (1./dt)*inner(c - c_1, v_c)*dx + \
                inner(dot(U_, grad(C)), v_c)*dx + \
                nu*(1. + c_**2)*inner(grad(C), grad(v_c))*dx
                # Note nonlinearity in c_ (above)

bcc = [DirichletBC(scalar.V['c'], Constant(1.0), dolfin)]

scalar.U_ = 0.5*(NStokes.u_ + NStokes.u_1) # The Scalar form uses the velocity
scalar.nu = NStokes.nu
csub1 = Scalar(vars(scalar), ['c'], bcs=bcc, max_inner_iter=5) # Iterate on c_
scalar.pdesubsystems['c'] = csub1

# Integrate both PDESystems from t=0.5 to t=1.0 using Picard
# iterations on each time step
problem.prm['T'] = 1.0
problem.solve()

# Switch to using the Newton method for the nonlinear variational form
# With these calls we replace c by c_ in the Scalar form and compute the Jacobian wrt c_
csub1.prm['iteration_type'] = 'Newton'
csub1.define()

# Integrate both PDESystems from T=1.0 to T=1.5 using Newton
# iterations on each time step for the scalar
problem.prm['T'] = 1.5
problem.solve()