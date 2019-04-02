"""This demo illustrates how to use of DOLFIN and CBC.PDESys
to solve the Stokes problem using a preconditioned iterative method.
"""
__author__ = "Mikael Mortensen <mikaem@math.uio.no>"
__date__ = "2011-08-16"
__copyright__ = "Copyright (C) 2011 " + __author__
__license__  = "GNU Lesser GPL version 3 or any later version"

from dolfin import *
import matplotlib.pyplot as plt
# from cbc.pdesys import *
import pdb
import sys
sys.path.append('/home/hakgra/workspace/fenics/cbcpdesys_update_RANS/cbc/pdesys')
from Problem import *
from PDESubSystems import *
from PDESystem import *

#set_log_active(True)
#set_log_level(5)

# Test for PETSc or Epetra
if not has_linear_algebra_backend("PETSc") and not has_linear_algebra_backend("Epetra"):
    print("DOLFIN has not been configured with Trilinos or PETSc. Exiting.")
    exit()

info_green("This demo is unlikely to converge if PETSc is not configured with Hypre or ML.")

# Boundaries
def right(x, on_boundary): return x[0] > (1.0 - DOLFIN_EPS)
def left(x, on_boundary): return x[0] < DOLFIN_EPS
def top_bottom(x, on_boundary):
    return x[1] > 1.0 - DOLFIN_EPS or x[1] < DOLFIN_EPS

class Stokes_Solver(PDESystem):
    
    def __init__(self, mesh, parameters=parameters):

        PDESystem.__init__(self, [['u', 'p']], mesh, parameters)
        
        self.f = Constant((0.0, 0.0, 0.0))
        
        # Set up boundary conditions.
        inflow = Expression(("-sin(x[1]*pi)", "0.0", "0.0"), degree = parameters['degree']['u'])
        noslip = Constant((0.0, 0.0, 0.0))
        zero = Constant(0)
        self.bcs = [DirichletBC(self.V['up'].sub(0), noslip, top_bottom),
                    DirichletBC(self.V['up'].sub(0), inflow, right),
                    DirichletBC(self.V['up'].sub(1), zero, left)] 
        
        # Set up variational problem
        self.define()
        
    def define(self):
        self.pdesubsystems['up'] = Stokes(vars(self), ['u', 'p'], bcs=self.bcs)
                        
class Stokes(PDESubSystem):
    """Variational form of the Stokes problem."""
    def form(self, u, u_, v_u, p, p_, v_p, f, **kwargs):
        self.prm['assemble_system'] = True # Assemble symmetric system with assemble_system
        return inner(grad(u), grad(v_u))*dx + div(v_u)*p*dx + v_p*div(u)*dx \
               - inner(f, v_u)*dx
                       
    def get_precond(self, u, v_u, p, v_p, f, **kwargs):
        # Form for use in constructing preconditioner matrix
        b = inner(grad(u), grad(v_u))*dx + p*v_p*dx
        L = inner(f, v_u)*dx
        
        # Assemble preconditioner system. B is hooked up in setup_solver
        B, dummy = assemble_system(b, L, self.bcs)
        
        return B
        
if __name__ == '__main__':
    # Update the default solver parameters
    solver_parameters = copy.deepcopy(default_solver_parameters)
    #solver_parameters = default_solver_parameters
    solver_parameters = recursive_update(solver_parameters, {
        'space': {'u': VectorFunctionSpace},
        'element':{'u': VectorElement},
        'degree': {'u': 2},
        'time_integration': 'Steady',
        'linear_solver': {'up': 'minres'},
        'precond': {'up': 'amg'},
        'monitor_convergence':{'up': True},        
    })
    mesh = UnitCubeMesh(16, 16, 16)
    solver = Stokes_Solver(mesh, solver_parameters)
    solver.pdesubsystems['up'].linear_solver.parameters['relative_tolerance'] = 1.e-9
    pdb.set_trace()
    solver.solve(max_iter=1, redefine=False)
    plt.figure(1) 
    plot(solver.u_, title="u_")
    plt.show()
    plt.figure(2) 
    plot(solver.p_, title="p_")
    plt.show()
