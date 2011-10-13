__author__ = "Mikael Mortensen <Mikael.Mortensen@ffi.no>"
__date__ = "2010-08-30"
__copyright__ = "Copyright (C) 2010 " + __author__
__license__  = "GNU GPL version 3 or any later version"

from NSProblem import *
from numpy import arctan,array
from cbc.cfd.tools.Streamfunctions import StreamFunction

def lid(x, on_boundary):
    return (on_boundary and x[1] > 1.0 - DOLFIN_EPS and x[0] < 1.0 - DOLFIN_EPS and x[0] > DOLFIN_EPS)
    
def stationary_walls(x, on_boundary):
    return on_boundary and not (x[1] > 1.0 - DOLFIN_EPS and x[0] < 1.0 - DOLFIN_EPS and x[0] > DOLFIN_EPS)

# Initdict is a dictionary that uses 'u'[0] for 'u0', 'u'[1] for 'u1'
lid_velocity = Initdict(u = ('1', '0'), p = ('0'))

class drivencavity(NSProblem):
    """2D lid-driven cavity."""
    def __init__(self, parameters):
        NSProblem.__init__(self, parameters=parameters)
        self.mesh = self.gen_mesh()
        # Set viscosity
        self.nu = self.prm['viscosity'] = 1./self.prm['Re']
        # Set timestep
        self.prm['dt'] = self.timestep()
        # Boundaries
        top   = FlowSubDomain(lid, func=lid_velocity, bc_type='Wall')
        walls = FlowSubDomain(stationary_walls, bc_type='Wall')
        self.boundaries = [top, walls]
        
    def gen_mesh(self):
        m = Rectangle(-1., -1., 1., 1., self.prm['Nx'], self.prm['Ny'], 'left')
        # Create stretched mesh in x- and y-direction
        x = m.coordinates()
        x[:, 1] = arctan(pi/2*(x[:, 1]))/arctan(pi/2) 
        x[:, 0] = arctan(pi/2*(x[:, 0]))/arctan(pi/2)
        x[:, 1] = (x[:, 1] + 1)/2.
        x[:, 0] = (x[:, 0] + 1)/2.        
        return m

    def initialize(self, pdesystem):
        """Initialize solution by applying lid_velocity to the top boundary"""
        for name in pdesystem.q_:
            d = DirichletBC(pdesystem.V[name], lid_velocity[name], lid)
            d.apply(pdesystem.x_[name])
            d.apply(pdesystem.x_1[name])

    def functional(self, u):
        """Compute stream function and report minimum node value."""
        psi = StreamFunction(u, [], use_strong_bc=True)
        vals  = psi.vector().array()
        vmin = vals.min()
        print "Stream function has minimal value" , vmin
        return vmin

    def reference(self, t):
        """Reference min streamfunction for T=2.5, Re = 1000."""
        return -0.061076605

    def __str__(self):
        return "Driven cavity"

if __name__ == '__main__':
    import cbc.cfd.icns as icns
    from cbc.cfd.icns import solver_parameters
    from time import time
    set_log_active(True)
    problem_parameters['time_integration']='Transient'
    problem_parameters['Nx'] = 150
    problem_parameters['Ny'] = 150
    problem_parameters['Re'] = 1000.
    problem_parameters['T'] = 0.5
    problem_parameters['max_iter'] = 1
    problem_parameters['plot_velocity'] = False
    solver_parameters = recursive_update(solver_parameters, 
    dict(degree=dict(u=1, u0=1, u1=1),
         pdesubsystem=dict(u=101, p=101, velocity_update=101, up=1), 
         linear_solver=dict(u='bicgstab', p='gmres', velocity_update='bicgstab'), 
         precond=dict(u='jacobi', p='amg', velocity_update='ilu'),
         iteration_type='Picard')
         )
    NS_problem = drivencavity(problem_parameters)
    NS_solver = icns.NSFullySegregated(NS_problem, solver_parameters)        
    #NS_solver = icns.NSSegregated(NS_problem, solver_parameters)  
    #NS_solver = icns.NSCoupled(NS_problem, solver_parameters) 
    #NS_solver.pdesubsystems['u'].prm['monitor_convergence'] = True
    #NS_solver.pdesubsystems['u0'].prm['monitor_convergence'] = True
    #NS_solver.pdesubsystems['u1'].prm['monitor_convergence'] = True
    NS_solver.pdesubsystems['p'].prm['monitor_convergence'] = True
    t0 = time()
    NS_problem.solve()
    print 'Time = ', time() - t0
    print summary()
    
    
    