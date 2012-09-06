__author__ = "Mikael Mortensen <mikaem@math.uio.no>"
__date__ = "2010-08-30"
__copyright__ = "Copyright (C) 2010 " + __author__
__license__  = "GNU Lesser GPL version 3 or any later version"

from NSProblem import *
from numpy import arctan, array, ceil
from cbc.cfd.tools.Streamfunctions import StreamFunction
    
def lid(x, on_boundary):
    return (on_boundary and near(x[1], 1.0))
    
def stationary_walls(x, on_boundary):
    return on_boundary and (near(x[0], 0.) or near(x[0], 1.) or near(x[1], 0.))

lid_velocity = Initdict(u=('1', '0'), p='0')

# set default to transient in case drivencavity is imported elsewhere
problem_parameters['time_integration'] = 'Transient' 

class drivencavity(NSProblem):
    """2D lid-driven cavity."""
    def __init__(self, parameters):
        NSProblem.__init__(self, parameters=parameters)
        self.init_memory_use = self.getMyMemoryUsage()
        
        self.mesh = UnitSquare(self.prm['Nx'], self.prm['Ny'])
        #self.mesh = self.gen_mesh()
        # Set viscosity
        self.prm['viscosity'] = 1./self.prm['Re']
        # Set timestep as NSbench
        self.prm['dt'] = 0.25*self.prm['T']/ceil(self.prm['T']/0.2/self.mesh.hmin())
        # Boundaries        
        walls = FlowSubDomain(stationary_walls, bc_type='Wall')
        top   = FlowSubDomain(lid, func=lid_velocity, bc_type='Wall')
        self.boundaries = [top, walls]
        self.u_file = File("results/drivencavity/u.pvd")
                
    def gen_mesh(self):
        m = Rectangle(-1., -1., 1., 1., self.prm['Nx'], self.prm['Ny'])
        # Create stretched mesh in x- and y-direction
        x = m.coordinates()
        x[:, 1] = arctan(pi/2*(x[:, 1]))/arctan(pi/2) 
        x[:, 0] = arctan(pi/2*(x[:, 0]))/arctan(pi/2)
        x[:, 1] = (x[:, 1] + 1)/2.
        x[:, 0] = (x[:, 0] + 1)/2.        
        return m

    def initialize(self, pdesystem):
        """Initialize solution simply by applying lid_velocity to the top boundary"""
        return False   # This will simply use u = (0, 0) and p = 0 all over
        if pdesystem.prm['familyname'] == 'Navier-Stokes':
            for name in pdesystem.system_names:
                if not name == 'up': # Coupled solver does not need this, makes no difference
                    d = DirichletBC(pdesystem.V[name], lid_velocity[name], lid)
                    d.apply(pdesystem.x_[name])
                    d.apply(pdesystem.x_1[name])
            return True
        else:
            return NSProblem.initialize(self, pdesystem)

    def functional(self, u):
        """Compute stream function and report minimum node value."""
        psi = StreamFunction(u, [], use_strong_bc=True)
        vals  = psi.vector().array()
        vmin = vals.min()
        #info_green("Stream function has minimal value {}".format(vmin))
        #info_green("Velocity at (0.75, 0.75) = {}".format(u[0]((0.75, 0.75))))
        return vmin
        
    def update(self):
        if hasattr(self, 'lp'):
            self.lp.step(self.pdesystems['Navier-Stokes'].u_, self.prm['dt'])
            if self.tstep % 500:
                self.lp.scatter()
        #if self.tstep % 100 == 0:
            #info_red('Memory usage = ' + self.getMyMemoryUsage())

    def reference(self, t):
        """Reference min streamfunction for T=2.5, Re = 1000."""
        return -0.061076605

    def __str__(self):
        return "Driven cavity"

def line(x0, y0, dx, dy, N=10):
    """Create points evenly distributed on a line"""
    L = sqrt(dx**2 + dy**2)
    dL = linspace(0, L, N)
    theta = arctan(dy/dx)
    x = x0 + dL*cos(theta)
    y = y0 + dL*sin(theta)
    points = []
    for xx, yy in zip(x, y):
        points.append(array([xx, yy]))
    return points
        
if __name__ == '__main__':
    import cbc.cfd.icns as icns
    from cbc.cfd.icns import solver_parameters
    from cbc.cfd.LP import LagrangianParticles
    from numpy import linspace, pi, zeros, where, array, ndarray, squeeze, load, sin, cos, arcsin, arctan
    from pylab import show
    import time
    import sys
    set_log_active(True)
    parameters["linear_algebra_backend"] = "PETSc"
    mesh_sizes = [2, 11, 16, 23, 32, 45, 64, 91, 128, 181, 256, 362]
    try:
        N = eval(sys.argv[-1])
    except:
        N = 2
    problem_parameters['Nx'] = mesh_sizes[N]
    problem_parameters['Ny'] = mesh_sizes[N]
    problem_parameters['Re'] = 1000.
    problem_parameters['T'] = 0.5
    #problem_parameters['T'] = 0.02
    #problem_parameters['plot_velocity'] = True
    #problem_parameters['plot_pressure'] = True
    #problem_parameters['max_iter'] = 1
    problem_parameters['iter_first_timestep'] = 2
    solver_parameters = recursive_update(solver_parameters, 
    dict(degree=dict(u=2, u0=1, u1=1),
        pdesubsystem=dict(u=101, p=101, velocity_update=101, up=1), 
        linear_solver=dict(u='bicgstab', p='gmres', velocity_update='bicgstab'), 
        precond=dict(u='jacobi', p='hypre_amg', velocity_update='jacobi'),
        iteration_type='Picard',
        max_iter=1 # Number of pressure/velocity iterations on given timestep
        ))
    problem = drivencavity(problem_parameters)
    #solver = icns.NSFullySegregated(problem, solver_parameters)
    solver = icns.NSCoupled(problem, solver_parameters)
    #x = line(x0=0.5, y0=0.5, dx=0.5, dy=0., N=10)
    #lp = LagrangianParticles(solver.V['u'])
    #lp.add_particles(x)
    #problem.lp = lp
    
    t0 = time.time()
    problem.solve(logging=True)
    t1 = time.time()-t0
    info_red('Total computing time = {0:f}'.format(t1))
    list_timings()

    print 'Functional = ', problem.functional(solver.u_), ' ref ', problem.reference(0)

    info_red('Additional memory use of solver = {}'.format(eval(problem.getMyMemoryUsage()) - eval(problem.init_memory_use)))

    ## plot result. For fully segregated solver one should project the velocity vector on the correct space, if not the plot will look bad
    if solver.__class__ is icns.NSFullySegregated:
       plot(project(solver.u_, VectorFunctionSpace(solver.mesh, 'CG', solver_parameters['degree']['u0'])))
    else:
       plot(solver.u_)
                
    psi = problem.functional(solver.u_)    
    
    psi_error = abs(psi-problem.reference(0))
    
    #dump_result(problem, solver, t1, psi_error)
    
