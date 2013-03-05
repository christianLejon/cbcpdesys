__author__ = "Mikael Mortensen <mikaem@math.uio.no>"
__date__ = "2010-11-03"
__copyright__ = "Copyright (C) 2010 " + __author__
__license__  = "GNU Lesser GPL version 3 or any later version"
""" 
    Taylor Green test problem in 3D
"""
from NSProblem import *

# Specify the initial velocity field
initial_velocity = Initdict( 
    u=('2./sqrt(3.)*sin(2.*pi/3.)*sin(x[0])*cos(x[1])*cos(x[2])',
       '2./sqrt(3.)*sin(-2.*pi/3.)*cos(x[0])*sin(x[1])*cos(x[2])',
       '0.0'),
    p=('0'))

# Required parameters
problem_parameters['Nx'] = 16
problem_parameters['Ny'] = 16
problem_parameters['Nz'] = 16
problem_parameters['Re'] = 100.

# Problem definition
class TaylorGreen(NSProblem):

    def __init__(self, parameters):
        NSProblem.__init__(self, parameters=parameters)
        self.mesh = self.gen_mesh()
        # Set viscosity
        self.prm['viscosity'] = 1.0/self.prm['Re']
        self.prm['dt'] = self.timestep()    
        self.boundaries = self.create_boundaries()
        self.vorticity_file = File("vorticity.pvd")
        self.q0 = initial_velocity
    
    def gen_mesh(self):
        #m = UnitCube(self.prm['Nx'], self.prm['Ny'], self.prm['Nz'])
        m = BoxMesh(0, 0, 0, 1, 1, 1, self.prm['Nx'], self.prm['Ny'], self.prm['Nz'])
        scale = 2*(m.coordinates() - 0.5)*pi
        m.coordinates()[:, :] = scale
        return m
                
    def create_boundaries(self):
        
        self.mf = FacetFunction("uint", self.mesh) # Facets
        self.mf.set_all(0)
        
        def inside(x, on_boundary):
            # return True if on left or bottom boundary AND NOT on one of the two corners (0, 1) and (1, 0)
            return bool((near(x[0], -pi) or near(x[1], -pi) or near(x[2], -pi)) and 
                    (not ((near(x[0], -pi) and near(x[1], pi) and near(x[2], pi)) or 
                          (near(x[0], pi) and near(x[1], -pi) and near(x[2], pi)) or
                          (near(x[0], -pi) and near(x[1], pi) and near(x[2], -pi)) or 
                          (near(x[0], pi) and near(x[1], -pi) and near(x[2], -pi)))) and on_boundary)

        def periodic_map(x, y):
            if near(x[0], pi) and near(x[1], pi):
                y[0] = x[0] - 2.0*pi
                y[1] = x[1] - 2.0*pi
                y[2] = x[2] - 2.0*pi
            elif near(x[0], pi):
                y[0] = x[0] - 2.0*pi
                y[1] = x[1]
                y[2] = x[2]
            elif near(x[1], pi):
                y[0] = x[0]
                y[1] = x[1] - 2.0*pi
                y[2] = x[2]
            elif near(x[2], pi):
                y[0] = x[0]
                y[1] = x[1]
                y[2] = x[2] - 2.0*pi
                
        periodic = FlowSubDomain(inside, bc_type='Periodic',
                                mf=self.mf, periodic_map=periodic_map)

        return [periodic]
       
    def update(self):
        if (self.tstep-1) % self.NS_solver.prm['save_solution'] == 0:
            V = MixedFunctionSpace([self.NS_solver.V['u0']]*3)
            ff = project(curl(self.NS_solver.u_), V)
            self.vorticity_file << ff
            
    def info(self):
        return "Taylor-Green vortex"

if __name__ == '__main__':
    from cbc.cfd import icns                    # Navier-Stokes solvers
    from cbc.cfd.icns import solver_parameters  # parameters to NS solver
    set_log_active(True)
    problem_parameters['time_integration']='Transient'
    problem_parameters['Nx'] = 1
    problem_parameters['Ny'] = 1
    problem_parameters['Nz'] = 1
    problem_parameters['T'] = 0.1
    solver_parameters = recursive_update(solver_parameters, 
    dict(degree=dict(u=1),
         pdesubsystem=dict(u=101, p=101, velocity_update=101, up=1), 
         linear_solver=dict(u='bicgstab', p='lu', velocity_update='bicgstab'), 
         precond=dict(u='hypre_euclid', p='hypre_amg', velocity_update='hypre_euclid'),
         save_solution=10)
         )
    
    NS_problem = TaylorGreen(problem_parameters)
    NS_solver = icns.NSFullySegregated(NS_problem, solver_parameters)
    #NS_solver = icns.NSSegregated(NS_problem, solver_parameters)
    #NS_solver = icns.NSCoupled(NS_problem, solver_parameters)
    
    #for name in NS_solver.system_names:
        #if name is not 'p':
            #NS_solver.pdesubsystems[name].prm['monitor_convergence'] = True
            #NS_solver.pdesubsystems[name].linear_solver.parameters['relative_tolerance'] = 1e-12
            #NS_solver.pdesubsystems[name].linear_solver.parameters['absolute_tolerance'] = 1e-12
    t0 = time()
    NS_problem.solve()
    print 'time = ', time()-t0
    list_timings()
    plot(NS_solver.u_)
    interactive()
    