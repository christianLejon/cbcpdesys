__author__ = "Mikael Mortensen <Mikael.Mortensen@ffi.no>"
__date__ = "2010-08-30"
__copyright__ = "Copyright (C) 2010 " + __author__
__license__  = "GNU GPL version 3 or any later version"

from NSProblem import *
from numpy import arctan, array

# Laminar velocity profiles that can be used in Expressions
# for initializing the solution or prescribing a VelocityInlet.
laminar_velocity = Initdict(u=(("(1.-x[1]*x[1])",  "0")), 
                            p=("0"))

zero_velocity = Initdict(u=(("0",  "0")), 
                         p=("0"))

class channel(NSProblem):
    """ 
    Set up laminar channel flow test case. 
    The channel spans  -1 <= y <= 1.
    
    For laminar flows the flow is usually caracterized by Re based on the 
    maximum velocity, i.e. Re = U/nu. If the steady solution is set to 
    u = 1-y**2, then Re=1/nu and the pressure gradient is 2/Re. This 
    is based on an exact integral across the channel height:
    
    \int_{-1}^{1}(dp/dx) dV = -\int_{-1}^{1} nu*d^2/dy^2u dV 
    2*dp/dx = - nu*(du/dy_top- du/dy_bottom) = -1/Re*(-2 - 2)
    dp/dx = 2/Re
        
    """    
    def __init__(self, parameters):
        NSProblem.__init__(self, parameters=parameters)
        self.mesh = self.gen_mesh()
        self.prm['viscosity'] = 1./self.prm['Re']
        self.prm['dt'] = self.timestep()
        self.periodic_boundaries = False
        self.boundaries = self.create_boundaries()
        
    def gen_mesh(self):
        self.L = 1.
        m = Rectangle(0., -1., self.L, 1., self.prm['Nx'], self.prm['Ny'])
        # Create stretched mesh in y-direction
        x = m.coordinates()        
        x[:, 1] = arctan(pi*(x[:, 1]))/arctan(pi) 
        return m        
        
    def create_boundaries(self):
        self.mf = FacetFunction("uint", self.mesh) # Facets
        self.mf.set_all(0)
                        
        walls = FlowSubDomain(lambda x, on_boundary: ((near(x[1], -1.) or 
                                                       near(x[1], 1.)) and
                                                       on_boundary),
                              bc_type = 'Wall',
                              mf = self.mf)

        #symmetry = FlowSubDomain(lambda x, on_boundary: near(x[1], 0.) and on_boundary,
                                #bc_type = 'Symmetry',
                                #mf = self.mf)

        if self.periodic_boundaries:

            def periodic_map(x, y):
                y[0] = x[0] - self.L
                y[1] = x[1]
            p = periodic_map
            p.L = self.L
                
            periodic = FlowSubDomain(lambda x, on_boundary: near(x[0], 0.) and on_boundary,
                                    bc_type = 'Periodic',
                                    mf = self.mf,
                                    periodic_map = p)
            bcs = [walls, periodic]
            
        else:
            # Note that VelocityInlet should only be used for Steady problem
            #inlet    = FlowSubDomain(lambda x, on_boundary: near(x[0], 0.) and on_boundary,
                                    #bc_type = 'VelocityInlet',
                                    #func = laminar_velocity,
                                    #mf = self.mf)
            #outlet   = FlowSubDomain(lambda x, on_boundary: near(x[0], self.L) and on_boundary,
                                    #bc_type = 'Outlet',
                                    #func = {'p': Constant(0.0)},
                                    #mf = self.mf)
                                    
            inlet    = FlowSubDomain(lambda x, on_boundary: near(x[0], 0.) and on_boundary,
                                    bc_type = 'ConstantPressure',
                                    func = {'p': Constant(2./self.prm['Re']*self.L)},
                                    mf = self.mf)
                                
            outlet   = FlowSubDomain(lambda x, on_boundary: near(x[0], self.L) and on_boundary,
                                    bc_type = 'ConstantPressure',
                                    func = {'p': Constant(0.0)},
                                    mf = self.mf)
                                
            bcs = [walls, inlet, outlet]                    
        
        return bcs
        
    def initialize(self, pdesystem):
        transient = self.prm['time_integration']=='Transient'
        Problem.initialize(self, pdesystem, zero_velocity if transient 
                                                          else laminar_velocity)
        
    def body_force(self):
        if self.periodic_boundaries:
            return Constant((2./self.prm['Re'], 0.))
        else:
            return Constant((0., 0.))
            
    def functional(self, u):
        x = array((1.0, 0.))
        values = array((0.0, 0.0))
        u.eval(values, x)
        return values[0]

    def reference(self):
        num_terms = 10000
        u = 1.0
        c = 1.0
        for n in range(1, 2*num_terms, 2):
            a = 32. / (pi**3*n**3)
            b = (0.25/self.prm['Re'])*pi**2*n**2
            c = -c
            u += a*exp(-b*self.t)*c
        return u
        
    def error(self):
        return self.functional(self.NS_solver.u_) - self.reference()
        
    def __info__(self):
        return 'Periodic channel flow'

if __name__ == '__main__':
    import cbc.cfd.icns as icns
    from time import time
    set_log_active(True)
    problem_parameters['time_integration']='Transient'
    problem_parameters['Nx'] = 10
    problem_parameters['Ny'] = 50
    problem_parameters['T'] = 0.5
    problem_parameters['max_iter'] = 1
    problem_parameters['plot_velocity'] = False
    from cbc.cfd.icns import solver_parameters
    solver_parameters = recursive_update(solver_parameters, 
    dict(degree=dict(u=1),
         pdesubsystem=dict(u=1, p=1, velocity_update=1, up=1), 
         linear_solver=dict(u='lu', p='lu', velocity_update='lu'), 
         precond=dict(u='jacobi', p='amg', velocity_update='jacobi'))
         )
    
    NS_problem = channel(problem_parameters)
    #NS_solver = icns.NSFullySegregated(NS_problem, solver_parameters)
    NS_solver = icns.NSSegregated(NS_problem, solver_parameters)
    #NS_solver = icns.NSCoupled(NS_problem, solver_parameters)
    #NS_solver.pdesubsystems['u1'].prm['monitor_convergence'] = True
    #NS_solver.pdesubsystems['u2'].prm['monitor_convergence'] = True
    #NS_solver.pdesubsystems['p'].prm['monitor_convergence'] = True
    #NS_solver.pdesubsystems['u1_update'].prm['monitor_convergence'] = True
    #NS_solver.pdesubsystems['u2_update'].prm['monitor_convergence'] = True
    t0 = time()
    NS_problem.solve()
    print 'time = ', time()-t0
    print summary()
    plot(NS_solver.u_)
    #interactive()
    