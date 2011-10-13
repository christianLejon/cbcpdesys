__author__ = "Mikael Mortensen <Mikael.Mortensen@ffi.no>"
__date__ = "2010-08-30"
__copyright__ = "Copyright (C) 2010 " + __author__
__license__  = "GNU GPL version 3 or any later version"

import cPickle
from cbc.cfd.problems.channel import *
from scipy.interpolate import InterpolatedUnivariateSpline as ius
from pylab import zeros, linspace

# Define initial constants for initializing the turbulence parameters.
# The length of the list must equal the number of spaces/subspaces in the 
# function.
initial_constants = recursive_update(
                    laminar_velocity, dict(
                        k=0.01, 
                        e=0.01, 
                        v2=('0.001'),
                        f=('1.'), 
                        R=('0.001',)*3,
                        Rij=('0.0',)*4, 
                        Fij=('0.0',)*4))

NSchannel = channel
class channel(NSchannel):
    """Set up turbulent channel.
    
    For turbulent flows it is common to characterize the flow using Re_tau, 
    a Reynolds number based on friction velocity u_tau=sqrt(nu*du/dy)_wall.
            
            Re_tau = u_tau/nu
            
    The constant pressure gradient is computed by integrating the NS-equations
    across the height of the channel
    
    \int_{-1}^{1}(dp/dx) dV = -\int_{-1}^{1} nu*d^2/dy^2u dV 
    2*dp/dx = - nu*(du/dy_top- du/dy_bottom) = - (-u_tau**2 - u_tau**2)
    dp/dx = u_tau**2
    """
    def __init__(self, parameters):
        NSchannel.__init__(self, parameters)
        self.nu = self.prm['viscosity'] = self.prm['utau']/self.prm['Re_tau']    

    def initialize(self, pdesystem):
        Problem.initialize(self, pdesystem, initial_constants)
        
    def body_force(self):
        return Constant((self.prm['utau']**2, 0.))
        
    def tospline(self): 
        """Store the results in a dictionary of splines and save to file.
        The channel solution is very often used to initialize other problems
        that have a VelocityInlet."""
        f = open('../data/channel_' + self.prm['Model'] + '_' + 
                 str(self.prm['Re_tau']) + '.ius','w')
        N = 1000
        xy = zeros(N)
        xy = zeros((N, 2))
        xy[:, 1] = linspace(0., 1., 1000) # y-direction
        spl = {}
        for solver in ('NS_solver', 'Turb_solver'):            
            s = getattr(self, solver)
            for name in s.names:
                vals = zeros((N, s.V[name].num_sub_spaces() or 1))
                for i in range(N):
                    getattr(s, name + '_').eval(vals[i, :], xy[i, :])
                spl[name] = []
                # Create spline for all components
                # Scalar has zero subspaces, hence the or
                for i in range(s.V[name].num_sub_spaces() or 1): 
                    spl[name].append(ius(xy[:, 1], vals[:, i]))
            
        cPickle.dump(spl, f)
        f.close()
        return spl

    def __info__(self):
        return "Turbulent channel solved with the %s turbulence model" %(self.prm['Model'])

if __name__ == '__main__':
    import cbc.cfd.icns as icns
    import cbc.cfd.ransmodels as ransmodels
    from time import time
    set_log_active(True)
    
    ## Set up problem ##
    problem_parameters['time_integration']='Steady'
    problem_parameters['Nx'] = 10
    problem_parameters['Ny'] = 100
    problem_parameters['Re_tau'] = 395.
    problem_parameters['utau'] = 0.05
    problem_parameters['plot_velocity'] = True
    problem = channel(problem_parameters)
    
    ## Set up Navier-Stokes solver ##
    from cbc.cfd.icns import solver_parameters
    solver_parameters['degree']['u'] = 1
    underrelax = lambda : 0.8
    solver_parameters['omega'].default_factory = underrelax
    #NS_solver = icns.NSSegregated(NS_problem, icns.parameters)
    NS_solver = icns.NSCoupled(problem, solver_parameters)
    
    ## Set up turbulence model ##
    from cbc.cfd.ransmodels.V2F import solver_parameters as turb_parameters
    underrelax = lambda : 0.7
    turb_parameters['omega'].default_factory = underrelax
    problem_parameters['model'] = 'OriginalV2F'
    Turb_solver = ransmodels.V2F_2Coupled(problem, turb_parameters, model=problem_parameters['model'])
    
    ## solve the problem ##    
    t0 = time()
    problem_parameters['max_iter'] = 0
    problem.solve()
    print 'time = ', time()-t0
    print summary()
    plot(NS_solver.u_)
    interactive()
