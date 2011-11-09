__author__ = "Mikael Mortensen <Mikael.Mortensen@ffi.no>"
__date__ = "2011-08-22"
__copyright__ = "Copyright (C) 2011 " + __author__
__license__  = "GNU GPL version 3 or any later version"

from NSProblem import *
from scipy.interpolate import InterpolatedUnivariateSpline as ius
from numpy import array, zeros

MCAtime = array([    0.,    27.,    42.,    58.,    69.,    88.,   110.,   130.,                                                                    
        136.,   168.,   201.,   254.,   274.,   290.,   312.,   325.,                                                                                      
        347.,   365.,   402.,   425.,   440.,   491.,   546.,   618.,                                                                                      
        703.,   758.,   828.,   897.,  1002.])/(75./60.0)/1000.
    
y1 = array([ 390.        ,  398.76132931,  512.65861027,  642.32628399,                                                        
        710.66465257,  770.24169184,  779.00302115,  817.55287009,                                                                                          
        877.12990937,  941.96374622,  970.        ,  961.2386707 ,                                                                                          
        910.42296073,  870.12084592,  843.83685801,  794.7734139 ,                                                                                          
        694.89425982,  714.16918429,  682.62839879,  644.07854985,                                                                                          
        647.58308157,  589.75830816,  559.96978852,  516.16314199,                                                                                          
        486.37462236,  474.10876133,  456.58610272,  432.05438066,  390.        ])/574.21

class SubDomains(SubDomain):
    
    def __init__(self, bid, func=None):
        SubDomain.__init__(self)
        self.bid = bid            # Boundary indicator
        self.boundary_info_in_mesh = True
        if func: self.func = func
            
    def apply(self, *args):
        """BCs that are applied weakly need to pass on regular apply.
        Other regular BCs like DirichletBC have their own apply method."""
        pass

class Walls(SubDomains):
    def type(self):
        return 'Wall'
        
class Inlet(SubDomains):
    def type(self):
        return 'VelocityInlet'
                
class PressureOutlet(SubDomains):
    def type(self):
        return 'ConstantPressure'
        
class aneurysm(NSProblem):
    
    def __init__(self, parameters):
        NSProblem.__init__(self, parameters=parameters)
        #self.mesh = Mesh("../data/100_1314k.xml.gz")
        self.mesh = Mesh("../data/Aneurysm.xml.gz")
        self.n = FacetNormal(self.mesh)        
        self.boundaries = self.create_boundaries()
        # To initialize solution set the dictionary q0: 
        #self.q0 = Initdict(u = ('0', '0', '0'), p = ('0')) # Or not, zero is default anyway
        
    def create_boundaries(self):
        # Define the spline for enough heart beats
        k = int((self.prm['T'] + 1.)/max(MCAtime))
        n = len(y1)        
        y = zeros((n - 1)*k)
        time = zeros((n - 1)*k)
        
        for i in range(k):
            y[i*(n - 1):(i + 1)*(n - 1)] = y1[:-1]
            time[i*(n - 1):(i + 1)*(n - 1)] = MCAtime[:-1] + i*max(MCAtime)

        self.inflow_t_spline = ius(time, y)
        n = self.n
        
        # Dictionary for inlet conditions
        self.n0 = assemble(-n[0]*ds(2), mesh=self.mesh)
        self.n1 = assemble(-n[1]*ds(2), mesh=self.mesh)
        self.n2 = assemble(-n[2]*ds(2), mesh=self.mesh)
        
        self.A0 = assemble(Constant(1.)*ds(2), mesh=self.mesh)
        
        # Set dictionary used for Dirichlet inlet conditions
        # For now we need to explicitly set u0, u1 and u2. Should be able to fix using just u.
        self.inflow = {'u': Expression(('n0*u_mean', 'n1*u_mean', 'n2*u_mean'), 
                                  n0=self.n0, n1=self.n1, n2=self.n2, u_mean=0),
                       'u0': Expression(('n0*u_mean'), n0=self.n0, u_mean=0),
                       'u1': Expression(('n1*u_mean'), n1=self.n1, u_mean=0),
                       'u2': Expression(('n2*u_mean'), n2=self.n2, u_mean=0)}

        # Pressures on outlets are specified by DirichletBCs, values are computed in prepare
        self.p_out1 = Expression('p', p=0)
        self.p_out2 = Expression('p', p=0)

        # Specify the boundary subdomains and hook up dictionaries for DirichletBCs
        walls = Walls(0)
        inlet = Inlet(2, self.inflow)
        pressure1 = PressureOutlet(1, {'p': self.p_out1})
        pressure2 = PressureOutlet(3, {'p': self.p_out2})
        
        return [walls, inlet, pressure1, pressure2]
        
    def prepare(self, pdesystems):
        """Called at start of a new timestep. Set the outlet pressure at new time."""
        solver = self.pdesystems['Navier-Stokes']
        u_mean = self.inflow_t_spline(self.t)[0]*695./750./self.A0
        for val in self.inflow.itervalues():
            val.u_mean = u_mean
        info_green('UMEAN = {0:2.5f} at time {1:2.5f}'.format(u_mean, self.t))
        self.p_out1.p = assemble(dot(solver.u_, self.n)*ds(1))
        info_green('Pressure outlet 2 = {0:2.5f}'.format(self.p_out1.p))
        self.p_out2.p = assemble(dot(solver.u_, self.n)*ds(3))
        info_green('Pressure outlet 3 = {0:2.5f}'.format(self.p_out2.p))

if __name__ == '__main__':
    from cbc.cfd.icns import NSFullySegregated, NSSegregated, solver_parameters
    import time
    set_log_active(True)
    problem_parameters['viscosity'] = 0.00345
    problem_parameters['T'] = 0.01
    problem_parameters['dt'] = 0.01
    solver_parameters = recursive_update(solver_parameters, 
    dict(degree=dict(u=1,u0=1,u1=1,u2=1),
         pdesubsystem=dict(u=101, p=101, velocity_update=101), 
         linear_solver=dict(u='bicgstab', p='gmres', velocity_update='bicgstab'), 
         precond=dict(u='jacobi', p='amg', velocity_update='ilu'))
         )
    
    problem = aneurysm(problem_parameters)
    solver = NSFullySegregated(problem, solver_parameters)
    #solver.pdesubsystems['u'].prm['monitor_convergence'] = True
    #solver.pdesubsystems['velocity_update'].prm['monitor_convergence'] = True
    #solver.pdesubsystems['p'].prm['monitor_convergence'] = True
    #solver.pdesubsystems['u0'].prm['monitor_convergence'] = True
    #solver.pdesubsystems['u1'].prm['monitor_convergence'] = True
    #solver.pdesubsystems['u2'].prm['monitor_convergence'] = True
    #solver.pdesubsystems['u0_update'].prm['monitor_convergence'] = True
    #solver.pdesubsystems['u1_update'].prm['monitor_convergence'] = True
    #solver.pdesubsystems['u2_update'].prm['monitor_convergence'] = True
    t0 = time.time()
    problem.solve()
    t1 = time.time() - t0

    print list_timings()

    dump_result(problem, solver, t1, 0)
    