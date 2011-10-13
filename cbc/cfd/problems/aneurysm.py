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
    
    def __init__(self, bid, mf, func=None):
        SubDomain.__init__(self)
        self.bid = bid
        self.mf = mf
        if func:
            self.func = func
            
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
    """
    """    
    def __init__(self, parameters):
        NSProblem.__init__(self, parameters=parameters)
        self.mesh = Mesh("../data/100_1314k.xml.gz")
        #self.mesh = Mesh("../data/m12_dns_2m.xml")
        self.bc_markers = self.mark_boundary()
        self.n = FacetNormal(self.mesh)        
        self.boundaries = self.create_boundaries()
                
    def mark_boundary(self):
        bc_markers = MeshFunction("uint", self.mesh, 2)
        file_in = File("../data/100_1314k_boundary.xml.gz")
        #file_in = File("../data/m12_dns_2m_bcs.xml")        
        file_in >> bc_markers
        return bc_markers
        
    def initialize(self, pdesystem):
        NSProblem.initialize(self, pdesystem, Initdict(u = ('0', '0', '0'),
                                                       p = ('0')))
        
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
        self.u0 = assemble(-n[0]*ds(2), mesh=self.mesh, exterior_facet_domains=self.bc_markers)
        self.u1 = assemble(-n[1]*ds(2), mesh=self.mesh, exterior_facet_domains=self.bc_markers)
        self.u2 = assemble(-n[2]*ds(2), mesh=self.mesh, exterior_facet_domains=self.bc_markers)
        
        self.A0 = assemble(Constant(1.)*ds(2), mesh=self.mesh, exterior_facet_domains=self.bc_markers)
        
        # For now I need to explicitly set u1, u2 and u3. Should be able to fix using just u.
        self.inflow = {'u': Expression(('u0*u_mean', 'u1*u_mean', 'u2*u_mean'), 
                                  u0=self.u0, u1=self.u1, u2=self.u2, u_mean=0),
                       'u0': Expression(('u0*u_mean'), u0=self.u0, u_mean=0),
                       'u1': Expression(('u1*u_mean'), u1=self.u1, u_mean=0),
                       'u2': Expression(('u2*u_mean'), u2=self.u2, u_mean=0)}
        self.p_out1 = Expression('p', p=0)
        self.p_out2 = Expression('p', p=0)
        self.p_out3 = Expression('p', p=0)
        walls = Walls(1, self.bc_markers)
        inlet = Inlet(2, self.bc_markers, self.inflow)
        pressure1 = PressureOutlet(3, self.bc_markers, {'p': self.p_out1})
        pressure2 = PressureOutlet(4, self.bc_markers, {'p': self.p_out2})
        pressure3 = PressureOutlet(5, self.bc_markers, {'p': self.p_out3})
        
        #return [walls, inlet, pressure1, pressure2]
        return [walls, inlet, pressure1, pressure2, pressure3]
        
    def prepare(self, pdesystems):
        """Called at start of a new timestep. Set the outlet pressure at new time."""
        NS_solver = pdesystems[0]
        u_mean = self.inflow_t_spline(self.t)[0]*695./750./self.A0
        for val in self.inflow.itervalues():
            val.u_mean = u_mean
        info_green('UMEAN = {} at time {}'.format(u_mean, self.t))
        self.p_out1.p = assemble(dot(NS_solver.u_, self.n)*ds(3), exterior_facet_domains=self.bc_markers)
        info_green('Pressure outlet 3 = {}'.format(self.p_out1.p))
        self.p_out2.p = assemble(dot(NS_solver.u_, self.n)*ds(4), exterior_facet_domains=self.bc_markers)
        info_green('Pressure outlet 4 = {}'.format(self.p_out2.p))
        self.p_out3.p = assemble(dot(NS_solver.u_, self.n)*ds(5), exterior_facet_domains=self.bc_markers)
        info_green('Pressure outlet 5 = {}'.format(self.p_out3.p))

if __name__ == '__main__':
    from cbc.cfd.icns import NSFullySegregated, NSSegregated, solver_parameters
    set_log_active(True)
    problem_parameters['viscosity'] = 0.00345
    problem_parameters['T'] = 0.5
    problem_parameters['dt'] = 0.001
    solver_parameters = recursive_update(solver_parameters, 
    dict(degree=dict(u=1),
         pdesubsystem=dict(u=101, p=101, velocity_update=101), 
         linear_solver=dict(u='bicgstab', p='gmres', velocity_update='bicgstab'), 
         precond=dict(u='jacobi', p='amg', velocity_update='ilu'))
         )
    
    NS_problem = aneurysm(problem_parameters)
    NS_solver = NSFullySegregated(NS_problem, solver_parameters)
    #NS_solver.pdesubsystems['u'].prm['monitor_convergence'] = True
    #NS_solver.pdesubsystems['velocity_update'].prm['monitor_convergence'] = True
    NS_solver.pdesubsystems['p'].prm['monitor_convergence'] = True
    NS_solver.pdesubsystems['u0'].prm['monitor_convergence'] = True
    NS_solver.pdesubsystems['u1'].prm['monitor_convergence'] = True
    NS_solver.pdesubsystems['u2'].prm['monitor_convergence'] = True
    #NS_solver.pdesubsystems['u0_update'].prm['monitor_convergence'] = True
    #NS_solver.pdesubsystems['u1_update'].prm['monitor_convergence'] = True
    #NS_solver.pdesubsystems['u2_update'].prm['monitor_convergence'] = True
    NS_problem.solve()

    print summary()
