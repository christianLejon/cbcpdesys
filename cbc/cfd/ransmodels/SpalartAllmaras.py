__author__ = "Mikael Mortensen <Mikael.Mortensen@ffi.no>"
__date__ = "2010-09-06"
__copyright__ = "Copyright (C) 2010 " + __author__
__license__  = "GNU GPL version 3 or any later version"
"""

    Spallart Allmaras turbulence model

"""
from TurbSolver import *
from cbc.cfd.tools.Eikonal import Eikonal

class SpalartAllmaras(TurbSolver):
    """Spallart Allmaras turbulence model."""
    def __init__(self, problem, parameters, model='SpalartAllmaras'):
        parameters['model'] = model
        self.classical = True
        TurbSolver.__init__(self, 
                            system_composition=[['nu_tilde']],
                            problem=problem, 
                            parameters=parameters)
                        
    def define(self):
        """Set up linear algebra schemes and their boundary conditions."""
        DQ, DQ_NoBC = DerivedQuantity, DerivedQuantity_NoBC
        V, NS = self.V['dq'], self.Turb_problem.NS_solver
        ns = {'u_':NS.u_}
        NS.schemes['derived quantities'] = [DQ_NoBC(NS, 'Omega_', NS.S,
                            "sqrt(2.*inner(omega(u_), omega(u_)))", ns)]
        self.Omega_ = NS.Omega_
        # Constant
        self.Cw1 = self.Cb1/self.kappa**2 + (1. + self.Cb2)/self.sigma
        ns = vars(self)
        self.schemes['derived quantities'] = [
              DQ(self, 'chi_', V, "nu_tilde_/nu", ns),
              DQ(self, 'fv1_', V, "chi_**3/(chi_**3 + Cv1**3)", ns),
              DQ(self, 'nut_', V, "nu_tilde_*fv1_", ns)]
        self.schemes['derived quantities'] += {
            True: lambda: [
                DQ(self, 'fv2_', V, "1. - chi_/(1. + chi_*fv1_)", ns, 
                   bounded=False),
                DQ_NoBC(self, 'St_', V, "Omega_ + nu_tilde_/(kappa*y)**2*fv2_", 
                        ns)],
            False: lambda: [
                DQ(self, 'fv2_', V, "1./(1. + chi_/Cv2)**3", ns,
                   wall_value=1.), 
                DQ(self, 'fv3', V, "(1. + chi_*fv1_)*(1 - fv2_)/chi_", ns, 
                   wall_value=1.),
                DQ_NoBC(self, 'St_', V, 
                        "fv3*Omega_ + nu_tilde_/(kappa*y)**2*fv2_", ns)]
            }[self.classical]()
                
        self.schemes['derived quantities'] += [
            DQ(self, 'r_', V, 
               "nu_tilde_/(Omega_*kappa**2*y**2 + nu_tilde_*fv2_)", ns, 
               wall_value=1.),
            DQ(self, 'g_', V, "r_ + Cw2*(r_**6 - r_)", ns, 
               wall_value=1-self.Cw2(0)),
            DQ(self, 'fw_', V, "g_*((1. + Cw3**6)/(g_**6 + Cw3**6))**(1./6.)", 
               ns, wall_value=1.)]
        
        classname = self.prm['time_integration'] + '_nu_tilde_' + \
                    str(self.prm['scheme']['nu_tilde'])
        self.schemes['nu_tilde'] = eval(classname)(self, 
                                                   self.system_composition[0])
        
        TurbSolver.define(self)
        
    def model_parameters(self):
        for dq in ['nut_', 'r_', 'g_', 'fw_']:
            self.prm['apply'][dq] = self.prm['apply'].get(dq, 'project')
            
        self.model_prm = dict(
            sigma = Constant(2./3.),
            Cv1 = Constant(7.1),
            Cb1 = Constant(0.1355),
            Cb2 = Constant(0.622),
            kappa = Constant(0.4187),
            Cw2 = Constant(0.3),
            Cw3 = Constant(2.),
            Ct3 = Constant(1.2),
            Cv2 = Constant(5.0),
            )
        self.__dict__.update(self.model_prm)
        
    def create_BCs(self, bcs):
        # Compute distance to nearest wall
        self.distance = Eikonal(self.V['dq'], self.boundaries)
        self.y = self.distance.y
        return TurbSolver.create_BCs(self, bcs)
        
# Model

class Steady_nu_tilde_1(TurbModel):
    
    def form(self, nu_tilde, v_nu_tilde, nu_tilde_, fw_, St_, u_, Omega_, nu, 
             y, sigma, Cw1, Cb1, Cb2, **kwargs):
        F = (1./sigma)*(nu + nu_tilde_)*inner(grad(nu_tilde), \
                                              grad(v_nu_tilde))*dx \
            - (Cb2/sigma)*inner(v_nu_tilde, dot(grad(nu_tilde_), \
                                                grad(nu_tilde_)))*dx \
            + inner(v_nu_tilde, dot(grad(nu_tilde), u_))*dx \
            + inner(v_nu_tilde, Cw1*fw_*nu_tilde_/y**2*nu_tilde)*dx \
            - inner(v_nu_tilde, Cb1*St_*nu_tilde_)*dx # Production explicit
        return F

class Steady_nu_tilde_2(TurbModel):
    
    def form(self, nu_tilde, v_nu_tilde, nu_tilde_, fw_, St_, u_, Omega_, nu, 
             dt, y, sigma, Cw1, Cb1, Cb2, **kwargs):
        F = (1./dt)*inner(v_nu_tilde, nu_tilde - nu_tilde_)*dx \
            + (1./sigma)*(nu + nu_tilde_)*inner(grad(nu_tilde), \
                                                grad(v_nu_tilde))*dx \
            - (Cb2/sigma)*inner(v_nu_tilde, dot(grad(nu_tilde_), \
                                                grad(nu_tilde_)))*dx \
            + inner(v_nu_tilde, dot(grad(nu_tilde), u_))*dx \
            + inner(v_nu_tilde, Cw1*fw_*nu_tilde_/y**2*nu_tilde)*dx \
            - inner(v_nu_tilde, Cb1*St_*nu_tilde_)*dx # Production explicit
        return F
