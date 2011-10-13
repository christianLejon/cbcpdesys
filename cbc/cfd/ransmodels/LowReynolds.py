__author__ = "Mikael Mortensen <Mikael.Mortensen@ffi.no>"
__date__ = "2010-09-06"
__copyright__ = "Copyright (C) 2010 " + __author__
__license__  = "GNU GPL version 3 or any later version"
"""

    LowReynolds turbulence models

"""
from TurbSolver import *
from cbc.cfd.tools.Eikonal import Eikonal

class LowReynolds(TurbSolver):
    """Base class for low-Reynolds number turbulence models."""
        
    def define(self):
        """ Set up linear algebra schemes and their boundary conditions """
        V,  NS = self.V['dq'], self.Turb_problem.NS_solver 
        model = self.Turb_problem.prm['Model']
        DQ, DQ_NoBC = DerivedQuantity, DerivedQuantity_NoBC
        ns = dict(u_=NS.u_)
        NS.schemes['derived quantities'] = [
            DQ_NoBC(NS, 'Sij_', NS.S, "0.5*(grad(u_)+grad(u_).T)", ns, 
                    bounded=False),
            DerivedQuantity_grad(NS, 'd2udy2_', V, "-grad(u_)", ns, 
                                 bounded=False)]        
        self.Sij_ = NS.Sij_; self.d2udy2_ = NS.d2udy2_
        ns = vars(self)   # No copy. It is updated automatically.
        self.schemes['derived quantities'] = dict(
            LaunderSharma=lambda :[
                DQ_NoBC(self, 'D_', V, 
                        "nu/2.*(1./k_)*inner(grad(k_), grad(k_))", ns),
                DQ(self, 'fmu_', V, "exp(-3.4/(1. + (k_*k_/nu/e_)/50.)**2)", 
                   ns, wall_value=exp(-3.4)),
                DQ(self, 'f2_', V, "1. - 0.3*exp(-(k_*k_/nu/e_)**2)", ns, 
                   wall_value=0.7),
                DQ(self, 'nut_', V, "Cmu*fmu_*k_*k_*(1./e_)", ns),
                DQ(self, 'E0_', V, "2.*nu*nut_*dot(d2udy2_, d2udy2_)", ns)],
            JonesLaunder=lambda :[
                DQ_NoBC(self, 'D_', V, 
                        "nu/2.*(1/k_)*inner(grad(k_), grad(k_))", ns),
                DQ(self, 'fmu_', V, "exp(-2.5/(1. + (k_*k_/nu/e_)/50.))", 
                   ns, wall_value=exp(-2.5)),
                DQ(self, 'f2_', V, "(1. - 0.3*exp(-(k_*k_/nu/e_)**2))", 
                   ns, wall_value=0.7),
                DQ(self, 'nut_', V, "Cmu*fmu_*k_*k_*(1./e_)", ns),
                DQ(self, 'E0_', V, "2.*nu*nut_*dot(d2udy2_, d2udy2_)", ns)],
            Chien=lambda :[
                DQ_NoBC(self, 'D_', V, "2.*nu*k_/y**2", ns),
                DQ(self, 'yplus_', V, "y*Cmu**(0.25)*k_**(0.5)/nu", ns),
                DQ(self, 'fmu_', V, "(1. - exp(-0.0115*yplus_))", ns),
                DQ(self, 'f2_', V, "(1. - 0.22*exp(-(k_*k_/nu/e_/6.)**2))", 
                   ns, wall_value=0.78),
                DQ(self, 'nut_', V, "Cmu*fmu_*k_*k_*(1./e_)", ns),
                DQ_NoBC(self, 'E0_', V, "-2.*nu*e_/y**2*exp(-yplus_/2.)", ns,
                        bounded=False)] # minus?
            )[model]() # Note. lambda is used to delay the construction of all the DQ objects until we index by [model] and call the lambda function
        
        TurbSolver.define(self)
                
    def model_parameters(self):
        model = self.Turb_problem.prm['Model']
        info('Setting parameters for model %s' %(model))
        for dq in ('nut_',):
            self.prm['apply'][dq] = self.prm['apply'].get(dq, 'project')
        self.Turb_problem.NS_solver.prm['apply']['d2udy2_'] = 'project' 

        self.model_prm = dict(
            Cmu = 0.09,
            sigma_e = 1.30,
            sigma_k = 1.0,
            e_nut = 1.0,
            e_d = 0.,
            f1 = 1.0)
        Ce1_Ce2 = dict(
            LaunderSharma=dict(Ce1 = 1.44, Ce2 = 1.92),
            JonesLaunder=dict(Ce1 = 1.55, Ce2 = 2.0),
            Chien=dict(Ce1 = 1.35, Ce2 = 1.80))
        self.model_prm.update(Ce1_Ce2[model])
        # wrap in Constant objects:
        for name in self.model_prm:
            self.model_prm[name] = Constant(self.model_prm[name])
            
        self.__dict__.update(self.model_prm)

    def create_BCs(self, bcs):
        # Compute distance to nearest wall
        self.distance = Eikonal(self.V['dq'], self.boundaries)
        self.y = self.distance.y
        return TurbSolver.create_BCs(self, bcs)
