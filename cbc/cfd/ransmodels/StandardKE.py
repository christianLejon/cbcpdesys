__author__ = "Mikael Mortensen <Mikael.Mortensen@ffi.no>"
__date__ = "2010-09-06"
__copyright__ = "Copyright (C) 2010 " + __author__
__license__  = "GNU GPL version 3 or any later version"
"""

    Base class for standard k-epsilon turbulence models

"""
from TurbSolver import *
from cbc.cfd.tools.Eikonal import Eikonal

solver_parameters['model'] = 'StandardKE'

class StandardKE(TurbSolver):
    """Base class for standard K-Epsilon turbulence models."""
        
    def define(self):
        DQ, DQ_NoBC = DerivedQuantity, DerivedQuantity_NoBC 
        NS, V = self.Turb_problem.NS_solver, self.V['dq']
        NS.schemes['derived quantities'] = [
            DQ_NoBC(NS, 'Sij_', NS.S, "epsilon(u_)", dict(u_=NS.u_), 
                    bounded=False)]
        self.Sij_ = NS.Sij_
        ns = vars(self)
        self.schemes['derived quantities'] = [
            DQ(self, 'nut_', V, 'Cmu*k_*k_*(1./e_)', ns, apply='project'),
            DQ_NoBC(self, 'T_', V, 'max_(k_*(1./e_), 6.*sqrt(nu*(1./e_)))', ns),
            DQ(self, 'P_', V, '2.*inner(grad(u_), Sij_)*nut_', ns, bounded=False)]
        TurbSolver.define(self)
        
    def model_parameters(self):
        info('Setting parameters for standard K-Epsilon model')
        for dq in ['nut_', 'T_']:
            # Specify projection as default
            # (remaining DQs are use_formula by default)
            self.prm['apply'][dq] = self.prm['apply'].get(dq, 'project')    
            
        self.model_prm = dict(
            Cmu = Constant(0.09),
            Ce1 = Constant(1.44),
            Ce2 = Constant(1.92),
            sigma_e = Constant(1.3),
            sigma_k = Constant(1.0),
            e_d = Constant(0.5))
        self.__dict__.update(self.model_prm)
    
    def create_BCs(self, bcs):
        # Compute distance to nearest wall
        self.distance = Eikonal(self.V['dq'], self.boundaries)
        self.y = self.distance.y
        return TurbSolver.create_BCs(self, bcs)
