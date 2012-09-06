__author__ = "Mikael Mortensen <mikaem@math.uio.no>"
__date__ = "2010-10-14"
__copyright__ = "Copyright (C) 2010 " + __author__
__license__  = "GNU Lesser GPL version 3 or any later version"
"""
Base class for transport of passive scalars
"""
from cbc.pdesys.PDESystem import *

solver_parameters  = copy.deepcopy(default_solver_parameters)
solver_parameters = recursive_update(solver_parameters, {
    'Schmidt': 0.7,
    'Schmidt_T': 0.7,
    'Prandtl': 7.,
    'familyname': 'Passive-Scalar'
})

class Scalar(PDESystem):
    
    def __init__(self, problem, parameters):
        PDESystem.__init__(self, [['c']], problem, parameters)
        
    def setup(self):
        PDESystem.setup(self)
            
        self.u_   = self.problem.pdesystems['Navier-Stokes'].u_
        self.u_1  = self.problem.pdesystems['Navier-Stokes'].u_1
        self.nut_ = self.problem.pdesystems['Navier-Stokes'].nut_
        self.n = FacetNormal(self.mesh)
        # Fixme. Add timelevels for nut_
        #self.nut_1 = self.problem.pdesystems['Navier-Stokes'].nut_1
        self.nu = Constant(self.problem.prm['viscosity']/self.prm['Prandtl']) 
        if self.nut_:
            self.nu = self.nu + self.nut_/self.prm['Schmidt_T']
        
        # Set coefficients for initial velocity and pressure
        if not self.problem.initialize(self):
            info_red('Initialization not performed for ' + self.prm['familyname'])
            
        # Get the list of boundaries from the problem class
        self.boundaries = self.problem.boundaries
                         
        # Generate boundary conditions using provided boundaries list
        self.bc = self.create_BCs(self.boundaries)
                
        self.define()
                    
    def define(self):
        classname = self.prm['time_integration'] + '_Scalar_' + \
                    str(self.prm['pdesubsystem']['c'])
        self.pdesubsystems['c'] = eval(classname)(vars(self), ['c'], bcs=self.bc['c'])
        
    def create_BCs(self, bcs):
        """Create boundary conditions for scalar
        """
        bcu = {}
        for name in self.system_names:
            bcu[name] = []
            
        for bc in bcs:
            for name in self.system_names:
                V = self.V[name]
                if bc.type() in ('VelocityInlet'):
                    if hasattr(bc, 'func'):
                        assert isinstance(bc.func, dict)
                        add_BC(bcu[name], V, bc, bc.func[name])
                    else:
                        raise TypeError('expected func for VelocityInlet')
                elif bc.type() == 'Wall':
                    if hasattr(bc, 'func'):
                        assert isinstance(bc.func, dict)
                        if 'c' in bc.func:
                            add_BC(bcu[name], V, bc, bc.func[name])
                        else:
                            bcu[name].append(bc)  # Neuman condition
                elif bc.type() in ('ConstantPressure', 'Outlet'):
                    # This bc could be weakly enforced
                    bcu[name].append(bc)
                elif bc.type() == 'Periodic':
                    add_BC(bcu[name], V, bc, None)
                else:
                    info("No assigned boundary condition for %s -- skipping..."
                         %(bc.__class__.__name__))                
        return bcu
        
class ScalarBase(PDESubSystem):
    
    def define(self):
        
        form_args = self.solver_namespace.copy()
        self.exterior = any([bc.type() in ['ConstantPressure', 'Outlet', 'Wall']
                             for bc in self.bcs]) 
        self.Laplace_C = self.solver_namespace['c']
        if self.prm['iteration_type'] == 'Picard':
            self.get_form(form_args)
            if self.exterior: 
                self.F = self.F + self.add_exterior(**form_args)
            self.a, self.L = lhs(self.F), rhs(self.F)
        else:
            # Set up Newton system by switching Function for TrialFunction
            for name in self.sub_system:
                form_args[name + '_'] = self.solver_namespace[name]
            self.get_form(form_args)
            if self.exterior: 
                self.F = self.F + self.add_exterior(**form_args)
            u_ = self.solver_namespace[self.name + '_']
            u  = self.solver_namespace[self.name]
            F_ = action(self.F, coefficient=u_)
            J_ = derivative(F_, u_, u)
            self.a, self.L = J_, F_
    
    def add_exterior(self, c, v_c, n, nu, **kwargs):
        C = self.Laplace_C
        L = []
        for bc in self.bcs:
            if (bc.type() in ('ConstantPressure', 'Outlet') or
                bc.type() == 'Wall' and isinstance(bc, SubDomain)):
                info_green('Assigning weak boundary condition for ' + bc.type())
                L.append(-nu*inner(v_c, dot(grad(C), n))*ds(bc.bid))
                self.exterior_facet_domains = bc.mf                 
            
        return reduce(operator.add, L)
        
class Transient_Scalar_1(ScalarBase):
    
    def form(self, c_, c_1, c, v_c, u_, u_1, nu, dt, **kwargs):
        C = 0.5*(c + c_1)
        U_ = 0.5*(u_ + u_1)
        self.Laplace_C = C
        return (1./dt)*inner(c - c_1, v_c)*dx + inner(dot(U_, grad(C)), v_c)*dx \
               + nu*inner(grad(C), grad(v_c))*dx

class Steady_Scalar_1(ScalarBase):
    
    def form(self, c_, c, v_c, u_, nu, nut_, **kwargs):
        C = 0.5*(c + c_1)
        self.Laplace_C = C
        return inner(dot(u_, grad(c)), v_c)*dx + nu*inner(grad(C), grad(v_c))*dx