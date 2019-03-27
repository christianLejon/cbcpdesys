__author__ = "Mikael Mortensen <mikaem@math.uio.no>"
__date__ = "2011-01-21"
__copyright__ = "Copyright (C) 2010-2016 " + __author__
__license__  = "GNU Lesser GPL version 3 or any later version"
"""Super class for solving systems of PDEs."""

#from cbc.pdesys.PDESubSystems import *
#import PDESubSystems
from cbc.pdesys.PDESubSystems import *
import collections as c
import pdb
from dolfin import *
from functools import reduce


default_solver_parameters = {
    'degree': c.defaultdict(lambda: 1),
    'family': c.defaultdict(lambda: 'CG'),
    'space' : c.defaultdict(lambda: FunctionSpace),
    'element' : c.defaultdict(lambda: FiniteElement),
    'constrained_domain': None,
    'constriction': c.defaultdict(lambda: dict(constrained_domain=None)),
    'pdesubsystem': c.defaultdict(lambda: 1),
    'iteration_type': 'Picard',   # or 'Newton'
    'linear_solver': c.defaultdict(lambda: 'lu'),
    'precond': c.defaultdict(lambda: 'default'),
    'omega': c.defaultdict(lambda: 1.0), # Underrelaxation
    'monitor_convergence': c.defaultdict(lambda: False), # Monitor Krylov solver
    'time_integration': 'Steady', # or 'Transient'
    'max_iter': 1,
    'max_err': 1e-7,
    'T': 1., # End time for simulation,
    'dt': 0.001,  # timestep,
    'familyname': 'default'    # Name of PDESystem (e.g., Navier-Stokes)
}

def split_Function(f):
    """
    Split a Function in a mixed space into its sub-Functions.
    Add extra info in the sub-Functions so that they have
    reference to the vector of the parent Function and a slice
    defining their part of the vector of the parent Function.
    """
    def add_info(sub_f, parent_f, start_index):
        """
        Modify a subFunction sub_f belonging to a parent Function
        parent_f, where sub_f is associated with the slice
        i_start to i_stop of parent_f's vector.
        Tasks: Add slice info to sub_f, reference to parent_f.vector(),
        and a method get_vector_slice() to return the associated
        NumPy array slice.
        """
        end_index = start_index + sub_f.function_space().dim()
        sub_f.parent_vector = parent_f.vector()
        sub_f.parent_vector_slice = slice(start_index, end_index)
        #sub_f.parent_vector_slice = slice(start_index, end_index - 1) # CL - ett försök att få rätt på "var" matris problemet. Men det tycks ju inte vara split kommandot det är fel på. Båda formerna (split() och .split) av split förekommer i original koden.
        setattr(sub_f, 'get_array_slice', 
                lambda self: \
                self.parent_vector.array()[self.parent_vector_slice])
        return end_index

    sub_fs = f.split() # Detta är originalet! Ej ufl.split, och det tycks fungera till att börja med iallafall
    #sub_fs = split(f)
    start_index = 0
    for sub_f in sub_fs:
        start_index = add_info(sub_f, f, start_index)
    return sub_fs
        
class PDESystem:
    """Base class for solving a system of equations.
    The argument system_composition is a list of lists defining the systems primary 
    variables and their composition. For example
    
        system_composition = [['u', 'p']] is used for a coupled Navier-Stokes solver
        system_composition = [['u'], ['p']] is used for a segregated Navier-Stokes
                                            solver
    
    Here u and p represent velocity and pressure respectively. 
    
        system_composition = [['u', 'p'], ['k', 'e']] can be used for a k-epsilon 
        Reynolds Averaged NS-solver, where k and e represent kinetic energy and 
        dissipation respectively. Here the variables within each sub-system, 
        ['u', 'p'] and ['k', 'e'] are coupled, but the two sub-systems are solved 
        in a segregated manner: ['u', 'p'] then ['k', 'e'] etc.
        
    The parameters dictionary provided by the user contains information about
    the functionspace (degree, family, dimension) of each variable.
    """
    def __init__(self, system_composition, problem, parameters):
        self.system_composition = system_composition         # Total system comp.
        self.system_names = []                               # Compounds solved for
        self.names = []                                      # All components
        self.prm = parameters
        #if isinstance(problem, dolfin.cpp.mesh.Mesh):
        if isinstance(problem, Mesh):    
            self.problem = None
            self.mesh = problem
        else:
            self.problem = problem
            self.mesh = problem.mesh
            self.prm['dt'] = problem.prm['dt']
            self.prm['time_integration'] = problem.prm['time_integration']
            self.dt = Constant(self.prm['dt'])
         
        # The following parameters are used by the solve functionality 
        # in case there's no problem instance used to perform the solve.
        self.t = 0                                           # Simulation time
        self.tstep = 0                                       # Time step
        self.total_number_iters = 0                          #
        self.num_timesteps = 0                               #
                
        for sub_system in system_composition: 
            system_name = ''.join(sub_system) 
            self.system_names.append(system_name)
            for n in sub_system:       # Run over all individual components
                self.names.append(n)
                
        # Create all FunctionSpaces, Functions, Test-TrialFunctions etc.
        self.setup()

    def setup(self):
        # Need to modify symmetry for TensorFunctionSpace because FunctionSpace, 
        # VectorFunctionSpace and TensorFunctionSpace do not take the same 
        # arguments. symmetry need to be {} for the first two and 
        # {'symmetry': symmetry[name]} for the TensorFunctionSpace
        cons = self.prm['constriction']
        constrained_domain = None
        if self.problem:
            for bc in self.problem.boundaries:
                if bc.bc_type == 'Periodic':
                    self.prm['constrained_domain'] = bc
                    for name in self.names + ['dq']:
                        cons[name]['constrained_domain'] = bc

        self.define_function_spaces(self.mesh, self.prm['degree'], 
                              self.prm['space'], self.prm['element'], self.prm['family'], cons)
        self.setup_subsystems()

        if self.prm['time_integration'] == 'Steady':
            self.add_function_on_timestep()
        else:
            self.add_function_on_timestep(['', '1', '2'])
        self.pdesubsystems = dict((name, None) for name in self.system_names)
        self.normalize     = dict((name, None) for name in self.system_names)
        self.bc            = dict((name,   []) for name in self.system_names)
        if hasattr(self.problem, 'add_pdesystem'):
            self.problem.add_pdesystem(self, self.prm['familyname'])

    def define_function_spaces(self, mesh, degree, space, element, family, cons):
        """Define functionspaces for names and system_names"""
        # 1. Skapa testfunktion som inte är noll på randen och se om den 2. Kolla hur många punkter som ligger på randen 3. Varför är Sij full längd och var skapas den ?
        V = self.V = dict((name, space[name](mesh, family[name], degree[name],
                           **cons[name])) for name in self.names + ['dq'])
        #V = self.V = dict((name, space[name](mesh, family[name], degree[name])) for name in self.names + ['dq']) # Test wo/ **cons to get correct size
        # Add function space for compound functions for the sub systems
        # First create individual Finite/VectorElements
        FE = self.FE = dict((name, element[name](family[name], mesh.ufl_cell(), degree[name])) for name in self.names)
        # Then make the mixed element if several sub-systems (of all individual Finite/VectorElements)
        FE.update(dict({self.system_names[0]: reduce(lambda x,y:x*y, FE.values())}))
        # And Finally add the corresponding Mixed functionspace in V that holds functionspaces for the compound system.
        V.update(dict({self.system_names[0]: FunctionSpace(mesh, FE[self.system_names[0]])}))
        # The above lines were tested for 'u' 'p' 'up'. If several MixedFunctionSpaces are needed the codework must be modified.

        #Original code (that adds functionspaces directly, deprecated in later FEniCS versions)
        #V.update(dict((sys_name, MixedFunctionSpace([V[name] for name in sub_sys]))
        #    for sub_sys, sys_name in zip(self.system_composition,
        #                                 self.system_names) 
        #                                 if len(sub_sys) > 1)) #sub_sys is ['u', 'p'] and sys_name = 'up'.  in Lejon_demo.py. 

    def add_function_on_timestep(self, timesteps=['']):
        """
        Add solution functions for timelevels determined by timesteps.
        timesteps=['', '1', '2'] generates functions on timesteps N, N-1 and N-2
        and puts them in dictionaries q_, q_1 and q_2. Vectors of dofs are 
        generated in x_, x_1 and x_2. Short forms are also generated in
        name_, name_1 and name_2, e.g. for velocity 'u', we create short forms
        u_, u_1 and u_2 as pointers to q_['u'], q_1['u'] and q_2['u'].
        """
        V, sys_names, sys_comp = \
               self.V, self.system_names, self.system_composition
                               
        for timestep in timesteps:
            funcname = 'q_' + timestep
            
            if not hasattr(self, funcname):
                func = dict((name, Function(V[name])) for name in sys_names)

                # Split the various compound functions into sub functions
                for sub_sys, sys_name in zip(sys_comp, sys_names):
                    if len(sub_sys) > 1:
                        func.update(dict((name, subFunction)
                                  for name, subFunction in 
                                  zip(sub_sys, split_Function(func[sys_name]))))
                setattr(self, funcname, func)
            
                # Short forms
                for key, value in func.items(): 
                    setattr(self, key + '_' + timestep, value) 
                
                setattr(self, 'x_' + timestep, dict((name, func[name].vector()) 
                                                for name in self.system_names))
                
                self.num_timesteps += 1
                            
    def setup_subsystems(self):
        V, sys_names, sys_comp = \
               self.V, self.system_names, self.system_composition
               
        # Create compound functions for the various sub systems
        q = dict((name, TrialFunction(V[name])) for name in sys_names)
        v = dict((name, TestFunction(V[name]))  for name in sys_names)
        
        # Split the various compound functions into sub functions
        for sub_sys, sys_name in zip(sys_comp, sys_names):
            if len(sub_sys) > 1:
                q.update(dict((name, subTrialFunction)
                              for name, subTrialFunction in 
                              zip(sub_sys, split(q[sys_name]))))
                v.update(dict((name, subTestFunction)
                              for name, subTestFunction in
                              zip(sub_sys, split(v[sys_name]))))

        self.qt, self.vt = q, v
        # Short forms
        for key, value in v.items(): setattr(self, 'v_' + key, value)
        for key, value in q.items(): setattr(self, key, value) 
        
    def initialize(self, q0):
        """Initialize system_names functions."""
        for sys_name in self.system_names:
            for t in range(self.num_timesteps):
                exec('self.x_%s[sys_name][:] = q0[sys_name].vector()[:]'
                     %(str(t).replace('0','')))
                
    def Transient_update(self):
        """Update system_names functions to the next timestep."""
        for sys_name in self.system_names:
            for t in range(self.num_timesteps - 1, 0, -1):
                exec('self.x_%s[sys_name][:] = self.x_%s[sys_name][:]'      
                     %(str(t), str(t - 1).replace('0', '')))

    def create_BCs(self, bcs):
        """Create boundary conditions for self.system_names based on boundaries
        in list bcs. Boundary conditions can be placed in bc.func as a function
        or dictionary using the pdesubsystem names as keys. 
        If func is not provided for Wall, then the default value of 1e-12 is 
        used for all components.
        """
        bcu = {}
        for name in self.system_names:
            bcu[name] = []
            
        for name in self.system_names:
            V = self.V[name]
            for bc in bcs:
                if bc.type() in ('VelocityInlet', 'Wall'):
                    if hasattr(bc, 'func'):
                        if isinstance(bc.func, dict):
                            add_BC(bcu[name], V, bc, bc.func[name])
                        else:
                            add_BC(bcu[name], V, bc, bc.func)
                    else:
                        if bc.type() == 'Wall': # Default is zero on walls
                            if isinstance(V.ufl_element(), FiniteElement):
                                func = Constant(1e-12)
                            elif isinstance(V.ufl_element(), (MixedElement,
                                                VectorElement)):
                                if not all([V.sub(0).dim() == V.sub(i).dim() 
                                       for i in range(1, V.num_sub_spaces())]):
                                    error("You need to subclass create_BCs for MixedFunctionSpaces consisting of not equal FunctionSpaces")
                                func = Constant((1e-12, )*V.num_sub_spaces())
                            elif isinstance(V.ufl_element(), TensorElement):
                                func = Expression((('1.e-12', )*V.ufl_cell().topological_dimension(), )*V.ufl_cell().topological_dimension())
                                # I assume that the topolocical dimension is correct... see ufl documentation for V.cell().d. The old expression: func = Expression((('1.e-12', )*V.cell().d, )*
                                #                  V.cell().d)
                            else:
                                raise NotImplementedError
                            add_BC(bcu[name], V, bc, func)
                        elif bc.type() == 'VelocityInlet':
                            raise TypeError('expected func for VelocityInlet')        
                elif bc.type() in ('ConstantPressure', 'Outlet'):
                    # This bc could be weakly enforced
                    bcu[name].append(bc)
                else:
                    #info("No assigned boundary condition for %s -- skipping..."
                    #     %(bc.__class__.__name__))
                    info("No assigned boundary condition to subdomain #%s# with #%s# type boundary condition -- skipping..."
                         % (bc.__class__.__name__, bc.type()))
        return bcu
        
    def solve(self, func = 'advance', redefine = True, **kwargs):
        """Call either:
            solve_Transient_advance      --  Advance solution to time T
            solve_Steady_advance         --  Steady iteration
        or user defined 
            solve_Transient_something
            solve_Steady_something   
        Will only be used in case there is no Problem class with
        solve functionality. Will probably be removed.
        """
        self.prm = recursive_update(self.prm, kwargs)
        if redefine: 
            self.define()
        return eval('self.solve_%s_%s()' %(self.prm['time_integration'], func))
            
    def solve_Transient_advance(self, max_iter=None, max_err=None, 
                                logging=True):
        """Advance solution in steps of dt to time T.
        Will only be used in case there is no Problem class with
        solve functionality. Will probably be removed.
        """
        while self.t < (self.prm['T'] - self.tstep*DOLFIN_EPS):
            self.t = self.t + self.dt(0)
            self.tstep = self.tstep + 1
            self.prepare()
            
            # Perform up to max_iter iterations on given timestep.
            err = self.solve_inner(max_iter=max_iter or self.prm['max_iter'],
                                   max_err=max_err or self.prm['max_err'],
                                   logging=logging)
            
            self.solve_derived_quantities()
            
            self.Transient_update()    
        
            self.update()

            info_green('Time = %s, End time = %s' %(self.t, self.prm['T']))
            
        return err
        
    def solve_Steady_advance(self, max_iter=None, max_err=None, logging=True):
        """Iterate solution max_iter iterations or until convergence.
        Will only be used in case there is no Problem class with
        solve functionality. Will probably be removed.
        """
        self.prepare()

        err = self.solve_inner(max_iter=max_iter or self.prm['max_iter'],
                               max_err=max_err or self.prm['max_err'],
                               logging=logging)
                            
        self.solve_derived_quantities()
        
        self.update()
        
        return err
                
    def solve_inner(self, max_iter=1, max_err=1e-7, logging=True):
        err, j = solve_nonlinear([self.pdesubsystems[name] 
                                  for name in self.system_names],
                                  max_iter=max_iter, max_err=max_err,
                                  logging=max_iter>1)
        self.total_number_iters += j
        return err

    def solve_derived_quantities(self):
        """
        Solve for derived quantities found
        in dictionary pdesubsystems['derived quantities'].
        """
        if 'derived quantities' in self.pdesubsystems:
            for pdesubsystem in self.pdesubsystems['derived quantities']:
                pdesubsystem.solve()
    
    def prepare(self):
        """Called at start of iterations over pdesystems"""
        pass
    
    def update(self):
        """Called at end of of iterations over pdesystems"""
        pass
    
    def define(self):
        """Hook up pdesubsystems"""
        pass
    
    def add_pdesubsystem(self, pdesubsystem, sub_system, **kwargs):
        name = ''.join(sub_system)
        if name in self.system_names:
            self.pdesubsystems[name] = pdesubsystem(vars(self), sub_system, **kwargs)
        else:
            info_red('Wrong sub_system!')
        
    def info(self):
        print("Base class for solving a system of PDEs")
        
