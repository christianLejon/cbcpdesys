__author__ = "Mikael Mortensen <Mikael.Mortensen@ffi.no>"
__date__ = "2011-12-02"
__copyright__ = "Copyright (C) 2011 " + __author__
__license__  = "GNU GPL version 3 or any later version"

from NSProblem import *
from scipy.interpolate import InterpolatedUnivariateSpline as ius
from scipy.interpolate import splrep, splev
from numpy import array, zeros, floor, ceil
from aneurysm import MCAtime, MCAval

A = [1, -0.23313344, -0.11235758, 0.10141715, 0.06681337, -0.044572343, -0.055327477, 0.040199067, 0.01279207, -0.002555173, -0.006805238, 0.002761498, -0.003147682, 0.003569664, 0.005402948, -0.002816467, 0.000163798, 8.38311E-05, -0.001517142, 0.001394522, 0.00044339, -0.000565792, -6.48123E-05] 

B = [0, 0.145238823, -0.095805132, -0.117147521, 0.07563348, 0.060636658, -0.046028338, -0.031658495, 0.015095811, 0.01114202, 0.001937877, -0.003619434, 0.000382924, -0.005482582, 0.003510867, 0.003397822, -0.000521362, 0.000866551, -0.001248326, -0.00076668, 0.001208502, 0.000163361, 0.000388013]

counter = 0 
N = 100 

def time_dependent_velocity(t): 
  velocity = 0 
  for k in range(len(A)): 
    velocity += A[k]*cos(2*pi*k*t)
    velocity += B[k]*sin(2*pi*k*t)
  return velocity

class InflowData(object):

    def __init__(self, problem):
        self.mesh = problem.mesh
        self.problem = problem
        self.velocity = problem.velocity
        self.val = self.velocity 
        self.stationary = problem.stationary
        self.t = 0 
        self.N = 100 

    def __call__(self, x, ufc_cell):
        cell = Cell(self.mesh, ufc_cell.index)
        n = cell.normal(ufc_cell.local_facet)

        global counter
        global N 

        if self.problem.t > self.t and not self.problem.stationary: 
            self.t = self.problem.t 
            self.val = self.velocity*time_dependent_velocity(self.t)
        if self.problem.stationary and counter <= N: 
            if self.problem.t > self.t: 
                self.t = self.problem.t 
                counter += 1 
                self.val = float(self.velocity*self.counter)/self.N
                print self.val, self.velocity, self.counter, self.N 
         
        val = self.val 
        return [-n.x()*val, -n.y()*val, -n.z()*val]

class InflowVec(Expression):
    def __init__(self, problem):
        self.data = InflowData(problem)
    def eval_cell(self, values, x, ufc_cell):
        values[:] = self.data(x, ufc_cell)
    def value_shape(self):
        return 3,

class InflowComp(Expression):
    def __init__(self, problem, component):
        self.data = InflowData(problem)
        self.component = component
    def eval_cell(self, values, x, ufc_cell):
        values[0] = self.data(x, ufc_cell)[self.component]

class Challenge(NSProblem):
    
    def __init__(self, parameters):
        NSProblem.__init__(self, parameters=parameters)

        self.mesh = Mesh("/home/kent-and/Challenge/mesh_500k.xml.gz")

        self.testcase = self.prm["test_case"] 

        self.flux = 0
        if self.testcase == 1: 
	    self.flux = 5.13 
        elif self.testcase == 2:
	    self.flux = 6.41 
        elif self.testcase == 3:
	    self.flux = 9.14 
        elif self.testcase == 4:
	    self.flux = 11.42 

        self.stationary = self.prm["time_integration"] == 'Steady'

        self.boundaries = self.create_boundaries()

        self.prm['dt'] = self.prm['T']/ceil(self.prm['T']/0.2/MPI.min(self.mesh.hmin()))

        print self.prm['dt']
        
        # To initialize solution set the dictionary q0: 
        #self.q0 = Initdict(u = ('0', '0', '0'), p = ('0')) # Or not, zero is default anyway
        
    def create_boundaries(self):
        # Define the spline for the heart beat
        #self.inflow_t_spline = ius(MCAtime, MCAval)
        
        # Preassemble normal vector on inlet
        #n = self.n = FacetNormal(self.mesh)        
        #self.normal = [assemble(-n[i]*ds(4), mesh=self.mesh) for i in range(3)]
        
        # Area of inlet 
        #self.A0 = assemble(Constant(1.)*ds(4), mesh=self.mesh)
        one = Constant(1)
	self.V0 = assemble(one*dx, mesh=self.mesh)
	self.A0 = assemble(one*ds(0), mesh=self.mesh)
	self.A1 = assemble(one*ds(1), mesh=self.mesh)
	self.A2 = assemble(one*ds(2), mesh=self.mesh)

	print "Volume of the geometry is (dx)   ", self.V0 
	print "Areal  of the no-slip is (ds(0)  ", self.A0 
	print "Areal  of the inflow is (ds(1))  ", self.A1 
	print "Areal  of the outflow is (ds(2)) ", self.A2 

	self.velocity = self.flux / self.A1 

        # Characteristic velocity (U) in the domain (used to determine timestep)
        self.U = self.velocity*16  
        h  = MPI.min(self.mesh.hmin())
        print "Characteristic velocity set to", self.U
        print "mesh size          ", h
        print "velocity at inflow ", self.velocity
        print "Number of cells    ", self.mesh.num_cells()
        print "Number of vertices ", self.mesh.num_vertices()
        
        # Create dictionary used for Dirichlet inlet conditions. Values are assigned in prepare
        #self.inflow = {'u' : Constant((0, 0, 0)),
        #               'u0': Constant(0),
        #               'u1': Constant(0),
        #               'u2': Constant(0)}

        self.inflow = {'u': InflowVec(self),
                       'u0': InflowComp(self, 0),
                       'u1': InflowComp(self, 1),
                       'u2': InflowComp(self, 2)}

        # Pressures on outlets are specified by DirichletBCs
        self.p_out1 = Constant(0)

        # Specify the boundary subdomains and hook up dictionaries for DirichletBCs
        walls     = MeshSubDomain(0, 'Wall')
        inlet     = MeshSubDomain(1, 'VelocityInlet', self.inflow)
        pressure1 = MeshSubDomain(2, 'ConstantPressure', {'p': self.p_out1})
        
        return [walls, inlet, pressure1]
        
    def prepare(self):
        """Called at start of a new timestep."""
        #t = self.t - floor(self.t/1002.0)*1002.0
        #u_mean = self.inflow_t_spline(t)[0]/self.A0        
        #self.inflow['u'].assign(Constant(u_mean*array(self.normal)))
        #for i in range(3):
        #    self.inflow['u'+str(i)].assign(u_mean*self.normal[i])

    def update(self):
        if self.tstep % 10 == 0:
            info_red('Memory usage = ' + self.getMyMemoryUsage())

    def functional(self):

         u = self.pdesystems['Navier-Stokes'].u_
         p = self.pdesystems['Navier-Stokes'].p_
         n = FacetNormal(self.mesh)
         b0 = assemble(dot(u,n)*ds(0)) 
         b1 = assemble(dot(u,n)*ds(1)) 
         b2 = assemble(dot(u,n)*ds(2)) 
         b3 = assemble(dot(u,n)*ds(3)) 
         p_max = p.vector().max()
         p_min = p.vector().min()

         print "flux ds0 ", b0 
         print "flux ds1 ", b1 
         print "flux ds2 ", b2 
         print "flux ds3 ", b3 
         print "p_min ", p_min 
         print "p_max ", p_max
         if isinstance(u, ListTensor): 
             u_max = max(ui.vector().norm('linf') for ui in u) 
         else:
             u_max = u.vector().norm('linf')  
         print "u_max ", u_max, " U ", self.U

 	 #FIXME should use selected points
         return p_max - p_min 

if __name__ == '__main__':
    from cbc.cfd.icns import NSFullySegregated, NSSegregated, solver_parameters
    import time
    parameters["linear_algebra_backend"] = "PETSc"
    set_log_active(True)
    problem_parameters['viscosity'] = 0.04
    problem_parameters['T'] = 0.01
    problem_parameters['dt'] = 0.01
    problem_parameters['iter_first_timestep'] = 2
    problem_parameters['test_case'] = 1
    solver_parameters = recursive_update(solver_parameters, 
    dict(degree=dict(u=1,u0=1,u1=1,u2=1),
         pdesubsystem=dict(u=101, p=101, velocity_update=101), 
         linear_solver=dict(u='bicgstab', p='gmres', velocity_update='bicgstab'), 
         precond=dict(u='jacobi', p='amg', velocity_update='jacobi'))
         )
    
    problem = Challenge(problem_parameters)

    solver = NSFullySegregated(problem, solver_parameters)
    for name in solver.system_names:
        solver.pdesubsystems[name].prm['monitor_convergence'] = False
    #solver.pdesubsystems['u0_update'].prm['monitor_convergence'] = True
    #solver.pdesubsystems['u1_update'].prm['monitor_convergence'] = True
    #solver.pdesubsystems['u2_update'].prm['monitor_convergence'] = True
    t0 = time.time()
    problem.solve()
    t1 = time.time() - t0

    # Save solution
    #V = VectorFunctionSpace(problem.mesh, 'CG', 1)
    #u_ = project(solver.u_, V)
    #file1 = File('u.pvd')
    #file1 << u_

    print list_timings()

    dump_result(problem, solver, t1, 0)
