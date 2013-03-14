__author__ = "Mikael Mortensen <mikaem@math.uio.no>"
__date__ = "2011-12-19"
__copyright__ = "Copyright (C) 2011 " + __author__
__license__  = "GNU Lesser GPL version 3 or any later version"
"""
This module contains functionality for efficiently probing a Function many times. 
"""
from dolfin import *
import ufc
from numpy import zeros, array, repeat, resize, squeeze, load, linspace, dot as ndot
from numpy.linalg import norm as numpy_norm
from scitools.basics import meshgrid
from scitools.std import surfc
import pyvtk
from mpi4py import MPI as nMPI
comm = nMPI.COMM_WORLD

class Probe:
    """Compute one single probe efficiently when it needs to be called repeatedly.
    Pure Python implementation (less efficient than the C++ version)
    """
    def __init__(self, x, V, max_probes=None):
        mesh = V.mesh()
        c = mesh.intersected_cell(Point(*x))
        if c == -1:
            raise RuntimeError('Probe not found on processor')
        
        # If you get here then the processor contains the probe. 
        # Allocate variables that are used repeatedly 
        self.ufc_cell = ufc.cell()
        self.x = x
        self.cell = Cell(mesh, c)
        self.element = V.dolfin_element()
        self.num_tensor_entries = 1
        for i in range(self.element.value_rank()):
            self.num_tensor_entries *= self.element.value_dimension(i)
        self.coefficients = zeros(self.element.space_dimension())
        basis = zeros(self.num_tensor_entries)
        if max_probes == None:
            self.probes = [[]]*self.num_tensor_entries
        else:
            # Preallocated number of probes
            self.probes = zeros((self.num_tensor_entries, max_probes))
        self.counter = 0
        self.basis_matrix = zeros((self.num_tensor_entries, self.element.space_dimension()))
        for i in range(self.element.space_dimension()):
            self.element.evaluate_basis(i, basis, x, self.cell)
            self.basis_matrix[:, i] = basis[:]
        
    def __call__(self, u):
        """Probe the Function u and store result in self.probes.""" 
        u.restrict(self.coefficients, self.element, self.cell, self.ufc_cell)
        value = ndot(self.basis_matrix, self.coefficients)
        if type(self.probes) == list:
            for p, val in zip(self.probes, value):
                p.append(val)
        else:
            self.probes[:, self.counter] = value
        self.counter += 1
        return value
        
    def value_size(self):
        return self.num_tensor_entries
            
    def coordinates(self):
        return self.x
        
    def get_probe(self, i):
        return self.probes[i]
        
    def get_probes(self, i):
        return array(self.probes)[:, i]
        
    def number_of_evaluations(self):
        return self.counter
        
    def dump(self, filename):
        """Dump probes to filename."""
        squeeze(self.probes).dump(filename)
        
    def load(self, filename):
        """Load the probe previously stored in filename"""
        return load(filename)
        
try:
    class CppProbe(cpp.Probe):
    
        def __call__(self, *args):
            return self.eval(args[0])
                
        def dump(self, filename):
            """Dump probes to filename."""
            p = zeros((self.value_size(), self.number_of_evaluations()))
            for i in range(self.value_size()):
                p[i, :] = self.get_probe(i)
            #squeeze(p).dump(filename)
            p.dump(filename)
            
        def load(self, filename):
            """Load the probe previously stored in filename"""
            return load(filename)
            
    class TurbulenceProbe(cpp.TurbulenceProbe):
    
        def __call__(self, *args):
            self.eval(*args)

        def dump(self, filename):
            """Dump probes to filename."""
            umean = self.umean()
            rs = self.reynoldsstress()
            umean.dump("umean_"+filename)
            rs.dump("rs_"+filename)
            
        def load(self, filename):
            """Load the probe previously stored in filename"""
            umean = load("umean_"+filename)
            rs = load("rs_"+filename)
            return umean, rs

except:
    print "No C++ Probe"

class Probes(list):
    """List of probes. Each processor is appended with a new 
    Probe class only when the probe is found on that processor.
    """
    def __init__(self, list_of_probes, V, max_probes=None, turbulence=False, use_python=False):
        for i, p in enumerate(list_of_probes):
            try:
                if hasattr(cpp, "Probe") and not use_python: # Use fast C++ version if available
                    if turbulence:
                        self.append((i, TurbulenceProbe(array(p), V)))
                    else:                        
                        self.append((i, CppProbe(array(p), V)))
                else:
                    self.append((i, Probe(array(p), V, max_probes=max_probes)))
            except RuntimeError:
                pass    
        print len(self), "of ", len(list_of_probes), " probes found on processor ", MPI.process_number()
        self.total_number_of_probes = len(list_of_probes)
        self.value_size = self[0][1].value_size()

    def __call__(self, *args):
        """eval for all probes"""
        for a in args:
            a.update()
        for i, p in self:
            p(*args)

    def dump(self, filename):
        for i, p in self:
            p.dump(filename + '_' + str(i) + '.probe')
            
    def tonumpy(self, N, filename=None):
        """Dump data in to numpy format"""
        z = zeros((self.total_number_of_probes, self.value_size))
        if len(self) > 0:
            for index, probe in self:
                z[index, :] = probe.get_probes(N)
        z0 = zeros((self.total_number_of_probes, self.value_size))
        num_evals = probe.number_of_evaluations()
        comm.Reduce(z, z0, op=nMPI.MAX, root=0)
        if comm.Get_rank() == 0:
            if filename:
                z0.dump(filename)
            return z0
            
class StructuredGrid(Probes):
    """A Structured grid of probe points. A slice of a 3D (possibly 2D if needed) 
    domain can be created with any orientation using two tangent vectors to span 
    a local coordinatesystem.
    
          dims = [N1, N2] number of points in each direction
          origin = (x, y, z) origin of slice 
          dX = [[dx1, dy1, dz1], [dx2, dy2, dz2]] tangent vectors (need not be orthogonal)
          dL = extent of each direction
          V  = FunctionSpace to probe in
    """
    def __init__(self, dims, origin, dX, dL, V, max_probes=None, turbulence=False, use_python=False):
        self.N1, self.N2 = dims
        num_sub = V.num_sub_spaces()
        self.dL = array(dL)
        self.dX = array(dX)
        self.origin = origin
        self.x = self.create_grid()
        Probes.__init__(self, self.x, V, max_probes, turbulence, use_python)
        
    def create_grid(self):
        o1, o2, o3 = self.origin
        dX0 = self.dX[0] / numpy_norm(self.dX[0]) * self.dL[0] / self.N1
        dX1 = self.dX[1] / numpy_norm(self.dX[1]) * self.dL[1] / self.N2
        x1 = linspace(o1, o1+dX0[0]*self.N1, self.N1)
        y1 = linspace(o2, o2+dX0[1]*self.N1, self.N1)
        z1 = linspace(o3, o3+dX0[2]*self.N1, self.N1)
        x2 = linspace(0, dX1[0]*self.N2, self.N2)
        y2 = linspace(0, dX1[1]*self.N2, self.N2)
        z2 = linspace(0, dX1[2]*self.N2, self.N2)
        x = zeros((self.N1*self.N2, 3))
        x[:, 0] = repeat(x1, self.N2)[:] + resize(x2, self.N1*self.N2)[:]
        x[:, 1] = repeat(y1, self.N2)[:] + resize(y2, self.N1*self.N2)[:]
        x[:, 2] = repeat(z1, self.N2)[:] + resize(z2, self.N1*self.N2)[:]
        return x

    def surf(self, N, value_size_num=0):
        """surf plot of scalar or one component (value_size_num) of vector"""
        if comm.Get_size() > 1:
            print "No surf for multiple processors"
            return
        z = zeros((self.N1, self.N2))
        for i in range(self.N1):
            for j in range(self.N2):
                val = self[i*self.N2 + j][1].get_probe(value_size_num)
                z[i, j] = val[N]
        # Use local coordinates
        xx, yy = meshgrid(linspace(0, dL[0], self.N2), 
                          linspace(0, dL[1], self.N1))        
        surfc(xx, yy, z, indexing='xy')

    def tovtk(self, N, filename="dump.vtk"):
        """Dump data in slice to VTK format"""
        z = zeros((self.N1*self.N2, self.value_size))
        if len(self) > 0:
            for index, probe in self:
                z[index, :] = probe.get_probes(N)
        z0 = zeros((self.N1*self.N2, self.value_size))
        num_evals = probe.number_of_evaluations()
        comm.Reduce(z, z0, op=nMPI.MAX, root=0)
        if comm.Get_rank() == 0:
            grid = pyvtk.StructuredGrid((self.N2, self.N1, 1), self.x)
            v = pyvtk.VtkData(grid)
            if self.value_size == 1:
                v.point_data.append(pyvtk.Scalars(z0))
            elif self.value_size == 3:
                v.point_data.append(pyvtk.Vectors(z0))
            elif self.value_size == 9: # Turbulence data
                v.point_data.append(pyvtk.Vectors(z0[:, :3]/num_evals, name="UMEAN"))
                rs = ["uu", "vv", "ww", "uv", "uw", "vw"]
                for i in range(3, 9):
                    v.point_data.append(pyvtk.Scalars(z0[:, i]/num_evals, name=rs[i-3])) 
            else:
                raise TypeError("Only vector or scalar data supported for VTK")
            v.tofile(filename)
            
class Probedict(dict):
    """Dictionary of probes. The keys are the names of the functions 
    we're probing and the values are lists of probes returned from 
    the function Probes.
    """
    def __call__(self, q_):
        for ui in self.keys():
            for i, p in self[ui]:
                p(q_[ui])
                
    def dump(self, filename):
        for ui in self.keys():
            for i, p in self[ui]:
                p.dump(filename + '_' + ui + '_' + str(i) + '.probe')

# Test the probe functions:
if __name__=='__main__':
    import time
    mesh = UnitCubeMesh(10, 10, 10)
    #mesh = UnitSquareMesh(10, 10)
    V = FunctionSpace(mesh, 'CG', 1)
    Vv = VectorFunctionSpace(mesh, 'CG', 1)
    W = V * Vv
    
    # Just create some random data to be used for probing
    u0 = interpolate(Expression('x[0]'), V)
    y0 = interpolate(Expression('x[1]'), V)
    z0 = interpolate(Expression('x[2]'), V)
    u1 = interpolate(Expression('x[0]*x[0]'), V)
    v0 = interpolate(Expression(('x[0]', 'x[1]', 'x[2]')), Vv)
    w0 = interpolate(Expression(('x[0]', 'x[1]', 'x[2]', 'x[1]*x[2]')), W)    
    
    # Test StructuredGrid
    origin = [0.4, 0.4, 0.5]             # origin of slice
    tangents = [[1, 0, 1], [0, 1, 1]]    # directional tangent directions (scaled in StructuredGrid)
    dL = [0.2, 0.3]                      # extent of slice in both directions
    N  = [25, 20]                           # number of points in each direction
    
    # Test scalar first
    sl = StructuredGrid(N, origin, tangents, dL, V, use_python=True)
    sl(u0)     # probe once
    sl(u0)     # probe once more
    sl.surf(0) # check first probe
    sl.tovtk(0, filename="dump_scalar.vtk")
    # then vector
    sl2 = StructuredGrid(N, origin, tangents, dL, Vv)
    for i in range(5): 
        sl2(v0)     # probe a few times
    sl2.surf(3)     # Check the fourth probe instance
    sl2.tovtk(3, filename="dump_vector.vtk")

    # Test mean
    sl3 = StructuredGrid(N, origin, tangents, dL, V, turbulence=True)
    for i in range(10): 
        sl3(u0, u0, u0)     # probe a few times
    sl3.surf(0)     # Check 
    sl3.tovtk(0, filename="dump_mean_vector.vtk")

    x = array([[0.5, 0.5, 0.5], [0.2, 0.3, 0.4], [0.8, 0.9, 1.0]])
    p = Probes(x, Vv, use_python=True)
    p(v0)
    p(v0)
    p.dump("testing")
    print p.tonumpy(0)
    
    #### Some more tests.
    ### Create a range of probes for a UnitSquare
    #N = 5
    #xx = linspace(0.25, 0.75, N)
    #xx = xx.repeat(N).reshape((N, N)).transpose()
    #yy = linspace(0.25, 0.75, N)
    #yy = yy.repeat(N).reshape((N, N))
    #x = zeros((N*N, 3))    
    #for i in range(N):
        #for j in range(N):
            #x[i*N + j, 0 ] = xx[i, j]   #  x-value
            #x[i*N + j, 1 ] = yy[i, j]   #  y-value
         
    #probesV = Probes(x, V, 1000, use_python=True)    
    #t0 = time.time()
    #for j in range(1000):
        #probesV(u0)
    #print 'Time to Python eval ', j+1 ,' probes = ', time.time() - t0

    #if hasattr(cpp, "Probe"):
        #cpprobesV = Probes(x, V)
        #t0 = time.time()
        #for j in range(1000):
            #cpprobesV(u0)
        #print 'Time to Cpp eval ', j+1 ,' probes = ', time.time() - t0
        
    ### Test Probedict
    #q_ = {'u0':u0, 'u1':u1}
    #VV = {'u0':V, 'u1':V}
    #pd = Probedict((ui, Probes(x, VV[ui])) for ui in ['u0', 'u1'])
    #pd(q_)
    #pd.dump('testV')
        
