__author__ = "Mikael Mortensen <mikaem@math.uio.no>"
__date__ = "2011-12-19"
__copyright__ = "Copyright (C) 2011 " + __author__
__license__  = "GNU Lesser GPL version 3 or any later version"
"""
This module contains functionality for efficiently probing a Function many times. 
"""
from dolfin import *
from numpy import zeros, array, repeat, resize, linspace
from numpy.linalg import norm as numpy_norm
from scitools.basics import meshgrid
from scitools.std import surfc
import pyvtk, os, copy
from mpi4py import MPI as nMPI
comm = nMPI.COMM_WORLD

def strip_essential_code(filenames):
    code = ""
    for name in filenames:
        f = open(name, 'r').read()
        code += f[f.find("namespace dolfin\n{\n"):f.find("#endif")]
    return code

dolfin_folder = "Probe"
sources = ["Probe.cpp", "Probes.cpp", "StatisticsProbe.cpp", "StatisticsProbes.cpp"]
headers = map(lambda x: os.path.join(dolfin_folder, x), ['Probe.h', 'Probes.h', 'StatisticsProbe.h', 'StatisticsProbes.h'])
code = strip_essential_code(headers)
compiled_module = compile_extension_module(code=code, source_directory=os.path.abspath(dolfin_folder),
                                           sources=sources, include_dirs=[".", os.path.abspath(dolfin_folder)])
 
# Give compiled classes some additional pythonic functionality
class Probe(compiled_module.Probe):
    
    def __call__(self, *args):
        return self.eval(*args)

    def __len__(self):
        return self.value_size()

class Probes(compiled_module.Probes):

    def __call__(self, *args):
        return self.eval(*args)
        
    def __len__(self):
        return self.local_size()

    def __iter__(self): 
        self.i = 0
        return self

    def __getitem__(self, i):
        return self.get_probe_id(i), self.get_probe(i)

    def next(self):
        try:
            p =  self[self.i]
        except:
            raise StopIteration
        self.i += 1
        return p    

    def array(self, N=None, filename=None):
        """Dump data to numpy format for all or one snapshot"""
        if N:
            z  = zeros((self.get_total_number_probes(), self.value_size()))
            z0 = zeros((self.get_total_number_probes(), self.value_size()))
        else:
            z  = zeros((self.get_total_number_probes(), self.value_size(), self.number_of_evaluations()))
            z0 = zeros((self.get_total_number_probes(), self.value_size(), self.number_of_evaluations()))
        if len(self) > 0:
            for index, probe in self:
                if N:
                    z[index, :] = probe.get_probes_at_snapshot(N)
                else:
                    for k in range(self.value_size()):
                        z[index, k, :] = probe.get_probe_sub(k)    
        comm.Reduce(z, z0, op=nMPI.MAX, root=0)
        if comm.Get_rank() == 0:
            if filename:
                if N:
                    z0.dump(filename+"_snapshot_"+str(N)+".probes")
                else:
                    z0.dump(filename+"_all.probes")
            return z0

class StatisticsProbe(compiled_module.StatisticsProbe):
    
    def __call__(self, *args):
        return self.eval(*args)

    def __len__(self):
        return self.value_size()

class StatisticsProbes(compiled_module.StatisticsProbes):

    def __call__(self, *args):
        return self.eval(*args)
        
    def __len__(self):
        return self.local_size()

    def __iter__(self): 
        self.i = 0
        return self

    def __getitem__(self, i):
        return self.get_probe_id(i), self.get_probe(i)

    def next(self):
        try:
            p = self[self.i]
        except:
            raise StopIteration
        self.i += 1
        return p   
        
    def array(self, filename=None):
        """Dump data to numpy format for all or one snapshot"""
        z  = zeros((self.get_total_number_probes(), self.value_size()))
        z0 = zeros((self.get_total_number_probes(), self.value_size()))
        if len(self) > 0:
            for index, probe in self:
                umean = probe.mean()
                z[index, :len(umean)] = umean[:]
                z[index, len(umean):] = probe.variance()
        comm.Reduce(z, z0, op=nMPI.MAX, root=0)
        if comm.Get_rank() == 0:
            if filename:
                z0.dump(filename+"_statistics.probes")
            return z0

class StructuredGrid:
    """A Structured grid of probe points. A slice of a 3D (possibly 2D if needed) 
    domain can be created with any orientation using two tangent vectors to span 
    a local coordinatesystem. Likewise, a Box in 3D is created by supplying three
    basis vectors.
    
          dims = [N1, N2 (, N3)] number of points in each direction
          origin = (x, y, z) origin of slice 
          dX = [[dx1, dy1, dz1], [dx2, dy2, dz2] (, [dx3, dy3, dz3])] tangent vectors (need not be orthogonal)
          dL = extent of each direction
          V  = FunctionSpace to probe in
          statistics = True  => Compute statistics in probe points (mean/covariance).
                       False => Store instantaneous snapshots of probepoints. 
    """
    def __init__(self, dims, origin, dX, dL, V, statistics=False):
        self.dims = dims
        self.dL, self.dX = array(dL, float), array(dX, float)
        self.origin = origin
        self.x = self.create_grid()
        if statistics:
            self.probes = StatisticsProbes(self.x.flatten(), V, V.num_sub_spaces()==0)
        else:
            self.probes = Probes(self.x.flatten(), V)
        
    def __call__(self, *args):
        self.probes.eval(*args)
        
    def __getitem__(self, i):
        return self.probes[i]

    def create_grid(self):
        """Create 2D slice or 3D box"""
        origin = self.origin
        dX, dL = self.dX, self.dL
        dims = array(self.dims)
        for i, N in enumerate(dims):
            dX[i, :] = dX[i, :] / numpy_norm(dX[i, :]) * dL[i] / N
        
        xs = [linspace(origin[0], origin[0]+dX[0][0]*dims[0], dims[0])]
        ys = [linspace(origin[1], origin[1]+dX[0][1]*dims[0], dims[0])]
        zs = [linspace(origin[2], origin[2]+dX[0][2]*dims[0], dims[0])]
        for k in range(1, len(dims)):
            xs.append(linspace(0, dX[k][0]*dims[k], dims[k]))
            ys.append(linspace(0, dX[k][1]*dims[k], dims[k]))
            zs.append(linspace(0, dX[k][2]*dims[k], dims[k]))
        
        dim = dims.prod()
        x = zeros((dim, 3))
        if len(dims) == 3:
            x[:, 0] = repeat(xs[2], dims[0]*dims[1])[:] + resize(repeat(xs[1], dims[0]), dim)[:] + resize(xs[0], dim)[:]
            x[:, 1] = repeat(ys[2], dims[0]*dims[1])[:] + resize(repeat(ys[1], dims[0]), dim)[:] + resize(ys[0], dim)[:]
            x[:, 2] = repeat(zs[2], dims[0]*dims[1])[:] + resize(repeat(zs[1], dims[0]), dim)[:] + resize(zs[0], dim)[:]
        else:
            x[:, 0] = repeat(xs[1], dims[0])[:] + resize(xs[0], dim)[:]
            x[:, 1] = repeat(ys[1], dims[0])[:] + resize(ys[0], dim)[:]
            x[:, 2] = repeat(zs[1], dims[0])[:] + resize(zs[0], dim)[:]

        return x

    def surf(self, N, component=0):
        """surf plot of scalar or one component of tensor"""
        if comm.Get_size() > 1:
            print "No surf for multiple processors"
            return
        if len(self.dims) == 3:
            print "No surf for 3D cube"
            return
        z = zeros((self.dims[0], self.dims[1]))
        for j in range(self.dims[1]):
            for i in range(self.dims[0]):
                val = self.probes[j*self.dims[0] + i][1].get_probe_sub(component)
                z[i, j] = val[N]
        # Use local coordinates
        yy, xx = meshgrid(linspace(0, dL[1], self.dims[1]),
                          linspace(0, dL[0], self.dims[0]))        
        surfc(yy, xx, z, indexing='xy')

    def tovtk(self, N, filename="dump.vtk"):
        """Dump data to VTK file"""
        z  = zeros((self.probes.get_total_number_probes(), self.probes.value_size()))
        z0 = zeros((self.probes.get_total_number_probes(), self.probes.value_size()))
        if len(self.probes) > 0:
            for index, probe in self.probes:
                z[index, :] = probe.get_probes_at_snapshot(N)
        # Put solution on process 0
        comm.Reduce(z, z0, op=nMPI.MAX, root=0)
        
        # Store in vtk-format
        if comm.Get_rank() == 0:
            d = self.dims
            d = (d[0], d[1], d[2]) if len(d) > 2 else (d[0], d[1], 1)
            grid = pyvtk.StructuredGrid(d, self.x)
            v = pyvtk.VtkData(grid)
            if self.probes.value_size() == 1:
                v.point_data.append(pyvtk.Scalars(z0))
            elif self.probes.value_size() == 3:
                v.point_data.append(pyvtk.Vectors(z0))
            elif self.probes.value_size() == 9: # StatisticsProbes
                num_evals = probe.number_of_evaluations()
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
        for ui, p in self.iteritems():
            p(q_[ui])
                
    def dump(self, filename):
        for ui, p in self.iteritems():
            p.dump(filename+"_"+ui)

# Test the probe functions:
if __name__=='__main__':
    import time
    mesh = UnitCubeMesh(10, 10, 10)
    #mesh = UnitSquareMesh(10, 10)
    V = FunctionSpace(mesh, 'CG', 2)
    Vv = VectorFunctionSpace(mesh, 'CG', 1)
    W = V * Vv
    
    # Just create some random data to be used for probing
    x0 = interpolate(Expression('x[0]'), V)
    y0 = interpolate(Expression('x[1]'), V)
    z0 = interpolate(Expression('x[2]'), V)
    s0 = interpolate(Expression('exp(-(pow(x[0]-0.5, 2)+ pow(x[1]-0.5, 2) + pow(x[2]-0.5, 2)))'), V)
    v0 = interpolate(Expression(('x[0]', '2*x[1]', '3*x[2]')), Vv)
    w0 = interpolate(Expression(('x[0]', 'x[1]', 'x[2]', 'x[1]*x[2]')), W)    
        
    x = array([[1.5, 0.5, 0.5], [0.2, 0.3, 0.4], [0.8, 0.9, 1.0]])
    p = Probes(x.flatten(), W)
    for i in range(6):
        p(w0)
        
    print p.array(2, "testarray")         # dump snapshot 2
    print p.array(filename="testarray")   # dump all snapshots
    print p.dump("testarray")

    # Test StructuredGrid
    # 3D box
    #origin = [0.1, 0.1, 0.1]               # origin of box
    #tangents = [[1, 0, 0], [0, 1, 0], [0, 0, 1]]    # directional tangent directions (scaled in StructuredGrid)
    #dL = [0.2, 0.4, 0.6]                      # extent of slice in both directions
    #N  = [5, 20, 40]                           # number of points in each direction
    
    # 2D slice
    origin = [0., 0., 0.5]               # origin of slice
    tangents = [[1, 0, 0], [0, 1, 0]]    # directional tangent directions (scaled in StructuredGrid)
    dL = [0.5, 0.9]                      # extent of slice in both directions
    N  = [8, 20]                           # number of points in each direction
   
    # Test scalar first
    sl = StructuredGrid(N, origin, tangents, dL, V)
    sl(s0)     # probe once
    sl(s0)     # probe once more
    sl.surf(0) # check first probe
    sl.tovtk(0, filename="dump_scalar.vtk")
    
    # then vector
    sl2 = StructuredGrid(N, origin, tangents, dL, Vv)
    for i in range(5): 
        sl2(v0)     # probe a few times
    sl2.surf(3)     # Check the fourth probe instance
    sl2.tovtk(3, filename="dump_vector.vtk")

    # Test statistics
    sl3 = StructuredGrid(N, origin, tangents, dL, V, statistics=True)
    for i in range(10): 
        sl3(x0, y0, z0)     # probe a few times
        #sl3(v0)
    sl3.surf(0)     # Check 
    sl3.tovtk(0, filename="dump_mean_vector.vtk")
            
    ## Test Probedict
    q_ = {'u0':v0, 'u1':x0}
    VV = {'u0':Vv, 'u1':V}
    pd = Probedict((ui, Probes(x.flatten(), VV[ui])) for ui in ['u0', 'u1'])
    for i in range(7):
        pd(q_)
    pd.dump('testdict')
