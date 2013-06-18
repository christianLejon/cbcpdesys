__author__ = "Mikael Mortensen <mikaem@math.uio.no>"
__date__ = "2011-12-19"
__copyright__ = "Copyright (C) 2011 " + __author__
__license__  = "GNU Lesser GPL version 3 or any later version"
"""
This module contains functionality for efficiently probing a Function many times. 
"""
from cbc.cfd.oasis import *
#from dolfin import *
from numpy import cos, zeros, array, repeat, squeeze, argmax, cumsum, reshape, resize, linspace, abs, sign, all, float32
from numpy.linalg import norm as numpy_norm
try:
    from scitools.std import surfc
    from scitools.basics import meshgrid
except:
    pass
import pyvtk, os, copy, cPickle, h5py, inspect
from mpi4py import MPI as nMPI
comm = nMPI.COMM_WORLD

# Compile Probe C++ code
def strip_essential_code(filenames):
    code = ""
    for name in filenames:
        f = open(name, 'r').read()
        code += f[f.find("namespace dolfin\n{\n"):f.find("#endif")]
    return code

dolfin_folder = os.path.abspath(os.path.join(inspect.getfile(inspect.currentframe()), "../Probe"))
sources = ["Probe.cpp", "Probes.cpp", "StatisticsProbe.cpp", "StatisticsProbes.cpp"]
headers = map(lambda x: os.path.join(dolfin_folder, x), ['Probe.h', 'Probes.h', 'StatisticsProbe.h', 'StatisticsProbes.h'])
code = strip_essential_code(headers)
compiled_module = compile_extension_module(code=code, source_directory=os.path.abspath(dolfin_folder),
                                           sources=sources, include_dirs=[".", os.path.abspath(dolfin_folder)])
 
# Give the compiled classes some additional pythonic functionality
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

    def array(self, N=None, filename=None, root=0):
        """Dump data to numpy format for all or one snapshot"""
        if N:
            z  = zeros((self.get_total_number_probes(), self.value_size()))
        else:
            z  = zeros((self.get_total_number_probes(), self.value_size(), self.number_of_evaluations()))
        z0 = z.copy()
        if len(self) > 0: 
            for index, probe in self:
                if N:
                    z[index, :] = probe.get_probe_at_snapshot(N)
                else:
                    for k in range(self.value_size()):
                        z[index, k, :] = probe.get_probe_sub(k)
        comm.Reduce(z, z0, op=nMPI.SUM, root=root)
        zsign = sign(z0)
        comm.Reduce(abs(z), z0, op=nMPI.MAX, root=root)
        z0 *= zsign
        if comm.Get_rank() == root:
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
        
    def array(self, N=None, filename=None, root=0):
        """Dump data to numpy format for all or one snapshot"""
        z  = zeros((self.get_total_number_probes(), self.value_size()))
        z0 = z.copy()
        if len(self) > 0:
            for index, probe in self:
                umean = probe.mean()
                z[index, :len(umean)] = umean[:]
                z[index, len(umean):] = probe.variance()
        comm.Reduce(z, z0, op=nMPI.SUM, root=root)
        zsign = sign(z0)
        comm.Reduce(abs(z), z0, op=nMPI.MAX, root=root)
        z0 *= zsign
        if comm.Get_rank() == root:
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
          dL = [dL1, dL2 (, dL3)] extent of each direction
          V  = FunctionSpace to probe in
          
          statistics = True  => Compute statistics in probe points (mean/covariance).
                       False => Store instantaneous snapshots of probepoints. 
          
          restart = hdf5-file => Restart probe from previous computations.
    """
    def __init__(self, V, dims=None, origin=None, dX=None, dL=None, statistics=False, restart=False, tstep=None):
        if restart:
            statistics = self.read_grid_from_hdf5(restart)
        else:
            self.dims = dims
            self.dL, self.dX = array(dL, float), array(dX, float)
            self.origin = origin
            self.dx, self.dy, self.dz = self.create_tangents()
            
        self.initialize_probes(statistics, V)
        if restart:
            self.set_data_from_hdf5(filename=restart, tstep=tstep)
                 
    def initialize_probes(self, statistics, V):
        if len(self.dims) == 2 and statistics:
            raise TypeError("No sampling of turbulence statistics in 2D")

        if len(self.dims) == 2:
            x = self.create_dense_grid()
            self.probes = Probes(x.flatten(), V)
        else:
            # Add plane by plane to avoid excessive memory use
            x = self.create_xy_slice(0)
            self.probes = (StatisticsProbes(x.flatten(), V, V.num_sub_spaces()==0) if statistics else
                           Probes(x.flatten(), V)) 
            for i in range(1, len(self.dz[2])):
                x = self.create_xy_slice(i)
                args = (x.flatten(), V, V.num_sub_spaces()==0) if statistics else (x.flatten(), V)
                self.probes.add_positions(*args)
                 
    def __call__(self, *args):
        self.probes(*args)
        
    def __getitem__(self, i):
        return self.probes[i]
    
    def __iter__(self): 
        self.i = 0
        return self

    def next(self):
        try:
            p = self[self.i]
        except:
            raise StopIteration
        self.i += 1
        return p   
        
    def create_tangents(self):
        """Create the vectors that span the entire computational box.
        dx, dy, dz are lists of length dim (2 for 2D, 3 for 3D)
        Each item of the list is an array spanning the box/slice in
        that direction. For example, a 3D UnitCube of size 5 x 4 x 3
        with tangents corresponding to the xyz-axes can be represented 
        as
        
        dx = [array([ 0., 0.25, 0.5, 0.75, 1.]),
              array([ 0., 0., 0., 0.]),
              array([ 0., 0., 0.])]

        dy = [array([ 0., 0., 0., 0., 0.]),
              array([ 0., 0.33333, 0.66667, 1.]),
              array([ 0., 0., 0.])]
 
        dz = [array([ 0., 0., 0., 0., 0.]),
              array([ 0., 0., 0., 0.]),
              array([ 0., 0.5, 1.])]
        
        We store the entire vector in each direction because each
        vector can be modified and need not be uniform (like here).
        """
        dX, dL = self.dX, self.dL
        dims = array(self.dims)
        for i, N in enumerate(dims):
            dX[i, :] = dX[i, :] / numpy_norm(dX[i, :]) * dL[i] / N
        
        dx, dy, dz = [], [], []
        for k in range(len(dims)):
            dx.append(linspace(0, dX[k][0]*dims[k], dims[k]))
            dy.append(linspace(0, dX[k][1]*dims[k], dims[k]))
            dz.append(linspace(0, dX[k][2]*dims[k], dims[k]))
                
        dx, dy, dz = self.modify_mesh(dx, dy, dz)
        return dx, dy, dz

    def modify_mesh(self, dx, dy, dz):
        """Use this function to for example skew a grid towards a wall.
        If you are looking at a channel flow with -1 < y < 1 and 
        walls located at +-1, then you can skew the mesh using e.g.
        
            dy[1][:] = arctan(pi*(dy[1][:]+origin[1]))/arctan(pi)-origin[1]
            
        Note: origin[1] will be -1 and 0 <= dy <= 2
        """
        return dx, dy, dz
    
    def create_dense_grid(self):
        """Create a dense 2D slice or 3D box."""
        origin = self.origin
        dx, dy, dz = self.dx, self.dy, self.dz
        dims = array(self.dims)
        dim = dims.prod()
        x = zeros((dim, 3))
        if len(dims) == 3:
            x[:, 0] = repeat(dx[2], dims[0]*dims[1])[:] + resize(repeat(dx[1], dims[0]), dim)[:] + resize(dx[0], dim)[:] + origin[0]
            x[:, 1] = repeat(dy[2], dims[0]*dims[1])[:] + resize(repeat(dy[1], dims[0]), dim)[:] + resize(dy[0], dim)[:] + origin[1]
            x[:, 2] = repeat(dz[2], dims[0]*dims[1])[:] + resize(repeat(dz[1], dims[0]), dim)[:] + resize(dz[0], dim)[:] + origin[2]
        else:
            x[:, 0] = repeat(dx[1], dims[0])[:] + resize(dx[0], dim)[:] + origin[0]
            x[:, 1] = repeat(dy[1], dims[0])[:] + resize(dy[0], dim)[:] + origin[1]
            x[:, 2] = repeat(dz[1], dims[0])[:] + resize(dz[0], dim)[:] + origin[2]
        return x

    def create_xy_slice(self, nz):
        origin = self.origin
        dx, dy, dz = self.dx, self.dy, self.dz
        dims = array(self.dims)
        dim = dims[0]*dims[1]
        x = zeros((dim, 3))
        x[:, 0] = resize(repeat(dx[1], dims[0]), dim)[:] + resize(dx[0], dim)[:] + origin[0] + dx[2][nz]
        x[:, 1] = resize(repeat(dy[1], dims[0]), dim)[:] + resize(dy[0], dim)[:] + origin[1] + dy[2][nz]
        x[:, 2] = resize(repeat(dz[1], dims[0]), dim)[:] + resize(dz[0], dim)[:] + origin[2] + dz[2][nz]
        return x
        
    def read_grid_from_hdf5(self, filename="restart.h5"):
        """Read grid data from filename"""
        f = h5py.File(filename, 'r')
        self.origin = f.attrs['origin']
        self.dL = f.attrs['dL']
        self._num_eval = f.attrs['num_evals']
        dx0 = f.attrs['dx0']
        dx1 = f.attrs['dx1']
        dx2 = f.attrs['dx2']
        self.dx = [dx0[0], dx1[0], dx2[0]]
        self.dy = [dx0[1], dx1[1], dx2[1]]
        self.dz = [dx0[2], dx1[2], dx2[2]]
        self.dims = map(len , self.dx)
        statistics = "stats" in f["FEniCS"]
        f.close()
        return statistics
        
    def set_data_from_hdf5(self, filename="restart.h5", tstep=None):
        """Set data in probes using values stored in filename.
        If no specific tstep is provided for regular probe, then choose 
        the tstep with the highest number.
        """
        f = h5py.File(filename, 'r')
        ids = self.probes.get_probe_ids()        
        nn = len(ids)
        
        if "stats" in f['FEniCS'].keys():
            loc = 'FEniCS/stats'
            rs = ["UMEAN", "VMEAN", "WMEAN", "uu", "vv", "ww", "uv", "uw", "vw"]
            data = zeros((nn, 9))
            for i, name in enumerate(rs):
                data[:, i] = f[loc][name].value.flatten()[ids]   
            self.probes.set_probes_from_ids(data.flatten(), self._num_eval)
        else:            
            if tstep == None:
                step = f['FEniCS'].keys()
                step.sort()
                loc = 'FEniCS/' + step[-1]
            else:
                loc = 'FEniCS/tstep'+str(tstep)
            data = zeros((nn, len(f[loc])))
            if len(f[loc]) == 1:
                data[:, 0] = f[loc]['Scalar'].value.flatten()[ids]
            else:
                for ii in range(self.probes.value_size()):
                    data[:, ii] = f[loc]['Comp-{}'.format(ii)].value.flatten()[ids]
                    
            self.probes.set_probes_from_ids(data.flatten())
        f.close()
        
    def get_ijk(self, global_index):
        """return i, j, k indices of structured grid based on global index"""
        d = self.dims
        return (global_index % d[0], 
               (global_index % (d[0]*d[1])) / d[0], 
                global_index / (d[0]*d[1]))
        
    def toh5(self, N, tstep, filename="restart.h5"):
        """Dump probes to HDF5 file. The data can be used for 3D visualization
        using voluviz or to restart the probe. 
        """
        f = h5py.File(filename, 'w')
        d = self.dims
        if not 'origin' in f.attrs:
            f.attrs.create('origin', self.origin)
            f.attrs.create('dx0', array([self.dx[0], self.dy[0], self.dz[0]]))
            f.attrs.create('dx1', array([self.dx[1], self.dy[1], self.dz[1]]))
            if len(self.dx) == 3:
                f.attrs.create('dx2', array([self.dx[2], self.dy[2], self.dz[2]]))
            f.attrs.create('dL', self.dL)
        if not 'num_evals' in f.attrs:
            f.attrs.create('num_evals', self.probes.number_of_evaluations())
        else:
            f.attrs['num_evals'] = self.probes.number_of_evaluations()
        try:
            f.create_group('FEniCS')
        except ValueError:
            pass
        if type(self.probes) == StatisticsProbes and N == 0:
            loc = 'FEniCS/stats'
        else:
            loc = 'FEniCS/tstep'+str(tstep)
        try:
            f.create_group(loc)
        except ValueError:
            pass
        
        # Create datasets if not there already
        d = list(d)
        if len(d) == 2: 
            d.append(1)
        dimT = (d[2], d[1], d[0])
        if self.probes.value_size() == 1:
            try:
                f.create_dataset(loc+"/Scalar", shape=dimT, dtype='f')
            except RuntimeError:
                print 'RuntimeError'
        elif type(self.probes) != StatisticsProbes:
            try:
                for ii in range(self.probes.value_size()):
                    f.create_dataset(loc+"/Comp-{}".format(ii), shape=dimT, dtype='f')
            except:
                pass
            
        elif self.probes.value_size() == 9: # StatisticsProbes
            if N == 0:
                try:
                    num_evals = self.probes.number_of_evaluations()
                    f.create_dataset(loc+"/UMEAN", shape=dimT, dtype='f')
                    f.create_dataset(loc+"/VMEAN", shape=dimT, dtype='f')
                    f.create_dataset(loc+"/WMEAN", shape=dimT, dtype='f')
                    rs = ["uu", "vv", "ww", "uv", "uw", "vw"]
                    for i in range(3, 9):
                        f.create_dataset(loc+"/"+rs[i-3], shape=dimT, dtype='f')
                except RuntimeError:
                    pass                    
            else: # Just dump latest snapshot
                try:                        
                    f.create_dataset(loc+"/U", shape=dimT, dtype='f')
                    f.create_dataset(loc+"/V", shape=dimT, dtype='f')
                    f.create_dataset(loc+"/W", shape=dimT, dtype='f')
                except RuntimeError:
                    pass
        # Last dimension of box is shared amongst processors
        # In case d[2] % Nc is not zero the last planes are distributed
        # between the processors starting with the highest rank and then 
        # gradually lower
        
        MPI.barrier()
        Nc = comm.Get_size()
        myrank = comm.Get_rank()
        Np = self.probes.get_total_number_probes()
        planes_per_proc = d[2] / Nc
        # Distribute remaining planes 
        if Nc-myrank <= (d[2] % Nc):
            planes_per_proc += 1
            
        # Let all processes know how many planes the different processors own
        all_planes_per_proc = comm.allgather(planes_per_proc)
        cum_last_id = cumsum(all_planes_per_proc)
        owned_planes = zeros(Nc+1, 'I')
        owned_planes[1:] = cum_last_id[:]
                            
        # Store owned data in z0
        z0 = zeros((d[0], d[1], planes_per_proc, self.probes.value_size()), dtype=float32)
        zhere = zeros(self.probes.value_size(), dtype=float32)
        zrecv = zeros(self.probes.value_size(), dtype=float32)
        sendto = zeros(Nc, 'I')
        # Run through all probes and send them to the processes 
        # that owns the plane its at
        for i, (global_index, probe) in enumerate(self.probes):
            plane = global_index / (d[0]*d[1])
            owner = argmax(cum_last_id > plane)
            zhere[:] = probe.get_probe_at_snapshot(N)
            if owner != myrank:
                # Send data to owner
                sendto[owner] +=1
                comm.send(global_index, dest=owner, tag=101)
                comm.Send(zhere, dest=owner, tag=102)
            else:
                # myrank owns the current probe and can simply store it
                i, j, k = self.get_ijk(global_index)
                z0[i, j, k-owned_planes[myrank], :] = zhere[:]
        # Let all processors know who they are receiving data from
        recvfrom = zeros((Nc, Nc), 'I')
        comm.Allgather(sendto, recvfrom)
        # Receive the data
        for ii in range(Nc):
            num_recv = recvfrom[ii, myrank]
            for kk in range(num_recv):
                global_index = comm.recv(source=ii, tag=101)
                i, j, k = self.get_ijk(global_index)
                comm.Recv(zrecv, source=ii, tag=102)
                z0[i, j, k-owned_planes[myrank], :] = zrecv[:]
                
        # Voluviz has weird ordering so transpose some axes
        z0 = z0.transpose((2,1,0,3))
        # Write owned data to hdf5 file
        owned = slice(owned_planes[myrank], owned_planes[myrank+1])
        if owned.stop > owned.start:
            if self.probes.value_size() == 1:
                f[loc+"/Scalar"][owned, :, :] = z0[:, :, :, 0]
            elif type(self.probes) != StatisticsProbes:
                for ii in range(self.probes.value_size()):
                    f[loc+"/Comp-{}".format(ii)][owned, :, :] = z0[:, :, :, ii]
            elif self.probes.value_size() == 9: # StatisticsProbes
                if N == 0:
                    num_evals = self.probes.number_of_evaluations()
                    f[loc+"/UMEAN"][owned, :, :] = z0[:, :, :, 0] / num_evals
                    f[loc+"/VMEAN"][owned, :, :] = z0[:, :, :, 1] / num_evals
                    f[loc+"/WMEAN"][owned, :, :] = z0[:, :, :, 2] / num_evals
                    rs = ["uu", "vv", "ww", "uv", "uw", "vw"]
                    for ii in range(3, 9):
                        f[loc+"/"+rs[ii-3]][owned, :, :] = z0[:, :, :, ii] / num_evals
                else: # Just dump latest snapshot
                    f[loc+"/U"][owned, :, :] = z0[:, :, :, 0]
                    f[loc+"/V"][owned, :, :] = z0[:, :, :, 1]
                    f[loc+"/W"][owned, :, :] = z0[:, :, :, 2]
        f.close()

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
        yy, xx = meshgrid(linspace(0, self.dL[1], self.dims[1]),
                          linspace(0, self.dL[0], self.dims[0]))
        surfc(yy, xx, z, indexing='xy')

    def tovtk(self, N, filename):
        """Dump probes to VTK file.
        ## FIXME
        This function is quite memory demanding since dense matrices of
        solution and position are created on each processor.
        """
        z  = zeros((self.probes.get_total_number_probes(), self.probes.value_size()))
        z0 = z.copy()
        if len(self.probes) > 0:
            for index, probe in self.probes:
                z[index, :] = probe.get_probe_at_snapshot(N)
        # Put solution on process 0
        comm.Reduce(z, z0, op=nMPI.SUM, root=0)
        zsign = sign(z0)
        comm.Reduce(abs(z), z0, op=nMPI.MAX, root=0)
        # Store probes in vtk-format
        if comm.Get_rank() == 0:
            z0 *= zsign
            d = self.dims
            d = (d[0], d[1], d[2]) if len(d) > 2 else (d[0], d[1], 1)
            grid = pyvtk.StructuredGrid(d, self.create_dense_grid())
            v = pyvtk.VtkData(grid, "Probe data. Evaluations = {}".format(self.probes.number_of_evaluations()))
            if self.probes.value_size() == 1:
                v.point_data.append(pyvtk.Scalars(z0))
            elif self.probes.value_size() == 3:
                v.point_data.append(pyvtk.Vectors(z0))
            elif self.probes.value_size() == 9: # StatisticsProbes
                if N == 0:
                    num_evals = self.probes.number_of_evaluations()
                    v.point_data.append(pyvtk.Vectors(z0[:, :3]/num_evals, name="UMEAN"))
                    rs = ["uu", "vv", "ww", "uv", "uw", "vw"]
                    for i in range(3, 9):
                        v.point_data.append(pyvtk.Scalars(z0[:, i]/num_evals, name=rs[i-3]))
                else: # Just dump latest snapshot
                    v.point_data.append(pyvtk.Vectors(z0[:, :3], name="U"))
            else:
                raise TypeError("Only vector or scalar data supported for VTK")
            v.tofile(filename)
                
    def average(self, i):
        """Contract results by averaging along axis. Useful for homogeneous
        turbulence geometries like channels or cylinders"""
        z = self.probes.array()
        if comm.Get_rank() == 0:
            z = reshape(z, self.dims + [z.shape[-1]])
            if isinstance(i, int):
                return z.mean(i)
            else:
                if len(i) == 2:
                    assert(i[1] > i[0])
                elif len(i) == 3:
                    assert(i[0] == 0 and i[1] == 1 and i[2] == 2)
                for k, j in enumerate(i):
                    j -= k
                    z = z.mean(j)
                return z
                    
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

# Create a class for a skewed channel mesh 
class ChannelGrid(StructuredGrid):
    def modify_mesh(self, dx, dy, dz):
        """Create grid skewed towards the walls located at y = 1 and y = -1"""
        dy[1][:] = cos(pi*(dy[1][:]+self.origin[1] - 1.) / 2.) - self.origin[1]
        return dx, dy, dz

def project_nonmatching_mesh(u0, V, constrained_domain=None):
    """project from Function u0 to Function in V."""
    V0 = u0.function_space()
    mesh0 = V0.mesh()
    mesh1 = V.mesh()
    gdim = mesh1.geometry().dim()
    x0 = interpolate(Expression('x[0]'), V)
    x1 = interpolate(Expression('x[1]'), V)
    if gdim == 3:
        x2 = interpolate(Expression('x[2]'), V)
    xs = zeros((x0.vector().local_size(), gdim), float)
    xs[:, 0] = x0.vector().array()[:]
    xs[:, 1] = x1.vector().array()[:]
    if gdim == 3:
        xs[:, 2] = x2.vector().array()[:]
        
    print comm.Get_rank(), x0.vector().local_size(), V.dofmap().ownership_range()[1]-V.dofmap().ownership_range()[0]
    
    u = Function(V, constrained_domain=constrained_domain)
    # Now find the values of all these locations using Probes
    # Do this sequentially to save memory
    for i in range(MPI.num_processes()):
        # Put all locations of this mesh on all processes
        xb = comm.bcast(xs, root=i)
        
        # Probe these locations and store result in new Functions
        probes = Probes(xb.flatten(), V0)
        probes(u0)  # evaluate the probes
        data = probes.array(N=0, root=i)
        probes.clear()
        if i == comm.Get_rank():            
            u.vector().set_local(squeeze(data))
    return u
    
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
    x = x*0.9 
    p.add_positions(x.flatten(), W)
    for i in range(6):
        p(w0)
        
    print p.array(2, "testarray")         # dump snapshot 2
    print p.array(filename="testarray")   # dump all snapshots
    print p.dump("testarray")

    # Test StructuredGrid
    # 3D box
    origin = [0.0, 0.0, 0.0]               # origin of box
    tangents = [[1, 0, 0], [0, 1, 0], [0, 0, 1]]    # directional tangent directions (scaled in StructuredGrid)
    dL = [1., 1., 1.]                      # extent of slice in both directions
    N  = [5, 4, 3]                           # number of points in each direction
    
    # 2D slice
    #origin = [0.1, 0.1, 0.5]               # origin of slice
    #tangents = [[1, 0, 0], [0, 1, 0]]    # directional tangent directions (scaled in StructuredGrid)
    #dL = [0.5, 0.8]                      # extent of slice in both directions
    #N  = [5, 5]                           # number of points in each direction
   
    # Test scalar first
    sl = StructuredGrid(V, N, origin, tangents, dL)
    sl(s0)     # probe once
    sl(s0)     # probe once more
    sl.surf(0) # check first probe
    sl.tovtk(0, filename="dump_scalar.vtk")
    sl.toh5(0, 1, 'dump_scalar.h5')
    sl.toh5(0, 2, 'dump_scalar.h5')
    sl.toh5(0, 3, 'dump_scalar.h5')
    
    # then vector
    sl2 = StructuredGrid(Vv, N, origin, tangents, dL)
    for i in range(5): 
        sl2(v0)     # probe a few times
    sl2.surf(3)     # Check the fourth probe instance
    sl2.tovtk(3, filename="dump_vector.vtk")
    sl2.toh5(0, 1, 'dump_vector.h5')
    sl2.toh5(0, 2, 'dump_vector.h5')
    sl2.toh5(0, 3, 'dump_vector.h5')
        
    # Test statistics
    x0.update()
    y0.update()
    z0.update()
    sl3 = StructuredGrid(V, N, origin, tangents, dL, True)
    for i in range(10): 
        sl3(x0, y0, z0)     # probe a few times
        #sl3(v0)
    sl3.surf(0)     # Check 
    sl3.tovtk(0, filename="dump_mean_vector.vtk")
    sl3.tovtk(1, filename="dump_latest_snapshot_vector.vtk")
    sl3.toh5(0, 1, 'reslowmem.h5')    
        
    # Restart probes from sl3
    sl4 = StructuredGrid(V, restart='reslowmem.h5')
    for i in range(10): 
        sl4(x0, y0, z0)     # probe a few more times    
    sl4.tovtk(0, filename="dump_mean_vector_restart_h5.vtk")
    sl4.toh5(0, 0, 'restart_reslowmem.h5')
    
    # Try mixed space
    sl5 = StructuredGrid(W, N, origin, tangents, dL)
    sl5(w0)     # probe once
    sl5(w0)     # probe once more
    sl5.toh5(0, 1, 'dump_mixed.h5')
    sl6 = StructuredGrid(W, restart='dump_mixed.h5')
    sl6.toh5(0, 1, 'dump_mixed_again.h5')
    
    # Test for nonmatching mesh and FunctionSpace
    mesh2 = UnitCubeMesh(16, 16, 16)
    x = mesh2.coordinates()
    x[:, :] = x[:, :] * 0.5 + 0.25
    V2 = FunctionSpace(mesh2, 'CG', 1)
    u = project_nonmatching_mesh(x0, V2)
    plot(u)
    interactive()

    ### Test Probedict
    #q_ = {'u0':v0, 'u1':x0}
    #VV = {'u0':Vv, 'u1':V}
    #pd = Probedict((ui, Probes(x.flatten(), VV[ui])) for ui in ['u0', 'u1'])
    #for i in range(7):
        #pd(q_)
    #pd.dump('testdict')
