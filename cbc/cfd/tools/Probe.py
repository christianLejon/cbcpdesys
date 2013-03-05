__author__ = "Mikael Mortensen <mikaem@math.uio.no>"
__date__ = "2011-12-19"
__copyright__ = "Copyright (C) 2011 " + __author__
__license__  = "GNU Lesser GPL version 3 or any later version"
"""
This module contains functionality for efficiently probing a Function many times. 
"""
#from dolfin import Point, Cell
from dolfin import *
import ufc
from numpy import zeros, array, squeeze, load, linspace, dot as ndot

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
            
    def coordinates(self):
        return self.x
        
    def get_probe(self, i):
        return self.probes[i]
        
    def number_of_evaluations(self):
        return self.counter
        
    def dump(self, filename):
        """Dump probes to filename."""
        squeeze(self.probes).dump(filename)
        
    def load(self, filename):
        """Load the probe previously stored in filename"""
        return load(filename)
        
class CppProbe(cpp.Probe):
    
    def __call__(self, u):
        return self.eval(u)
            
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
    
class CppProbes(list):
    """List of probes. Each processor is appended with a new 
    CppProbe class only when the probe is found on that processor.
    """
    def __init__(self, list_of_probes, V):
        for i, p in enumerate(list_of_probes):
            try:
                self.append((i, CppProbe(array(p), V)))
            except RuntimeError:
                pass
        print len(self), "of ", len(list_of_probes), " probes found on processor ", MPI.process_number()
    
    def __call__(self, u):
        """eval for all probes"""
        u.update()
        for i, p in self:
            p(u)

class Probes(list):
    """List of probes. Each processor is appended with a new 
    Probe class only when the probe is found on that processor.
    """
    def __init__(self, list_of_probes, V, max_probes=None):
        for i, p in enumerate(list_of_probes):
            try:
                self.append((i, Probe(array(p), V, max_probes=max_probes)))
            except RuntimeError:
                pass    
    
    def __call__(self, u):
        """eval for all probes"""
        u.update()
        for i, p in self:
            p(u)

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
    import h5py
    mesh = UnitCubeMesh(10, 10, 10)
    V = FunctionSpace(mesh, 'CG', 1)
    Vv = VectorFunctionSpace(mesh, 'CG', 1)
    W = V * Vv
    u0 = interpolate(Expression('x[0]'), V)
    u1 = interpolate(Expression('x[0]*x[0]'), V)
    v0 = interpolate(Expression(('x[0]', 'x[1]', 'x[2]')), Vv)
    w0 = interpolate(Expression(('x[0]', 'x[1]', 'x[2]', 'x[1]*x[2]')), W)
    #x = [array([0.56, 0.27, 0.5]),
         #array([0.26, 0.27, 0.5]),
         #array([0.99, 0.01, 0.5])]

    # Create a range of probes for a UnitSquare
    N = 5
    xx = linspace(0, 1, N)
    xx = xx.repeat(5).reshape((N, N)).transpose()
    yy = linspace(0, 1, N)
    yy = yy.repeat(N).reshape((N, N))
    x = zeros((N*N, 3))    
    for i in range(N):
        for j in range(N):
            x[i*N + j, 0 ] = xx[i, j]   #  x-value
            x[i*N + j, 1 ] = yy[i, j]   #  y-value
         
    probesV = Probes(x, V, 1000)
    
    t0 = time.time()
    for j in range(1000):
        probesV(u0)
    print 'Time to Python eval ', j+1 ,' probes = ', time.time() - t0

    cpprobesV = CppProbes(x, V)
    t0 = time.time()
    for j in range(1000):
        cpprobesV(u0)
    print 'Time to Cpp eval ', j+1 ,' probes = ', time.time() - t0

    #probesV = Probes(x, V, 2)
    #probesVv = Probes(x, Vv, 1)
    #probesW = Probes(x, W, 1)
    
    #for i, probe in probesV:
        #print 'Process rank that contains the probe = ', MPI.process_number()
        #print 'Probe of u0 at ({0}, {1}, {2}) = {3}'.format(probe.x[0], probe.x[1], probe.x[2], *probe(u0))
        #print 'Probe of u1 at ({0}, {1}, {2}) = {3}'.format(probe.x[0], probe.x[1], probe.x[2], *probe(u1))
    #for i, probe in probesVv:
        #print 'Process rank that contains the probe = ', MPI.process_number()
        #print 'Probe v at ({0}, {1}, {2}) = {3}, {4}, {5}'.format(probe.x[0], probe.x[1], probe.x[2], *probe(v0))
    #for i, probe in probesW:
        #print 'Process rank that contains the probe = ', MPI.process_number()
        #print 'Probe v at ({0}, {1}, {2}) = {3}, {4}, {5}, {6}'.format(probe.x[0], probe.x[1], probe.x[2], *probe(w0))
        
    ## Test Probedict
    q_ = {'u0':u0, 'u1':u1}
    VV = {'u0':V, 'u1':V}
    #pd = Probedict((ui, Probes(x, VV[ui])) for ui in ['u0', 'u1'])
    pd = Probedict((ui, CppProbes(x, VV[ui])) for ui in ['u0', 'u1'])
    pd(q_)
    pd.dump('testV')
    
    dd = zeros((10, 10, 10))
    for i in range(1,10):
        dd[i, :, :] = i*1.
    
    f = h5py.File("mydata.hdf5", "w")
    f.create_dataset("mydata", dtype="Float32", data=dd)
    f.close()

    
    #q_ = {'v0':v0}
    #VV = {'v0':Vv}
    #pd2 = Probedict((ui, CppProbes(x, VV[ui])) for ui in ['v0'])
    ##pd2 = Probedict((ui, Probes(x, VV[ui])) for ui in ['v0'])
    #pd2.probe(q_)
    #pd2.probe(q_)
    #pd2.probe(q_)
    #pd2.probe(q_)
    #pd2.dump('testVv')
    
    #p = CppProbes(x, V)
    
    #print "MPI ", MPI.process_number(), p
    

    
    
    #p = cpp.Probe(x[0], V)
    #p2= cpp.Probe(x[0], Vv)
    #"Probe ok"
    #p.probe(u0); p.probe(u0)
    #p2.probe(v0); p2.probe(v0)
    #"Probe.probe ok"
    #print p.get_probe(0)
    #print p2.get_probe(0)
    #print p2.get_probe(1)
   