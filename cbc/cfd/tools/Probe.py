__author__ = "Mikael Mortensen <mikaem@math.uio.no>"
__date__ = "2011-12-19"
__copyright__ = "Copyright (C) 2011 " + __author__
__license__  = "GNU Lesser GPL version 3 or any later version"
"""
This module contains functionality for efficiently probing a Function many times. 
"""
from dolfin import Point, Cell
#from dolfin import *
import ufc
from numpy import zeros, array, squeeze, load, dot as ndot

class Probe:
    """Compute one single probe efficiently when it needs to be called repeatedly."""
    def __init__(self, x, V, max_probes=1):
        mesh = V.mesh()
        c = mesh.intersected_cell(Point(*x))
        if c == -1:
            raise RuntimeError('Probe not found on processor')
        
        # If you get here then the processor contains the probe. 
        # Allocate variables that are used repeatedly 
        self.x = x
        self.ufc_cell = ufc.cell()
        self.cell = Cell(mesh, c)
        self.element = V.dolfin_element()
        self.num_tensor_entries = 1
        for i in range(self.element.value_rank()):
            self.num_tensor_entries *= self.element.value_dimension(i)
        self.coefficients = zeros(self.element.space_dimension())
        self.basis = zeros(self.num_tensor_entries)
        self.val = zeros(self.num_tensor_entries)
        self.probes = zeros((self.num_tensor_entries, max_probes))
        self.counter = 0
        self.basis_matrix = zeros((self.num_tensor_entries, self.element.space_dimension()))
        for i in range(self.element.space_dimension()):
            self.element.evaluate_basis(i, self.basis, self.x, self.cell)
            self.basis_matrix[:, i] = self.basis[:]
        
    def __call__(self, u):
        """Probe the Function u and store result in self.probes."""        
        u.restrict(self.coefficients, self.element, self.cell, self.ufc_cell)
        value = ndot(self.basis_matrix, self.coefficients)
        self.probes[:, self.counter] = value
        self.counter += 1
        return value
            
    def dump(self, filename):
        """Dump probes to filename."""
        squeeze(self.probes).dump(filename)
        
    def load(self, filename):
        """Load the probe previously stored in filename"""
        return load(filename)
        
class Probedict(dict):
    """Dictionary of probes. The keys are the names of the functions 
    we're probing and the values are lists of probes returned from 
    the function Probes.
    """
    def probe(self, q_):
        for ui in self.keys():
            for i, probe in self[ui]:
                probe(q_[ui])
                
    def dump(self, filename):
        for ui in self.keys():
            for i, probe in self[ui]:
                probe.dump(filename + '_' + ui + '_' + str(i) + '.probe')

def Probes(list_of_probes, V, max_probes=None):
    """Return a list of probes. Each processor is appended with a new 
    Probe class only when the probe is found on that processor.
    """
    probes = []
    for i, probe in enumerate(list_of_probes):
        try:
            probes.append((i, Probe(probe, V, max_probes)))
        except RuntimeError:
            pass
    return probes

            
# Test the probe functions:
if __name__=='__main__':
    from dolfin import *
    import time
    mesh = UnitCube(10, 10, 10)
    V = FunctionSpace(mesh, 'CG', 1)
    Vv = VectorFunctionSpace(mesh, 'CG', 1)
    W = V * Vv
    u0 = interpolate(Expression('x[0]'), V)
    u1 = interpolate(Expression('x[0]*x[0]'), V)
    v0 = interpolate(Expression(('x[0]', 'x[1]', 'x[2]')), Vv)
    w0 = interpolate(Expression(('x[0]', 'x[1]', 'x[2]', 'x[1]*x[2]')), W)
    x = [array([0.56, 0.27, 0.5]),
         array([0.26, 0.27, 0.5]),
         array([0.99, 0.01, 0.5])]
         
    probesV = Probes(x, V, 1000)
    
    t0 = time.time()
    for i, probe in probesV:
        for j in range(probesV[0][1].probes.shape[1]):
            probe(u0)
    print 'Time to look up ', j+1 ,' probes = ', time.time() - t0    
    
    probesV = Probes(x, V, 2)
    probesVv = Probes(x, Vv, 1)
    probesW = Probes(x, W, 1)
    
    for i, probe in probesV:
        print 'Process rank that contains the probe = ', MPI.process_number()
        print 'Probe of u0 at ({0}, {1}, {2}) = {3}'.format(probe.x[0], probe.x[1], probe.x[2], *probe(u0))
        print 'Probe of u1 at ({0}, {1}, {2}) = {3}'.format(probe.x[0], probe.x[1], probe.x[2], *probe(u1))
    for i, probe in probesVv:
        print 'Process rank that contains the probe = ', MPI.process_number()
        print 'Probe v at ({0}, {1}, {2}) = {3}, {4}, {5}'.format(probe.x[0], probe.x[1], probe.x[2], *probe(v0))
    for i, probe in probesW:
        print 'Process rank that contains the probe = ', MPI.process_number()
        print 'Probe v at ({0}, {1}, {2}) = {3}, {4}, {5}, {6}'.format(probe.x[0], probe.x[1], probe.x[2], *probe(w0))
        
    ## Test Probedict
    q_ = {'u0':u0, 'u1':u1}
    VV = {'u0':V, 'u1':V}
    pd = Probedict((ui, Probes(x, VV[ui], 1)) for ui in ['u0', 'u1'])
    pd.probe(q_)
    pd.dump('test')
    

   