__author__ = "Mikael Mortensen <mikaem@math.uio.no>"
__date__ = "2011-12-19"
__copyright__ = "Copyright (C) 2011 " + __author__
__license__  = "GNU GPL version 3 or any later version"
"""
This module contains functionality for probing a problem. 
"""
from dolfin import Point, Cell
import ufc
from numpy import zeros, array, squeeze, load

class Probe:
    """Compute one single probe efficiently when it needs to be called repeatedly."""
    def __init__(self, x, V, max_probes=None):
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
        self.probes = max_probes
        if not self.probes is None:             
            self.probes = zeros((self.num_tensor_entries, max_probes))
        
    def __call__(self, u, n=0):
        """Probe the Function u and either return or store result in self.probes."""
        
        u.restrict(self.coefficients, self.element, self.cell, self.ufc_cell)
        
        self.val[:] = 0.
        for i in range(self.element.space_dimension()):
            self.element.evaluate_basis(i, self.basis, self.x, self.cell)
            self.val[:] += self.coefficients[i]*self.basis[:]
        
        if not self.probes is None:
            self.probes[:, n] = self.val[:] 
        else:
            return self.val
            
    def dump(self, filename):
        """Dump probes to filename."""
        squeeze(self.probes).dump(filename)
        
    def load(self, filename):
        """Load the probe previously stored in filename"""
        return load(filename)
        
class Probedict(dict):
    """Dictionary of probes. The key is the variable we're probing 
    and the value is a list of probes returned from the function Probes.
    """
    def probe(self, q_, n):
        for ui in self.keys():
            for i, probe in self[ui]:
                probe(q_[ui], n)
                
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
            
# Test the functions:
if __name__=='__main__':
    from dolfin import *
    mesh = UnitSquare(10, 10)
    V = FunctionSpace(mesh, 'CG', 1)
    Vv = VectorFunctionSpace(mesh, 'CG', 1)
    u0 = interpolate(Expression('x[0]'), V)
    u1 = interpolate(Expression('x[0]*x[0]'), V)
    v = interpolate(Expression(('x[0]', 'x[1]')), Vv)
    x = [array([0.56, 0.27]),
         array([0.26, 0.27]),
         array([0.99, 0.01])]
         
    probesV = Probes(x, V)
    probesVv = Probes(x, Vv)

    for i, probe in probesV:
        print 'Process rank that contains the probe = ', MPI.process_number()
        print 'Probe of u0 at ({0}, {1}) = {2}'.format(probe.x[0], probe.x[1], *probe(u0))
        print 'Probe of u1 at ({0}, {1}) = {2}'.format(probe.x[0], probe.x[1], *probe(u1))
    for i, probe in probesVv:
        print 'Process rank that contains the probe = ', MPI.process_number()
        print 'Probe v at ({0}, {1}) = {2}, {3}'.format(probe.x[0], probe.x[1], *probe(v))

    # Test Probedict
    q_ = {'u0':u0, 'u1':u1}
    VV = {'u0':V, 'u1':V}
    pd = Probedict((ui, Probes(x, VV[ui], 5)) for ui in ['u0', 'u1'])
    pd.probe(q_, 0)
    pd.dump('test')
    
