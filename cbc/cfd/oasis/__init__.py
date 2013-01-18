"""
Optimized And StrIpped Solvers
"""
from dolfin import *
from mpi4py import MPI as nMPI
from os import getpid, path, makedirs, getcwd, listdir
from commands import getoutput
import time
from numpy import ceil

comm = nMPI.COMM_WORLD

def getMyMemoryUsage():
    mypid = getpid()
    mymemory = getoutput("ps -o rss %s" % mypid).split()[1]
    return mymemory

class extended_normalize:
    """Normalize part or whole of vector.

    V    = Functionspace we normalize in

    u    = Function where part is normalized

    part = The index of the part of the mixed function space
           that we want to normalize.
        
    For example. When solving for velocity and pressure coupled in the
    Navier-Stokes equations we sometimes (when there is only Neuman BCs 
    on pressure) need to normalize the pressure.

    Example of use:
    mesh = UnitSquare(1, 1)
    V = VectorFunctionSpace(mesh, 'CG', 2)
    Q = FunctionSpace(mesh, 'CG', 1)
    VQ = V * Q
    up = Function(VQ)
    normalize_func = extended_normalize(VQ, 2)
    up.vector()[:] = 2.
    print 'before ', up.vector().array().astype('I')
    normalize_func(up.vector())
    print 'after ', up.vector().array().astype('I')

    results in: 
        before [2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2]   
        after  [2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 0 0 0 0]
    """
    def __init__(self, V, part='entire vector'):
        self.part = part
        if isinstance(part, int):
            self.u = Function(V)
            v = TestFunction(V)
            self.c = assemble(Constant(1., cell=V.cell())*dx, mesh=V.mesh())        
            self.pp = ['0']*self.u.value_size()
            self.pp[part] = '1'
            self.u0 = interpolate(Expression(self.pp, element=V.ufl_element()), V)
            self.x0 = self.u0.vector()
            self.C1 = assemble(v[self.part]*dx)
        else:
            self.u = Function(V)
            self.vv = self.u.vector()
        
    def __call__(self, v):
        if isinstance(self.part, int):
            # assemble into c1 the part of the vector that we want to normalize
            c1 = self.C1.inner(v)
            if abs(c1) > 1.e-8:
                # Perform normalization
                self.x0[:] = self.x0[:]*(c1/self.c)
                v.axpy(-1., self.x0)
                self.x0[:] = self.x0[:]*(self.c/c1)
        else:
            # normalize entire vector
            dummy = normalize(v) # does not work in parallel
            
# The following helper functions are available in dolfin
# They are redefined here for printing only on process 0. 
RED   = "\033[1;37;31m%s\033[0m"
BLUE  = "\033[1;37;34m%s\033[0m"
GREEN = "\033[1;37;32m%s\033[0m"

def info_blue(s):
    if MPI.process_number()==0:
        print BLUE % s

def info_green(s):
    if MPI.process_number()==0:
        print GREEN % s
    
def info_red(s):
    if MPI.process_number()==0:
        print RED % s
        
