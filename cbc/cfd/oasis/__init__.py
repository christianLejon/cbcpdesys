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

class dolfin_normalize:
    """Normalization function. Use until bug 900765 in dolfin has been fixed.
    """
    def __init__(self, V):
        self.u = Function(V)
        self.vv = self.u.vector()
        
    def __call__(self, v):
        # normalize entire vector
        #dummy = normalize(v) # does not work in parallel
        self.vv[:] = 1./v.size()
        c = v.inner(self.vv)
        self.vv[:] = c
        v.axpy(-1., self.vv)

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
        
