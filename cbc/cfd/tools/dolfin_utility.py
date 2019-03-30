__author__ = "Christian Lejon <christian.lejon@foi.se>"
__date__ = "20190205"
__copyright__ = "Copyright (C) 2019-2025 " + __author__
__license__  = "GNU Lesser GPL version 3 or any later version"


from dolfin import *
import pdb



class MeshFunction_CL(MeshFunction):
    pdb.set_trace()
#    def __init__(self, tp, m, dim):
    def __init__(self, tp, m, dim, boundary_indicators = 1):
        self.boundary_indicators = boundary_indicators
        pdb.set_trace()
        super.__init__(self, tp, m, dim)

