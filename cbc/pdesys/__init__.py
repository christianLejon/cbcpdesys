__author__ = "Mikael Mortensen <mikaem@math.uio.no>"
__date__ = "2010-09-08"
__copyright__ = "Copyright (C) 2010-2016 " + __author__
__license__  = "GNU Lesser GPL version 3 or any later version"
from PDESubSystems import *
from PDESystem import *
from Problem import *
problem_parameters = copy.deepcopy(default_problem_parameters)
solver_parameters  = copy.deepcopy(default_solver_parameters)
prm = solver_parameters
