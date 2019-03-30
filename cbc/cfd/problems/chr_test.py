from dolfin import *
from numpy import arctan, array
import pdb
import matplotlib.pyplot as plt

mesh = UnitSquareMesh(1, 1)
V = FunctionSpace(mesh, 'CG', 1)
u = Function(V)
pdb.set_trace()
u.value_size()
