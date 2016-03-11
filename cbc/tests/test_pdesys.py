import pytest
import subprocess

from dolfin import *
from cbc.pdesys import PDESubSystem, PDESystem, Problem, default_problem_parameters, \
    default_solver_parameters, copy

mesh = UnitSquareMesh(10, 10)
    
def test_Problem():
    problem_parameters = copy.deepcopy(default_problem_parameters)
    problem = Problem(mesh, problem_parameters)

def test_PDESystem1():
    solver_parameters  = copy.deepcopy(default_solver_parameters)
    solver_parameters['space']['u'] = VectorFunctionSpace   # default=FunctionSpace
    solver_parameters['degree']['u'] = 2                    # default=1
    NStokes = PDESystem([['u', 'p']], mesh, solver_parameters)

def test_PDESystem3():
    solver_parameters  = copy.deepcopy(default_solver_parameters)
    NStokes = PDESystem([['u', 'p', 'c'], ['d', 'e']], mesh, solver_parameters)
    
def test_demo():
    problem_parameters = copy.deepcopy(default_problem_parameters)
    solver_parameters  = copy.deepcopy(default_solver_parameters)
    f = open('../demo/flow_past_dolfin.py', 'r').read()
    try:
        mesh = Mesh('dolfin_fine.xml.gz')
    except:
        subprocess.call('wget http://fenicsproject.org/pub/data/meshes/dolfin_fine.xml.gz', shell=True)
        
    exec f in locals()

def test_demo2():
    f = open('../demo/drivencavity_demo.py', 'r').read()
    exec f in locals()
        
def test_demo3():
    f = open('../demo/CahnHilliard_demo.py', 'r').read()
    exec f in locals()

def test_demo4():
    f = open('../demo/Stokes_demo.py', 'r').read()
    exec f in locals()
