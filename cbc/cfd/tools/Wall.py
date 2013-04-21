__author__ = "Mikael Mortensen <mikaem@math.uio.no>"
__date__ = "2010-09-06"
__copyright__ = "Copyright (C) 2010 " + __author__
__license__  = "GNU Lesser GPL version 3 or any later version"

from dolfin import *
from cbc.pdesys import epsilon, sigma
from sets import Set 
from numpy import array, zeros
from numpy import sqrt as nsqrt
from pylab import find

code = \
"""
#include "dolfin.h"

namespace dolfin{

void allow_nonzero_allocation(GenericMatrix* A)
{
  #ifdef HAS_PETSC
  if (A && has_type<PETScMatrix>(*A))
  {
    PETScMatrix& petsc_A = A->down_cast<PETScMatrix>();
    MatSetOption(*petsc_A.mat(), MAT_NEW_NONZERO_ALLOCATION_ERR, PETSC_FALSE);
  }
  #endif
}
}
"""
compiled_module = compile_extension_module(code=code)

class Wallfunction:
    """ 
    Compute the vertices located on and off a wall.
    The wall is provided through the subdomain in bc.
    V is a functionspace
    """    
    def __init__(self, V, bc):
        mesh = self.mesh = V.mesh()
        y = self.mesh_y = mesh.coordinates()
        # Mark a meshfunction's walls with unity if it's not already there
        if not hasattr(bc,'mf'):
            mf = self.mf = FacetFunction("size_t", mesh) # Facets
            mf.set_all(0)
            bc.mark(mf, 1) # Mark all wall facets
            bc.mf = mf
            bc.bid = 1
            
        # Create lists of subdofmaps and dofs for all sub_spaces of V
        dofmaps = self.dofmaps = []
        dofs = self.dofs = []
        for i in range(max(1, V.num_sub_spaces())):
            if V.num_sub_spaces() > 1:
                dofmaps.append(V.sub(i).dofmap())
            else:
                dofmaps.append(V.dofmap())
            dofs.append(set(dofmaps[-1].collapse(mesh)[1].values()))
            off_dofs = dofmaps[-1].off_process_owner()
            dofs[-1].difference_update(off_dofs.keys())
            dofs[-1] = list(dofs[-1])
        
        self.d2c, self.c2d = d2c, c2d = self.map_cells_and_dofs(V)
        
        self.dofs_on_boundary = dob = self.get_dofs_on_boundary(V, bc)
        
        self.dofs_inside_boundary = dib = \
                           self.get_dofs_inside_boundary(dob, d2c, c2d)
        
        self.corners, self.corner_inner_node = \
                           self.handle_corners(V, dob, dib, d2c, c2d)
        
        self.bnd_to_in = self.map_boundary_dof_to_inner_dof(V, 
                           dob, dib, d2c, self.corner_inner_node) 
            
    def map_boundary_dof_to_inner_dof(self, V, vertices_on_boundary, 
                             vertices_inside_boundary, d2c_list, corner_inner_node):
        # Get a map from boundary nodes to closest internal node
        bnds_to_in = []
        for dofmap, vob, vib, d2c, corner in zip(self.dofmaps, vertices_on_boundary, vertices_inside_boundary, d2c_list, corner_inner_node):
            a = zeros(dofmap.cell_dimension(0), dtype="I")
            mesh = self.mesh
            bnd_to_in = {}
            for i in vob:
                dxmin = 1e8
                if i in d2c:
                    for ci in d2c[i]:
                        c = Cell(mesh,ci)
                        a[:] = dofmap.cell_dofs(c.index())
                        x = dofmap.tabulate_coordinates(c)
                        aa = list(a)
                        ii = aa.index(i) # The local index of the boundary node
                        yy = x[ii]       # Coordinates of boundary node
                        for kk, jj in enumerate(aa):
                            if jj in vib:
                                if not kk == ii:
                                    dxmin_ = nsqrt((yy[0] - x[kk,0])**2 + (yy[1] -
                                                                        x[kk,1])**2)
                                    if dxmin_ < dxmin: 
                                        dxmin = dxmin_
                                        bnd_to_in[i] = int(aa[kk])
                if not i in bnd_to_in:
                    if i in corner: 
                        bnd_to_in[i] = corner[i]
            bnds_to_in.append(bnd_to_in)    
        return bnds_to_in
        
    def handle_corners(self, V, dofs_on_boundary, 
                       dofs_inside_boundary, d2c_list, c2d_list):
        # Locate corner nodes that belong to cells without internal nodes
        # This only applies to some corner-elements using CG = 1.
        mesh = self.mesh
        corners = []
        corners_inner_node = []
        for dofmap, dob, dib, d2c, c2d in zip(self.dofmaps, dofs_on_boundary, dofs_inside_boundary, d2c_list, c2d_list):
            a = zeros(dofmap.cell_dimension(0), dtype="I")
            corner = []
            for i in dob:
                if i in d2c:
                    if len(d2c[i]) == 1:
                        ci = d2c[i][0]
                        c = Cell(mesh, ci)
                        a[:] = dofmap.cell_dofs(c.index())
                        if not any([jj in dib for jj in a]):
                            corner.append(i)

            # Find an internal node that we can use for corners
            corner_inner_node={}
            for cor in corner:
                c = d2c[cor][0]
                nodes = c2d[c]
                nodes.remove(cor)
                c2 = d2c[nodes[0]]
                c2.remove(c)
                for c3 in c2:
                    if nodes[0] in c2d[c3] and nodes[1] in c2d[c3]:
                        corner_inner_node[cor] = list(Set(c2d[c3]) - 
                                                        Set(nodes))[0]
                nodes.append(cor)
                c2.append(c)
            corners.append(corner)
            corners_inner_node.append(corner_inner_node)
        return corners, corners_inner_node

    def map_cells_and_dofs(self, V):
        # dof to cell and cell to dof maps for all sub_spaces
        d2c = []
        c2d = []
        for dofmap in self.dofmaps:
            a = zeros(dofmap.cell_dimension(0), dtype="I")
            d2c_ = {}
            c2d_ = {}
            for c in cells(self.mesh): 
                c2d_[c.index()] = []
                a[:] = dofmap.cell_dofs(c.index())
                for v in a: 
                    c2d_[c.index()].append(int(v))
                    if d2c_.has_key(v):
                        d2c_[v].append(c.index())
                    else: 
                        d2c_[v] = [c.index()]
            d2c.append(d2c_)
            c2d.append(c2d_)
        return d2c, c2d
        
    def get_dofs_on_boundary(self, V, bc):        
        d = DirichletBC(V, Constant((1, )*V.num_sub_spaces() if V.num_sub_spaces() else 1), bc.mf, bc.bid)
        u = Function(V)
        #u = Function(FunctionSpace(V.mesh(), V.ufl_element().family(), V.ufl_element().degree()))
        u.vector()[:] = 0
        d.apply(u.vector())
        on_b = Set(array(find(abs(u.vector().array() - 1.) < DOLFIN_EPS),
                         dtype='int32'))
        dob = []
        for dof in self.dofs:
            dob.append(on_b.intersection(dof))
        return dob

    def get_dofs_inside_boundary(self, dob_list, d2c_list, c2d_list):
        dibs = []
        for dob, d2c, c2d in zip(dob_list, d2c_list, c2d_list):
            dib = []
            for v in dob: 
                if v in d2c:
                    for c in d2c[v]:
                        if c in c2d:
                            for v2 in c2d[c]:
                                dib.append(v2)
            dibs.append(Set(dib) - dob)
        return dibs

class Yplus(Wallfunction):
    """
    Compute 
    wall shear stress,
    normal stress,
    utau = sqrt(nu*du/dy)_wall,
    yplus = y*u_tau/nu,  
    
    V is velocity functionspace
    The wall subdomain is provided in bc
    u is velocity
    p is pressure (not used)
    y is the distance to the walls
    nu is viscosity
    """
    def __init__(self, bc, u, p, y, nu, constrained_domain=None):
        V = u.function_space()
        Wallfunction.__init__(self, y.function_space(), bc)
        mesh = V.mesh()
        # Compute stress tensor
        sigma = 2*nu*epsilon(u)

        # Compute surface traction
        n = FacetNormal(mesh)
        #T = - sigma*n
        T = - nu*(grad(u) + grad(u).T)*n

        # Compute normal and tangential components
        Tn = inner(T, n)    # scalar - valued
        Tt = T - Tn*n       # vector - valued
        
        # Define functionspaces
        #vector = V
        vector = VectorFunctionSpace(V.mesh(), V.sub(0).ufl_element().family(), V.ufl_element().degree(), constrained_domain=constrained_domain)
        scalar = FunctionSpace(mesh , "CG", 1, constrained_domain=constrained_domain)  # Need CG=1 for this to work.
        
        # Use the velocity functionspace to compute the distance to the wall. 
        # If y is different from u, then project onto velocity space to get 
        # the same numbering
        #if y.function_space().ufl_element().degree() == V.ufl_element().degree():
            #yy = y
        #else:
            #Q = FunctionSpace(V.mesh(), V.sub(0).ufl_element().family(), V.ufl_element().degree())
            #yy = project(y, Q)
        v = TestFunction(scalar)
        w = TestFunction(vector)
        
        # Declare solution functions
        self.normal_stress = Function(scalar)
        self.shear_stress = Function(vector)
        self.utau = Function(scalar)
        self.yplus = Function(scalar)
        self.uwall = Function(vector)
        
        # Define forms
        Ln = (1 / FacetArea (mesh)) * v * Tn * ds(bc.bid)
        Lt = (1 / FacetArea (mesh)) * inner(w, Tt) * ds(bc.bid)
        Ls = (1 / FacetArea (mesh)) * sqrt(sqrt(inner(Tt, Tt))) * v * ds(bc.bid)
        Ll = (1 / FacetArea (mesh)) * inner(u, w) * ds(bc.bid)
        
        # Assemble results
        assemble(Ln, tensor=self.normal_stress.vector(), exterior_facet_domains=bc.mf)
        assemble(Lt, tensor=self.shear_stress.vector(), exterior_facet_domains=bc.mf)
        assemble(Ls, tensor=self.utau.vector(),  exterior_facet_domains=bc.mf)
        assemble(Ll, tensor=self.uwall.vector(), exterior_facet_domains=bc.mf)
        
        # Compute yplus
        dob = self.dofs_on_boundary[0]
        for i in dob:  # For the boundary nodes belonging to scalar
            j = self.bnd_to_in[0][i]      
            self.yplus.vector()[i] = self.utau.vector()[i] * y.vector()[j] / nu(0)

class KEWall(Wallfunction):
    """Set epsilon = 2*nu*k/y**2 implicitly."""
    def __init__(self, V, bc, y, nu):
        Wallfunction.__init__(self, V, bc)
        self.y = y.vector()
        self.nu = nu
        self.wf = Wallfunction(y.function_space(), bc)
    
    def apply(self, *args):
        """Apply boundary condition to tensors."""
        k_dofs_ob = array(list(self.dofs_on_boundary[0]), 'intc')
        e_dofs_ob = array(list(self.dofs_on_boundary[1]), 'intc')
        k_dofs_ib = array(list(self.dofs_inside_boundary[0]), 'intc')
        e_dofs_ib = array(list(self.dofs_inside_boundary[1]), 'intc')        
        for var in args:
            if isinstance(var, Matrix):
                compiled_module.allow_nonzero_allocation(var)
                var.ident(e_dofs_ob)
                var.ident(e_dofs_ib)
                for kj, ej, yj in zip(k_dofs_ob, e_dofs_ob, self.wf.dofs_on_boundary[0]):
                    yi = self.wf.bnd_to_in[0][yj]
                    ki = self.bnd_to_in[0][kj]
                    ei = self.bnd_to_in[1][ej]
                    col = array([ki], 'uintp')
                    val = array([-2. * self.nu / self.y[yi]**2])
                    var.setrow(ej, col, val)
                    var.setrow(ei, col, val)
                    var.apply('insert')

            if isinstance(var, (Vector, GenericVector)):
                var[e_dofs_ib] = 0.
                var[e_dofs_ob] = 0.

class FWall(Wallfunction):
    """Set f = -20*nu**2*v2/y**4/e implicitly."""
    def __init__(self, V, bc, y, nu, ke):
        Wallfunction.__init__(self, V, bc)
        self.y = y.vector()
        self.nu = nu
        self.wf = Wallfunction(y.function_space(), bc)
        self.ke = ke.vector()
    
    def apply(self, *args):
        """Apply boundary condition to tensors."""
        v2_dofs_ob = array(list(self.dofs_on_boundary[0]), 'intc')
        f_dofs_ob  = array(list(self.dofs_on_boundary[1]), 'intc')
        v2_dofs_ib = array(list(self.dofs_inside_boundary[0]), 'intc')
        f_dofs_ib  = array(list(self.dofs_inside_boundary[1]), 'intc')        
        for var in args:
            if isinstance(var, Matrix):
                compiled_module.allow_nonzero_allocation(var)
                var.ident(f_dofs_ob)
                var.ident(f_dofs_ib)
                for v2j, fj, yj in zip(v2_dofs_ob, f_dofs_ob, self.wf.dofs_on_boundary[0]):
                    yi = self.wf.bnd_to_in[0][yj]
                    v2i = self.bnd_to_in[0][v2j]
                    fi = self.bnd_to_in[1][fj]
                    col = array([v2i], 'uintp')
                    eps = max(1e-6, self.ke[fi])
                    val = array([20. * self.nu**2 / self.y[yi]**4 / eps])
                    var.setrow(fj, col, val)
                    var.setrow(fi, col, val)
                    var.apply('insert')

            if isinstance(var, (Vector, GenericVector)):
                var[f_dofs_ib] = 0.
                var[f_dofs_ob] = 0.

class V2FWall(Wallfunction):
    """Set epsilon = 2*nu*k/y**2 and f = -20*nu**2*v2/y**4/e implicitly."""
    def __init__(self, V, bc, y, nu, v2f):
        Wallfunction.__init__(self, V, bc)
        self.y = y.vector()
        self.nu = nu
        self.wf = Wallfunction(y.function_space(), bc)
        self.v2f = v2f.vector()

    def apply(self, *args):
        """Apply boundary condition to tensors."""
        k_dofs_ob = array(list(self.dofs_on_boundary[0]), 'intc')
        e_dofs_ob = array(list(self.dofs_on_boundary[1]), 'intc')
        k_dofs_ib = array(list(self.dofs_inside_boundary[0]), 'intc')
        e_dofs_ib = array(list(self.dofs_inside_boundary[1]), 'intc')
        v2_dofs_ob = array(list(self.dofs_on_boundary[2]), 'intc')
        f_dofs_ob  = array(list(self.dofs_on_boundary[3]), 'intc')
        v2_dofs_ib = array(list(self.dofs_inside_boundary[2]), 'intc')
        f_dofs_ib  = array(list(self.dofs_inside_boundary[3]), 'intc')        
        nu = self.nu
        for var in args:
            if isinstance(var, Matrix):
                var.ident(e_dofs_ob)
                var.ident(e_dofs_ib)
                var.ident(f_dofs_ob)
                var.ident(f_dofs_ib)
                for kj, ej, v2j, fj, yj in zip(k_dofs_ob, e_dofs_ob, v2_dofs_ob, f_dofs_ob, self.wf.dofs_on_boundary[0]):
                    yi = self.wf.bnd_to_in[0][yj]
                    ki = self.bnd_to_in[0][kj]
                    ei = self.bnd_to_in[1][ej]
                    v2i = self.bnd_to_in[2][v2j]
                    fi = self.bnd_to_in[3][fj]
                    col = array([ki], 'uintp')
                    val = array([-2. * nu / self.y[yi]**2])
                    var.setrow(ej, col, val)
                    var.setrow(ei, col, val)
                    col[0] = v2j
                    val[0] = 20. * nu**2 / self.y[yi]**4 / self.v2f[ei]
                    var.setrow(fj, col, val)
                    var.setrow(fi, col, val)
                    var.apply('insert')
                    
            if isinstance(var, (Vector, GenericVector)):
                var[e_dofs_ib] = 0.
                var[e_dofs_ob] = 0.
                var[f_dofs_ib] = 0.
                var[f_dofs_ob] = 0.

class Ce1Wall(Wallfunction):
    """Parameter in V2F model that requires wall function due to 
    the term k/v2 that only can be evaluated inside a wall
    """
    def __init__(self, bc, V, ke, k, v2f, v2, Ced):
        Wallfunction.__init__(self, V, bc)
        self.ke = ke
        self.k = k
        self.v2f = v2f
        self.v2 = v2
        self.Ced = Ced
        if parameters['reorder_dofs_serial'] == True or MPI.num_processes() > 1:
            raise TypeError("Not implemented")

    def apply(self, *args):
        aro = array(list(self.dofs_on_boundary[0]), 'intc')
        ari = array(list(self.dofs_inside_boundary[0]), 'intc')
        for var in args:
            if isinstance(var, Matrix):
                var.ident(aro)
                var.ident(ari)
            if isinstance(var, (Vector, GenericVector)):
                for co in self.dofs_on_boundary[0]:
                    dib = self.bnd_to_in[0][co]
                    v2ok = max(min(self.v2f[dib]/self.ke[dib], 2./3.), 0.001)
                    var[dib] = 1.4*(1 + self.Ced*nsqrt(1./v2ok))
                    var[co] = 1.4*(1 + self.Ced*nsqrt(1./v2ok))

class FIJWall_1(Wallfunction):
    """Wall-BC for uncoupled Fij"""
    """Set F11 = -0.5*F22 implicitly, F22 = -20*(v**2/y**4)*vv/e and F12 = -8*(v**2/y**4)*uv/e explicitly."""
    def __init__(self, bc, y, nu, ke, Rij):
        Wallfunction.__init__(self, y.function_space(), bc)
        self.y = y.vector()
        self.N = len(self.y)
        self.nu = nu
        self.ke = ke.vector()
        self.Rij = Rij.vector()
        if not (len(self.ke) == 2*self.N):
            info('Warning! Only works when functionspace of Eikonal is equal to epsilon')
    
    def apply(self, *args):
        """Apply boundary condition to tensors."""
        aro = array(list(self.vertices_on_boundary),'I')
        ari = array(list(self.vertices_inside_boundary),'I')
        for var in args:
            if(isinstance(var, Matrix)):
                var.ident(aro)
                var.ident(ari)
                var.ident(aro + self.N)
                var.ident(ari + self.N)
                var.ident(aro + 2*self.N)
                var.ident(ari + 2*self.N)
                # Set F11 = -0.5*F22 approaching boundaries
                for i in self.vertices_inside_boundary:
                    col = array([i + 2*self.N], 'I')
                    val = array([0.5])
                    var.setrow(i, col, val)
                    var.apply('insert')
                #var[i, i + 2*self.N] = 0.5
                for j in self.vertices_on_boundary:
                    i = self.bnd_to_in[j]
                    col = array([i + 2*self.N], 'I')
                    val = array([0.5])
                    var.setrow(j, col, val)
                    var.apply('insert')
                    #var[j, i + 2*self.N] = 0.5
                    
            if isinstance(var, (Vector, GenericVector)):
                var[aro] = 0.
                var[ari] = 0.
                #Set Fij = - (8 or 10)*nu**2/y**4*Rij
        #for v2f sets eps = max(1e-3,e), maybe do that here too?
        for j in self.vertices_on_boundary:
            i = self.bnd_to_in[j]
            var[j + self.N] = -8.*(self.nu**2/self.y[i]**4)*self.Rij[i+self.N]*(1./self.ke[i +self.N])
            var[j + 2*self.N] = -20.*(self.nu**2/self.y[i]**4)*self.Rij[i+2*self.N]*(1./self.ke[i +self.N])
            for i in self.vertices_inside_boundary:
                var[i + self.N] = -8.*(self.nu**2/self.y[i]**4)*self.Rij[i+self.N]*(1./self.ke[i +self.N])
                var[i + 2*self.N] = -20.*(self.nu**2/self.y[i]**4)*self.Rij[i+2*self.N]*(1./self.ke[i +self.N])

class FIJWall_2_UNSYMMETRIC(Wallfunction):
    """Wall-BC for Rij and Fij coupled"""
    """Set F11 = -0.5*F22 implicitly, F22 = -20*(v**2/y**4)*vv/e and F12 = -8*(v**2/y**4)*uv/e explicitly."""
    def __init__(self, bc, y, nu, ke):
        Wallfunction.__init__(self, y.function_space(), bc)
        self.y = y.vector()
        self.N = len(self.y)
        self.nu = nu
        self.ke = ke.vector()
        if not (len(self.ke) == 2*self.N):
            info('Warning! Only works when functionspace of Eikonal is equal to epsilon')
    
    def apply(self, *args):
        """Apply boundary condition to tensors."""
        aro = array(list(self.vertices_on_boundary),'I')
        ari = array(list(self.vertices_inside_boundary),'I')
        for var in args:
            if(isinstance(var, Matrix)):
                var.ident(aro + 4*self.N)
                var.ident(ari + 4*self.N)
                var.ident(aro + 5*self.N)
                var.ident(ari + 5*self.N)
                var.ident(aro + 6*self.N)
                var.ident(ari + 6*self.N)
                var.ident(aro + 7*self.N)
                var.ident(ari + 7*self.N)

                for i in self.vertices_inside_boundary:
                    col = array([i], 'I')
                    col[0] = i + 7*self.N
                    val = array([0.5])
                    var.setrow(i + 4*self.N, col, val)
                    col[0] = i + self.N
                    val = array([ 8.*(self.nu**2/self.y[i]**4)*(1./self.ke[i +self.N]) ])
                    var.setrow(i + 5*self.N, col, val)
                    col[0] = i + 2*self.N
                    val = array([ 8.*(self.nu**2/self.y[i]**4)*(1./self.ke[i +self.N]) ])
                    var.setrow(i + 6*self.N, col, val)
                    col[0] = i + 3*self.N
                    val = array([ 20.*(self.nu**2/self.y[i]**4)*(1./self.ke[i +self.N]) ])
                    var.setrow(i + 7*self.N, col, val)
                    var.apply('insert')

                for j in self.vertices_on_boundary:
                    i = self.bnd_to_in[j]
                    col = array([i], 'I')
                    col[0] = i + 7*self.N
                    val = array([0.5])
                    var.setrow(j + 4*self.N, col, val)
                    col[0] = i + self.N
                    val = array([ 8.*(self.nu**2/self.y[i]**4)*(1./self.ke[i +self.N]) ])
                    var.setrow(j + 5*self.N, col, val)
                    col[0] = i + 2*self.N
                    val = array([ 8.*(self.nu**2/self.y[i]**4)*(1./self.ke[i +self.N]) ])
                    var.setrow(j + 6*self.N, col, val)
                    col[0] = i + 3*self.N
                    val = array([ 20.*(self.nu**2/self.y[i]**4)*(1./self.ke[i +self.N]) ])
                    var.setrow(j + 7*self.N, col, val)
                    var.apply('insert')
                    
            if(isinstance(var, (Vector, GenericVector))):
                var[aro + 4*self.N] = 0.
                var[ari + 4*self.N] = 0.
                var[aro + 5*self.N] = 0.
                var[ari + 5*self.N] = 0.
                var[aro + 6*self.N] = 0.
                var[ari + 6*self.N] = 0.
                var[aro + 7*self.N] = 0.
                var[ari + 7*self.N] = 0.

class FIJWall_2_UNSYMMETRIC2(Wallfunction):
    """Wall-BC for Rij and Fij coupled"""
    """Set F11 = -0.5*F22 implicitly, F22 = -20*(v**2/y**4)*vv/e and F12 = -8*(v**2/y**4)*uv/e explicitly."""
    def __init__(self, bc, y, nu, ke, ni):
        Wallfunction.__init__(self, y.function_space(), bc)
        self.y = y.vector()
        self.N = len(self.y)
        self.nu = nu
        self.ke = ke.vector()
        self.ni = ni.vector()
        if not (len(self.ke) == 2*self.N):
            info('Warning! Only works when functionspace of Eikonal is equal to epsilon')
    
    def apply(self, *args):
        """Apply boundary condition to tensors."""
        aro = array(list(self.vertices_on_boundary),'I')
        ari = array(list(self.vertices_inside_boundary),'I')
        N = self.N
        nn = self.ni
        for var in args:
            if(isinstance(var, Matrix)):
                # Just keep these because they set everything to zero except 
                # the diagonal that is overloaded anyway
                var.ident(aro + 4*N)
                var.ident(ari + 4*N)
                var.ident(aro + 5*N)
                var.ident(ari + 5*N)
                var.ident(aro + 6*N)
                var.ident(ari + 6*N)
                var.ident(aro + 7*N)
                var.ident(ari + 7*N)

                for i in self.vertices_inside_boundary:
                    colF = array([i + 4*N, i + 5*N, i + 6*N, i + 7*N], 'I')
                    colR = array([i, i + N, i + 2*N, i + 3*N], 'I')
                    n1 = nn[i]
                    n2 = nn[i + N]
                    t1 = n2
                    t2 = -n1
                    valn =  array([n1*n1, n1*n2, n2*n1, n2*n2]) # nn[0,0], nn[0,1], nn[1,0], nn[1,1]
                    valt =  array([t1*t1, t1*t2, t2*t1, t2*t2]) # tt[0,0], tt[0,1], tt[1,0], tt[1,1]
                    valnt = array([n1*t1, n2*t1, n1*t2, n2*t2]) # nt[0,0], nt[0,1], nt[1,0], nt[1,1]
                    valtn = array([t1*n1, t2*n1, t1*n2, t2*n2]) # tn[0,0], tn[0,1], tn[1,0], tn[1,1]
                    
                    #print 'valn  ', i, valn
                    #print 'valt  ', i, valt
                    #print 'valnt ', i, valnt
                    #print 'valtn ', i, valtn
                    
                    # Ftt = -0.5*Fnn
                    var.setrow(i + 4*self.N, colF, valt + 0.5*valn)
                    
                    # Fnt = -8*(nu**2/y**4)*Rnt/e
                    vv = 8.*(self.nu**2/self.y[i]**4)*(1./self.ke[i +self.N])
                    var.setrow(i + 5*self.N, colR, vv*valnt)
                    var.setrow(i + 5*self.N, colF, valnt)
                    # Ftn = -8*(nu**2/y**4)*Rtn/e
                    var.setrow(i + 6*self.N, colR, vv*valtn)
                    var.setrow(i + 6*self.N, colF, valtn)
                    
                    # Fnn = -20*(nu**2/y**4)*Rnn/e
                    vv = 20.*(self.nu**2/self.y[i]**4)*(1./self.ke[i +self.N])
                    var.setrow(i + 7*self.N, colR, vv*valn)
                    var.setrow(i + 7*self.N, colF, valn)
                    var.apply('insert')

                for j in self.vertices_on_boundary:
                    i = self.bnd_to_in[j]
                    colF = array([j + 4*N, j + 5*N, j + 6*N, j + 7*N], 'I')
                    colR = array([i, i + N, i + 2*N, i + 3*N], 'I')
                    n1 = nn[i]
                    n2 = nn[i + N]
                    t1 = -n2
                    t2 = n1
                    valn =  array([n1*n1, n1*n2, n2*n1, n2*n2]) # nn[0,0], nn[0,1], nn[1,0], nn[1,1]
                    valt =  array([t1*t1, t1*t2, t2*t1, t2*t2]) # tt[0,0], tt[0,1], tt[1,0], tt[1,1]
                    valnt = array([n1*t1, n2*t1, n1*t2, n2*t2]) # nt[0,0], nt[0,1], nt[1,0], nt[1,1]
                    valtn = array([t1*n1, t2*n1, t1*n2, t2*n2]) # tn[0,0], tn[0,1], tn[1,0], tn[1,1]
                    
                    # Ftt = -0.5*Fnn
                    var.setrow(j + 4*self.N, colF, valt + 0.5*valn)
                                        
                    # F12 = -8*(nu**2/y**4)*Rnt/e
                    vv = 8.*(self.nu**2/self.y[i]**4)*(1./self.ke[i + N])
                    var.setrow(j + 5*N, colR, vv*valnt)
                    var.setrow(j + 5*N, colF, valnt)
                    var.setrow(j + 6*N, colR, vv*valtn)
                    var.setrow(j + 6*N, colF, valtn)
                    
                    # Fnn = -20*(nu**2/y**4)*Rnn/e
                    vv = 20.*(self.nu**2/self.y[i]**4)*(1./self.ke[i +N])
                    var.setrow(j + 7*N, colR, vv*valn)
                    var.setrow(j + 7*N, colF, valn)
                    var.apply('insert')
                    
            if(isinstance(var, (Vector, GenericVector))):
                var[aro + 4*N] = 0.
                var[ari + 4*N] = 0.
                var[aro + 5*N] = 0.
                var[ari + 5*N] = 0.
                var[aro + 6*N] = 0.
                var[ari + 6*N] = 0.
                var[aro + 7*N] = 0.
                var[ari + 7*N] = 0.

class FIJWall_2(Wallfunction):
    """Wall-BC for Rij and Fij coupled"""
    """Set F11 = -0.5*F22 implicitly, F22 = -20*(v**2/y**4)*vv/e and F12 = -8*(v**2/y**4)*uv/e explicitly."""
    def __init__(self, bc, y, nu, ke):
        Wallfunction.__init__(self, y.function_space(), bc)
        self.y = y.vector()
        self.N = len(self.y)
        self.nu = nu
        self.ke = ke.vector()
        if not (len(self.ke) == 2*self.N):
            info('Warning! Only works when functionspace of Eikonal is equal to epsilon')
    
    def apply(self, *args):
        """Apply boundary condition to tensors."""
        aro = array(list(self.vertices_on_boundary),'I')
        ari = array(list(self.vertices_inside_boundary),'I')
        for var in args:
            if(isinstance(var, Matrix)):
                var.ident(aro + 3*self.N)
                var.ident(ari + 3*self.N)
                var.ident(aro + 4*self.N)
                var.ident(ari + 4*self.N)
                var.ident(aro + 5*self.N)
                var.ident(ari + 5*self.N)
                for i in self.vertices_inside_boundary:
                    col = array([i], 'I')
                    col[0] = i + 5*self.N
                    val = array([0.5])
                    var.setrow(i + 3*self.N, col, val)
                    col[0] = i + self.N
                    val = array([ 8.*(self.nu**2/self.y[i]**4)*(1./self.ke[i +self.N]) ])
                    var.setrow(i + 4*self.N, col, val)
                    col[0] = i + 2*self.N
                    val = array([ 20.*(self.nu**2/self.y[i]**4)*(1./self.ke[i +self.N]) ])
                    var.setrow(i + 5*self.N, col, val)
                    var.apply('insert')
                #var[i + 3*N , i + 5*self.N] = 0.5
                #var[i + 4*N , i + 1*self.N] = 8.*(self.nu**2/self.y[i]**4)*(1./self.ke[i +self.N])
                #var[i + 5*N , i + 2*self.N] = 20.*(self.nu**2/self.y[i]**4)*(1./self.ke[i +self.N])
                for j in self.vertices_on_boundary:
                    i = self.bnd_to_in[j]
                    col = array([i], 'I')
                    col[0] = j + 5*self.N
                    val = array([0.5])
                    var.setrow(j + 3*self.N, col, val)
                    col[0] = i + self.N
                    val = array([ 8.*(self.nu**2/self.y[i]**4)*(1./self.ke[i +self.N]) ])
                    var.setrow(j + 4*self.N, col, val)
                    col[0] = i + 2*self.N
                    val = array([ 20.*(self.nu**2/self.y[i]**4)*(1./self.ke[i +self.N]) ])
                    var.setrow(j + 5*self.N, col, val)
                    var.apply('insert')
                #var[j + 3*N , j + 5*self.N] = 0.5
                #var[j + 4*N , i + 1*self.N] = 8.*(self.nu**2/self.y[i]**4)*(1./self.ke[i +self.N])
                #var[j + 5*N , i + 2*self.N] = 20.*(self.nu**2/self.y[i]**4)*(1./self.ke[i +self.N])
                    
            if(isinstance(var, (Vector, GenericVector))):
                var[aro + 3*self.N] = 0.
                var[ari + 3*self.N] = 0.
                var[aro + 4*self.N] = 0.
                var[ari + 4*self.N] = 0.
                var[aro + 5*self.N] = 0.
                var[ari + 5*self.N] = 0.

class FIJWall_3(Wallfunction):
    """Wall-BC for fully coupled ER system"""
    """Set F11 = -0.5*F22 implicitly, F22 = -20*(v**2/y**4)*vv/e and F12 = -8*(v**2/y**4)*uv/e explicitly."""
    def __init__(self, bc, y, nu, ke):
        Wallfunction.__init__(self, y.function_space(), bc)
        self.y = y.vector()
        self.N = len(self.y)
        self.nu = nu
        self.ke = ke.vector()
        if not (len(self.ke) == 2*self.N):
            info('Warning! Only works when functionspace of Eikonal is equal to epsilon')
            
    def apply(self, *args):
        """Apply boundary condition to tensors."""
        aro = array(list(self.vertices_on_boundary),'I')
        ari = array(list(self.vertices_inside_boundary),'I')
        for var in args:
            if(isinstance(var, Matrix)):
                var.ident(aro)
                var.ident(ari)
                var.ident(aro + self.N)
                var.ident(ari + self.N)
                var.ident(aro + 2*self.N)
                var.ident(ari + 2*self.N)
                # Set F11 = -0.5*F22 approaching boundaries
                for i in self.vertices_inside_boundary:
                    col = array([i], 'I')
                    col[0] = i + 2*self.N
                    val = array([0.5])
                    var.setrow(i, col, val)
                    var.apply('insert')
                #var[i, i + 2*self.N] = 0.5
                for j in self.vertices_on_boundary:
                    i = self.bnd_to_in[j]
                    col = array([i], 'I')
                    col[0] = i + 2*self.N
                    val = array([0.5])
                    var.setrow(j, col, val)
                    var.apply('insert')
                    #var[j, i + 2*self.N] = 0.5
                    
            if isinstance(var, (Vector, GenericVector)):
                var[aro] = 0.
                var[ari] = 0.
                #Set Fij = - (2 or 5)*e/k**2*Rij
        #for v2f sets eps = max(1e-3,e), maybe do that here too?
        for j in self.vertices_on_boundary:
            var[j + self.N] = -8.*(self.nu**2/self.y[j]**4)*self.Rij[j+self.N]*(1./self.ke[j +self.N])
            var[j + 2*self.N] = -20.*(self.nu**2/self.y[j]**4)*self.Rij[j+2*self.N]*(1./self.ke[j +self.N])
            for i in self.vertices_inside_boundary:
                var[i + self.N] = -8.*(self.nu**2/self.y[i]**4)*self.Rij[i+self.N]*(1./self.ke[i +self.N])
                var[i + 2*self.N] = -20.*(self.nu**2/self.y[i]**4)*self.Rij[i+2*self.N]*(1./self.ke[i +self.N])

# Collect all boundary conditions in a common dictionary
QWall = dict(Ce1=Ce1Wall, ke=KEWall, v2f=FWall, kev2f=V2FWall, Fij= FIJWall_1, Fij_2 = FIJWall_2_UNSYMMETRIC2)
