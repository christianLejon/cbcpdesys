__author__ = "Mikael Mortensen <mikaem@math.uio.no>"
__date__ = "2012-01-7"
__copyright__ = "Copyright (C) 2011 " + __author__
__license__  = "GNU Lesser GPL version 3 or any later version"
"""
This module contains functionality for Lagrangian tracking of particles 
"""
#from cbc.pdesys import *
from dolfin import Point, Cell, cells, project, grad, RectangleMesh, UnitSquareMesh, FunctionSpace, VectorFunctionSpace, TensorFunctionSpace, interpolate, Expression, parameters
import ufc
from numpy import linspace, pi, squeeze, eye, outer, zeros, ones, where, array, ndarray, squeeze, load, sin, cos, arcsin, sqrt, arctan, resize, dot as ndot
from pylab import figure, scatter, show, quiver, axis, axes, savefig
from copy import copy, deepcopy
import time
from mpi4py import MPI as nMPI

comm = nMPI.COMM_WORLD
parameters["form_compiler"]["optimize"]     = True
parameters["form_compiler"]["cpp_optimize"] = True
#parameters["form_compiler"].add('no-evaluate_basis_derivatives', False)

class celldict(dict):
    
    def __add__(self, (key, value)):
        
        if key in self:
            self[key].append(value)
        else:
            self[key] = [value]
            
        return self            

class CellParticleMap(dict):
    
    def __add__(self, ins):
        if isinstance(ins, tuple) and len(ins) in (3, 4):
            if ins[1] in self: # if the cell is already in the dict
                self[ins[1]] += ins[2]
            else:
                self[ins[1]] = CellWithParticles(ins[0], ins[1], ins[2])
            if len(ins) == 4:
                self[ins[1]].particles[-1].prm.update(ins[3])
        else:
            raise TypeError("Wrong numer of arguments to CellParticleMap")
        return self
    
    def pop(self, c, i):
        particle = self[c].particles.pop(i)
        if self[c].particles == []:
            del self[c]
        return particle
            
    def total_number_of_particles(self):
        count = 0
        for cell in self.itervalues():
            count += len(cell.particles)
        return count
        
class CellWithParticles(Cell):
    
    def __init__(self, mesh, c, particle):
        Cell.__init__(self, mesh, c)
        self.particles = []
        self += particle
                    
    def __add__(self, particle):
        if isinstance(particle, ndarray):
            self.particles.append(Particle(particle))
        elif isinstance(particle, Particle):
            self.particles.append(particle)
        else:
            raise TypeError("Wrong type of particle")
        return self
                        
class Particle:
    
    def __init__(self, x):
        self.position = x
        self.prm = {}  # Simple parameters dictionary to attach anything to a particle
        
    def send(self, dest):
        """Send particle to dest"""
        if not isinstance(dest, (int, list)):
            raise TypeError("Provide a list of int destinations or a single int")
        dest = [dest] if isinstance(dest, int) else dest
        for i in dest:
            comm.Send(self.position, dest=i)
            comm.send(self.prm, dest=i)
            
    def recv(self, source):
        """Receive info of a new particle sent from another process"""
        if not isinstance(source, (int, list)):
            raise TypeError("Provide a list of int sources or a single int")
        source = [source] if isinstance(source, int) else source        
        for i in source:
            comm.Recv(self.position, source=i)
            self.prm = comm.recv(source=i)

class LagrangianParticlesPosition:
    """Lagrangian tracking of massless particles in a velocity field
    from the VectorFunctionSpace V."""
    def __init__(self, V):
        self.V = V
        self.mesh = V.mesh()
        self.mesh.init(2, 2)
        
        # Allocate some variables used to look up the velocity
        self.ufc_cell = ufc.cell()         # Empty cell
        self.element = V.dolfin_element()
        self.dim = self.mesh.topology().dim()
        self.num_tensor_entries = 1
        for i in range(self.element.value_rank()):
            self.num_tensor_entries *= self.element.value_dimension(i)
        self.coefficients = zeros(self.element.space_dimension())
        self.basis_matrix = zeros((self.element.space_dimension(), 
                                   self.num_tensor_entries))                                   
        
        # Allocate a dictionary to hold all particles
        self.cellparticles = CellParticleMap()
        
        # Allocate some MPI stuff
        self.num_processes = comm.Get_size()
        self.myrank = comm.Get_rank()
        self.all_processes = range(self.num_processes)
        self.other_processes = range(self.num_processes)
        self.other_processes.remove(self.myrank)
        self.my_escaped_particles = zeros(1, dtype='I')
        self.tot_escaped_particles = zeros(self.num_processes, dtype='I')
        self.particle0 = Particle(zeros(self.mesh.geometry().dim()))
        self.verbose = False
            
    def add_particles(self, list_of_particles, prm=None):
        """Add particles and search for their home on all processors. 
        list_of_particles must be identical on all processors before 
        calling this function.
        prm is an optional dictionary of additional info.
        The prm dictionary must have values that are lists of the 
        same length as list_of_particles.
        """
        cwp = self.cellparticles        
        my_found = zeros(len(list_of_particles), 'I')
        all_found = zeros(len(list_of_particles), 'I')
        if not prm == None:
            particle_prm = dict((key, 0) for key in prm.keys())
        for i, particle in enumerate(list_of_particles): # particle or array
            c = self.locate(particle)
            if not c == -1:
                my_found[i] = 1
                if prm == None:
                    cwp += self.mesh, c, particle
                else:
                    for key in prm.keys():
                        particle_prm[key] = prm[key][i]
                    cwp += self.mesh, c, particle, particle_prm
        
        comm.Reduce(my_found, all_found, root=0)
        if self.myrank == 0:
            x = where(all_found == 0)
            if not x[0].shape[0] == 0:                    
                print 'Number of particles not found = ', x[0].shape[0]
                
    def add_particles_ring(self, list_of_particles):
        """Add particles on process 0 and search in ring through all 
        processors until found."""
        cwp = self.cellparticles
        num_particles = comm.bcast(len(list_of_particles), 0)
        particle0 = zeros(self.mesh.geometry().dim())
        if self.myrank == 0:
            not_found = 0
            found_particles = zeros(len(list_of_particles), 'I')
            for particle in list_of_particles:
                c = self.locate(particle)
                if c > -1:
                    cwp += self.mesh, c, particle
                if self.num_processes > 1:
                    comm.Send(particle, dest=1, tag=101)
                    comm.send(c, dest=1, tag=102)
                    c = comm.recv(source=self.num_processes-1, tag=102)
                if c == -1:
                    not_found += 1
                    #print 'Particle not found on any processor', particle
            print 'Total number of particles not found = ', not_found            
            
        else: # Send the particle in a ring    
            for j in range(num_particles):
                comm.Recv(particle0, source=self.myrank-1, tag=101)
                p0 = deepcopy(particle0)
                c = comm.recv(source=self.myrank-1, tag=102)        
                if c < 0: # Look for it because particle is still not found
                    c = self.locate(p0)
                    if c > -1: # found it!
                        cwp += self.mesh, c, p0
                comm.send(c, dest=(self.myrank+1) % self.num_processes, tag=102)
                if self.myrank < self.num_processes:
                    comm.Send(p0, dest=(self.myrank+1) % self.num_processes, tag=101)        
                                
    def step(self, u, dt, duidxj=False):
        """Move particles one timestep using velocity u and timestep dt."""
        for cell in self.cellparticles.itervalues():
            u.restrict(self.coefficients, self.element, cell, self.ufc_cell)
            for particle in cell.particles:
                x = particle.position
                self.element.evaluate_basis_all(self.basis_matrix, x, cell)
                x[:] = x[:] + dt*ndot(self.coefficients, self.basis_matrix)[:]
                
        self.relocate()
                
    def relocate(self):
        # Relocate particles on cells and processors                
        cwp = self.cellparticles 
        myrank = self.myrank                
        new_cell_map = celldict()
        for cell in cwp.itervalues():
            for i, particle in enumerate(cell.particles):
                point = Point(*particle.position)
                if cell.intersects(point):# particle is still in the same cell
                    pass
                else:
                    found = False 
                    # Check neighbor cells
                    for neighbor in cells(cell): 
                        if neighbor.intersects(point):
                            new_cell_map += cell.index(), (neighbor.index(), i)
                            found = True
                            break
                    # Do a completely new search if not found by now
                    if not found:
                        c = self.locate(point)
                        new_cell_map += cell.index(), (c, i)
                                
        list_of_escaped_particles = [] 
        for oldcell, newcells in new_cell_map.iteritems():
            newcells.reverse()
            for (newcell, i) in newcells:
                particle = cwp.pop(oldcell, i)
                if newcell == -1:# With MPI the particles may travel between processors. Capture these traveling particles here
                    list_of_escaped_particles.append(particle)
                else:
                    cwp += self.mesh, newcell, particle
                                
        # Create a list of how many particles escapes from each processor
        self.my_escaped_particles[0] = len(list_of_escaped_particles)        
        comm.Allgather(self.my_escaped_particles, self.tot_escaped_particles)
        
        # Print for debugging etc.
        if self.verbose:
            print 'Escaped ', myrank, list_of_escaped_particles
            if self.myrank==0:
                print 'Total escaped = ', self.tot_escaped_particles
        
        # Send the escaping particles to the other processes
        # For now send to myrank=0. Could send only to neighbors using, e.g., info in parallel_data().shared_vertices()
        for particle in list_of_escaped_particles:
            particle.send(0)
                
        # Receive the particles escaping from other processors
        if self.myrank == 0:
            for j in self.other_processes:
                for i in range(self.tot_escaped_particles[j]):
                    self.particle0.recv(j)
                    list_of_escaped_particles.append(deepcopy(self.particle0))
                    
        # Put all the travelling particles on all processes, then perform new search
        travelling_particles = comm.bcast(list_of_escaped_particles, root=0)
        self.add_particles(travelling_particles)
        
    def total_number_of_particles(self):
        num_p = self.cellparticles.total_number_of_particles()
        if self.verbose: print "particles on proc ", self.myrank, num_p    
        tot_p = comm.reduce(num_p, root=0)    
        if self.myrank == 0 and self.verbose:
            print 'Total num particles ', tot_p
        return tot_p            
        
    def locate(self, x):
        if isinstance(x, Point):
            point = x
        elif isinstance(x, ndarray):
            point = Point(*x)
        elif isinstance(x, Particle):
            point = Point(*x.position)
        return self.mesh.intersected_cell(point)
        
    def scatter(self, skip=1):
        """Put all particles on processor 0 for scatter plotting"""
        cwp = self.cellparticles
        all_particles = zeros(self.num_processes, dtype='I')
        my_particles = cwp.total_number_of_particles()
        comm.Gather(array([my_particles], 'I'), all_particles, root=0)
        if self.myrank > 0: # Send all particles from processes > 0 to 0
            for cell in cwp.itervalues():
                for p in cell.particles:
                    p.send(0)
        else:               # Receive on proc 0
            recv = [copy(p.position) for cell in cwp.itervalues() for p in cell.particles]
            for i in self.other_processes:
                for j in range(all_particles[i]):
                    self.particle0.recv(i)
                    recv.append(copy(self.particle0.position))
            xx = array(recv)
            scatter(xx[::skip, 0], xx[::skip, 1])

class OrientedParticles(LagrangianParticlesPosition):
    """
    Oriented particles solve Lagrangian equations for position and 
    a normal vector. The normal vector points from one fluid into 
    another. For example, for hydrodynamic waves the surface defines
    the interface between water and air. The position of the particles
    then give the location of the surface wave, whereas the normal 
    vectors point into the air phase.
    """
    def __init__(self, V, S=None):
        """Oriented particles require the velocity gradient. The velocity
        gradient can be computed from the velocity Function residing in
        VectorFunctionSpace V, or it can be used directly through a Function
        in the TensorFunctionSpace S."""
        LagrangianParticlesPosition.__init__(self, V)
        if S:
            self.S = S
            self.Selement = S.dolfin_element()
            self.Snum_tensor_entries = 1
            for i in range(self.Selement.value_rank()):
                self.Snum_tensor_entries *= self.Selement.value_dimension(i)
            self.Scoefficients = zeros(self.Selement.space_dimension())
            self.Sbasis_matrix = zeros((self.Selement.space_dimension(), 
                                        self.Snum_tensor_entries))                                   
            self.Sbasis_matrix_du = zeros((self.Selement.space_dimension(), 
                                        self.Snum_tensor_entries*self.dim))
        
    def add_particles(self, list_of_particles, prm=None, add_phi_mag=True):
        if len(list_of_particles) == 0: return
        if prm == None and isinstance(list_of_particles[0], ndarray):
            raise TypeError("Oriented particles must have position and normal vector as initial conditions")
        if add_phi_mag and isinstance(list_of_particles[0], ndarray):
            prm['phi_mag'] = ones((len(list_of_particles), 1))*0.5
        LagrangianParticlesPosition.add_particles(self, list_of_particles, prm)
        
    def step(self, u, dt, duidxj=False):
        """Move particles one timestep using velocity u and timestep dt.
        The velocity strain can be provided through tensor duidxj. If duidxj
        is not provided it will be computed from u."""
        cwp = self.cellparticles 
        myrank = self.myrank
        # Get particle velocities and move
        for cell in cwp.itervalues():
            u.restrict(self.coefficients, self.element, cell, self.ufc_cell)
            if duidxj:
                duidxj.restrict(self.Scoefficients, self.Selement, cell, self.ufc_cell)
            for particle in cell.particles:
                x = particle.position
                self.element.evaluate_basis_all(self.basis_matrix, x, cell)
                du = ndot(self.coefficients, self.basis_matrix)
                if duidxj:
                    self.Selement.evaluate_basis_all(self.Sbasis_matrix, x, cell)
                    dudx = ndot(self.Scoefficients, self.Sbasis_matrix).reshape((self.dim, self.dim)).transpose()
                else:
                    self.element.evaluate_basis_derivatives_all(1, self.basis_matrix_du, x, cell)
                    dudx = ndot(self.coefficients, self.basis_matrix_du).reshape((self.dim, self.dim)).transpose()
                n = particle.prm['normal']
                dujdxk_njnk = ndot(ndot(dudx, n), n)
                duidxj_nj = ndot(dudx, n)
                # Move particle position
                x[:] = x[:] + dt*du[:]
                # Evolve normal
                n[:] = n[:] + dt*(-duidxj_nj[:] + dujdxk_njnk*n[:])
                n[:] = n[:]/sqrt(ndot(n, n))
                # Evolve phi_mag
                phim = particle.prm['phi_mag']
                phim[0] = phim[0] - dt*2.*phim[0]*dujdxk_njnk
  
        self.relocate()

    def scatter(self, factor=1., skip=1):
        """Put all particles on processor 0 for scatter plotting"""
        cwp = self.cellparticles
        all_particles = zeros(self.num_processes, dtype='I')
        my_particles = cwp.total_number_of_particles()
        comm.Gather(array([my_particles], 'I'), all_particles, root=0)
        if self.myrank > 0: # Send all particles from processes > 0 to 0
            for cell in cwp.itervalues():
                for p in cell.particles:
                    p.send(0)
        else:               # Receive on proc 0
            recv = [copy(p.position) for cell in cwp.itervalues() for p in cell.particles]
            recn = [copy(p.prm['normal']) for cell in cwp.itervalues() for p in cell.particles]
            rscp = [copy(p.prm['phi_mag']) for cell in cwp.itervalues() for p in cell.particles]
                    
            for i in self.other_processes:
                for j in range(all_particles[i]):
                    self.particle0.recv(i)
                    recv.append(copy(self.particle0.position))
                    recn.append(copy(self.particle0.prm['normal']))
                    rscp.append(copy(self.particle0.prm['phi_mag']))
                            
            xx = array(recv)
            scatter(xx[::skip, 0], xx[::skip, 1])
            xn = array(recn)
            rscp = squeeze(array(rscp))[::skip]
            quiver(xx[::skip, 0], xx[::skip, 1], xn[::skip, 0], xn[::skip, 1], rscp, scale=factor)

class OrientedParticlesWithShape(OrientedParticles):
    """
    Oriented particles with shape solve Lagrangian equations for 
    position, a normal vector and the gradient of the normal vector.
    """
    def __init__(self, V, S):
        OrientedParticles.__init__(self, V, S)

    def step(self, u, dt, duidxj):
        """Move particles one timestep using velocity u and timestep dt."""
        cwp = self.cellparticles 
        myrank = self.myrank
        # Get particle velocities and move
        for cell in cwp.itervalues():
            u.restrict(self.coefficients, self.element, cell, self.ufc_cell)
            duidxj.restrict(self.Scoefficients, self.Selement, cell, self.ufc_cell)
            for particle in cell.particles:
                x = particle.position
                self.element.evaluate_basis_all(self.basis_matrix, x, cell)
                du = ndot(self.coefficients, self.basis_matrix)
                self.Selement.evaluate_basis_all(self.Sbasis_matrix, x, cell)
                dudx = ndot(self.Scoefficients, self.Sbasis_matrix).reshape((self.dim, self.dim)).transpose()
                self.Selement.evaluate_basis_derivatives_all(1, self.Sbasis_matrix_du, x, cell)
                d2udxidxj = ndot(self.Scoefficients, self.Sbasis_matrix_du).reshape((self.dim, self.dim, self.dim)).transpose()
                n = particle.prm['normal']
                dndx = particle.prm['dndx']
                dujdxk_njnk = ndot(ndot(dudx, n), n)
                duidxj_nj = ndot(dudx, n)
                d2ujdxis_nj = ndot(d2udxidxj, n)
                d2ujdxks_njnkni = outer(ndot(d2ujdxis_nj, n), n)
                dujdxi_dnjdxs = ndot(dudx, dndx.transpose()).transpose()
                dujdxk_dnjdxs_nkni = outer(ndot(dujdxi_dnjdxs, n), n)
                dujdxk_nj_dnkdxs_ni = outer(ndot(dndx, duidxj_nj), n)
                dujdxk_njnk_dnidxs =  ndot(duidxj_nj, n)*dndx                
                dujdxs_dnidxj = ndot(dudx, dndx)
                # Move particle
                x[:] = x[:] + dt*du[:]
                # Move normal
                n[:] = n[:] + dt*(-duidxj_nj[:] + dujdxk_njnk*n[:])
                n[:] = n[:]/sqrt(ndot(n, n))
                # Move gradient
                dndx[:, :] = dndx[:, :] + dt*(d2ujdxks_njnkni[:, :] + dujdxk_dnjdxs_nkni[:, :] + dujdxk_nj_dnkdxs_ni[:, :] + dujdxk_njnk_dnidxs[:, :] 
                                                - d2ujdxis_nj[:, :] - dujdxi_dnjdxs[:, :] - dujdxs_dnidxj[:, :])
                if 'phi_mag' in particle.prm:
                    phim = particle.prm['phi_mag']
                    phim[0] = phim[0] - dt*2.*phim[0]*dujdxk_njnk
                
        self.relocate()

    def scatter(self, factor=1., skip=1):
        """Put all particles on processor 0 for scatter plotting"""
        cwp = self.cellparticles
        all_particles = zeros(self.num_processes, dtype='I')
        my_particles = cwp.total_number_of_particles()
        comm.Gather(array([my_particles], 'I'), all_particles, root=0)
        if self.myrank > 0: # Send all particles from processes > 0 to 0
            for cell in cwp.itervalues():
                for p in cell.particles:
                    p.send(0)
        else:               # Receive on proc 0
            recv = [copy(p.position) for cell in cwp.itervalues() for p in cell.particles]
            recn = [copy(p.prm['normal']) for cell in cwp.itervalues() for p in cell.particles]
            rscp = [copy(p.prm['dndx'].trace()) for cell in cwp.itervalues() for p in cell.particles]
                
            for i in self.other_processes:
                for j in range(all_particles[i]):
                    self.particle0.recv(i)
                    recv.append(copy(self.particle0.position))
                    recn.append(copy(self.particle0.prm['normal']))
                    rscp.append(copy(self.particle0.prm['dndx'].trace()))
                            
            xx = array(recv)
            scatter(xx[::skip, 0], xx[::skip, 1])
            xn = array(recn)
            rscp = squeeze(array(rscp))[::skip]
            quiver(xx[::skip, 0], xx[::skip, 1], xn[::skip, 0], xn[::skip, 1], rscp, scale=factor)

def LagrangianParticles(V, S=None, normal=False, dndx=False):
    if dndx:
        return OrientedParticlesWithShape(V, S)
    elif normal: 
        return OrientedParticles(V, S)
    else:
        return LagrangianParticlesPosition(V)
           
def zalesak(center=(0, 0), radius=15, width=5, slot_length=25, N=50, normal=False):
    """Create points evenly distributed on Zalesak's disk 
    """
    theta = arcsin(width/2./radius)
    l0 = radius*cos(theta)
    disk_length = width + 2.*slot_length + 2.*radius*(pi - theta)
    all_points = linspace(0, disk_length, N, endpoint=False)
    y0 = center[1] + slot_length - l0
    x0 = center[0]
    x = zeros(N)
    y = zeros(N)
    points = []
    if normal: nn = []
    for i, point in enumerate(all_points):
        if point <= width/2.:
            y[i] = y0
            x[i] = x0 + point
            nm = array([0., -1.])
        elif point <= width/2. + slot_length:
            y[i] = y0 - (point - width/2.)
            x[i] = x0 + width/2.
            nm = array([-1., 0.])
        elif point <= disk_length - width/2. - slot_length:
            phi = theta + (point - slot_length - width/2.)/radius
            #y[i] = radius*(1 - cos(phi)) + y0 - slot_length
            y[i] = center[1] - radius*cos(phi)
            x[i] = radius*sin(phi) + x0
            nm = array([x[i]-center[0], y[i]-center[1]])/sqrt((x[i]-center[0])**2 + (y[i]-center[1])**2)
        elif point <= disk_length - width/2.:
            y[i] = y0 - (disk_length - point - width/2.)
            x[i] = x0 - width/2.
            nm = array([1., 0.])
        else:
            y[i] = y0
            x[i] = x0 - (disk_length - point)
            nm = array([0., -1.])
        points.append(array([x[i], y[i]]))
        if normal:
            nn.append(nm)
    if normal:
        return points, nn
    else:
        return points
    
def line(x0, y0, dx, dy, L, N=10):
    """Create points evenly distributed on a line"""
    dL = linspace(0, L, N)
    theta = arctan(dy/dx)
    x = x0 + dL*cos(theta)
    y = y0 + dL*sin(theta)
    points = []
    for xx, yy in zip(x, y):
        points.append(array([xx, yy]))
    return points

def random_circle(x0, radius, N=10):
    from numpy.random import rand
    r0 = rand(N)*radius
    theta = rand(N)*2*pi
    points = []
    for xx, yy in zip(r0, theta):
        points.append(array([x0[0] + xx*cos(yy), x0[1] + xx*sin(yy)]))
    return points 

def circle(x0, radius=0.15, N=100, normal=False, dndx=False):
    theta = linspace(0, 2.*pi, N, endpoint=False)
    points = []
    if normal: 
        nn = []
    if dndx:
        dn = []
    for t in theta:
        points.append(array([x0[0] + radius*cos(t), x0[1] + radius*sin(t)]))
        if normal:
            n = array([cos(t), sin(t)])
            nn.append(n)
        if dndx:            
            dn.append(1./radius*(eye(2)-outer(n, n)))
    if normal and dndx:
        return points, nn, dn
    elif normal and not dndx:
        return points, nn
    else:
        return points
    
#def main():
if __name__=='__main__':    
    #mesh = RectangleMesh(0, 0, 100, 100, 50, 50)
    mesh = UnitSquareMesh(100, 100)
    V = VectorFunctionSpace(mesh, 'CG', 2)
    
    #u = interpolate(Expression(('pi/314.*(50.-x[1])', 
                                #'pi/314.*(x[0]-50.)')), V)
    #u = interpolate(Expression(('pi/314.*(50.-x[1])*sqrt((x[0]-50.)*(x[0]-50.)+(x[1]-50.)*(x[1]-50.))', 
                                #'pi/314.*(x[0]-50.)*sqrt((x[0]-50.)*(x[0]-50.)+(x[1]-50.)*(x[1]-50.))')), V)
    u = interpolate(Expression(('-2.*sin(pi*x[1])*cos(pi*x[1])*sin(pi*x[0])*sin(pi*x[0])', 
                                '2.*sin(pi*x[0])*cos(pi*x[0])*sin(pi*x[1])*sin(pi*x[1])')), V)
    #u = interpolate(Expression(('x[0]*x[0]+0.5*x[1]*x[0]', 'x[1]*x[1]+1.5*x[0]*x[1]')), V)    
    S = TensorFunctionSpace(mesh, 'CG', 2)
    duidxj = project(grad(u), S)
                                  
    #u = interpolate(Expression(('1.', '0.')), Vv)
    u.update() # Required for parallel
    duidxj.update()

    # Initialize particles
    # Note, with a random function one must compute points on 0 and bcast to get the same values on all procs
    x = []
    nn = []    
    dn = []
    if comm.Get_rank() == 0:        
        #pass
        #x, nn = zalesak(center=(50, 75), N=100, normal=True)    
        #x, nn = circle((0.5, 0.75), N=500, normal=True)
        x, nn, dn = circle((0.5, 0.75), N=1000, normal=True, dndx=True)
        #x = line(x0=39.5, y0=40, dx=1, dy=1, L=20, N=4)    
        #x = [array([0.39, 0.4]), array([0.6, 0.61]), array([0.49, 0.5])]
        #x = random_circle((50, 50), 50, N=1000)
        
    x = comm.bcast(x, root=0) 
    nn = comm.bcast(nn, root=0)
    dn = comm.bcast(dn, root=0)
    #lp = LagrangianParticlesPosition(V)
    #lp = OrientedParticles(V, S)
    #lp = OrientedParticlesWithShape(V, S)    
    lp = LagrangianParticles(V, S, dndx=True)
    
    t0 = time.time()    
    #lp.add_particles(x, prm={'normal': nn})
    lp.add_particles(x, prm={'normal': nn, 'dndx': dn})    
    #xx = array([0.25, 0.75])
    #c = mesh.intersected_cell(Point(xx[0], xx[1]))
    #cell = Cell(mesh, c)
    #duidxj.restrict(lp.Scoefficients, lp.Selement, cell, lp.ufc_cell)
    #lp.Selement.evaluate_basis_all(lp.Sbasis_matrix, xx, cell)
    #dudx = ndot(lp.Scoefficients, lp.Sbasis_matrix).reshape((lp.dim, lp.dim))
    #lp.Selement.evaluate_basis_derivatives_all(1, lp.Sbasis_matrix_du, xx, cell)
    #d2udxidxj = ndot(lp.Scoefficients, lp.Sbasis_matrix_du).reshape((lp.dim, lp.dim, lp.dim)).transpose()

    ##lp.add_particles_ring(x)
    ##print comm.Get_rank(), ' time ', time.time() - t0
    dt = 0.005
    t = 0
    lp.scatter()
    t0 = time.time()
    tstep = 0
    count = 0
    while t < 1.5:
        t = t + dt
        tstep += 1
        lp.step(u, dt, duidxj=duidxj)  
        if tstep % 20 == 0:
            if lp.myrank == 0: 
                print 'Plotting at ', t
            figure()
            lp.scatter(factor=10., skip=1)    
            count += 1
            axis([0.1, 1, 0.1, 1])
            #axis('equal')
            #savefig('zalesak_' + str(count) + '.jpg')
            
    figure()
    lp.scatter(skip=1, factor=10.)    
    print 'Computing time ', time.time() - t0
    print 'Total number of particles = ', lp.total_number_of_particles()
    show()    
    #return lp, u

#def profiler():
    #import cProfile
    #import pstats
    #cProfile.run('main()', 'mainprofile')
    #p = pstats.Stats('mainprofile')
    #p.sort_stats('time').print_stats(50)
    
#if __name__=='__main__':
    ##profiler()
    #lp, u = main()
    #show()            

    #mf = MeshFunction('uint', lp.mesh, 2)
    #mf.set_all(0)
    #for cell in lp.cellparticles.iterkeys():
        #mf[cell] += 1
        
    #plot(mf)
