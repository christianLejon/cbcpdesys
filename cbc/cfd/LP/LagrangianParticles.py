__author__ = "Mikael Mortensen <mikaem@math.uio.no>"
__date__ = "2012-01-7"
__copyright__ = "Copyright (C) 2011 " + __author__
__license__  = "GNU GPL version 3 or any later version"
"""
This module contains functionality for Lagrangian tracking of particles 
"""
#from cbc.pdesys import *
from dolfin import Point, Cell, cells, Rectangle, FunctionSpace, VectorFunctionSpace, interpolate, Expression, parameters
import ufc
from numpy import linspace, pi, zeros, where, array, ndarray, squeeze, load, sin, cos, arcsin, arctan, resize, dot as ndot
from pylab import scatter, show
from copy import copy, deepcopy
import time
from mpi4py import MPI as nMPI

comm = nMPI.COMM_WORLD
parameters["form_compiler"]["optimize"]     = True
parameters["form_compiler"]["cpp_optimize"] = True

class celldict(dict):
    
    def __add__(self, (key, value)):
        
        if key in self:
            self[key].append(value)
        else:
            self[key] = [value]
            
        return self            

class CellParticleMap(dict):
    
    def __add__(self, ins):
        if isinstance(ins, tuple) and len(ins) == 3:
            if ins[1] in self:
                self[ins[1]] += ins[2]
            else:
                self[ins[1]] = CellWithParticles(*ins)
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

class LagrangianParticles:
    """Base class for all particles"""
    def __init__(self, V):
        self.V = V
        self.mesh = V.mesh()
        self.mesh.init(2, 2)
        
        # Allocate some variables used to look up the velocity
        self.ufc_cell = ufc.cell()         # Empty cell
        self.element = V.dolfin_element()
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
            
    def add_particles(self, list_of_particles):
        """Add particles and search for their home on all processors. 
        list_of_particles must be identical on all processors before 
        calling this function.
        """
        cwp = self.cellparticles        
        my_found = zeros(len(list_of_particles), 'I')
        all_found = zeros(len(list_of_particles), 'I')
        for i, particle in enumerate(list_of_particles): # particle or array
            c = self.locate(particle)
            if not c == -1:
                my_found[i] = 1
                cwp += self.mesh, c, particle
        
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
                                
    def step(self, u, dt):
        """Move particles one timestep using velocity u and timestep dt."""
        cwp = self.cellparticles 
        myrank = self.myrank
        # Get particle velocities and move
        for cell in cwp.itervalues():
            u.restrict(self.coefficients, self.element, cell, self.ufc_cell)
            for particle in cell.particles:
                x = particle.position
                self.element.evaluate_basis_all(self.basis_matrix, x, cell)
                du = ndot(self.coefficients, self.basis_matrix)
                particle.prm['velocity'] = du # Just for fun remember the velocity. Could use this for higher order schemes
                x[:] = x[:] + dt*du[:]
                
        # Relocate particles on cells and processors                
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
        
    def locate(self, x):
        if isinstance(x, Point):
            point = x
        elif isinstance(x, ndarray):
            point = Point(*x)
        elif isinstance(x, Particle):
            point = Point(*x.position)
        return self.mesh.intersected_cell(point)
        
    def scatter(self):
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
            scatter(xx[:, 0], xx[:, 1])
            
def zalesak(center=(0, 0), radius=15, width=5, sloth_length=25, N=50):
    """Create points evenly distributed on Zalesak's disk 
    """
    theta = arcsin(width/2./radius)
    l0 = radius*cos(theta)
    disk_length = width + 2.*sloth_length + 2.*radius*(pi - theta)
    all_points = linspace(0, disk_length, N, endpoint=False)
    y0 = center[1] + sloth_length - l0
    x0 = center[0]
    x = zeros(N)
    y = zeros(N)
    points = []
    for i, point in enumerate(all_points):
        if point <= width/2.:
            y[i] = y0
            x[i] = x0 + point
        elif point <= width/2. + sloth_length:
            y[i] = y0 - (point - width/2.)
            x[i] = x0 + width/2.
        elif point <= disk_length - width/2. - sloth_length:
            phi = theta + (point - sloth_length - width/2.)/radius
            y[i] = radius*(1 - cos(phi)) + y0 - sloth_length
            x[i] = radius*sin(phi) + x0 
        elif point <= disk_length - width/2.:
            y[i] = y0 - (disk_length - point - width/2.)
            x[i] = x0 - width/2.
        else:
            y[i] = y0
            x[i] = x0 - (disk_length - point)
        points.append(array([x[i], y[i]]))
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
    
def main():
    mesh = Rectangle(0, 0, 100, 100, 50, 50)
    V = VectorFunctionSpace(mesh, 'CG', 2)
    u = interpolate(Expression(('pi/314.*(50.-x[1])', 
                                'pi/314.*(x[0]-50.)')), V)
    #u = interpolate(Expression(('1.', '0.')), Vv)
    u.gather() # Required for parallel

    x = []
    if comm.Get_rank() == 0:        
        #pass
        x = zalesak(center=(50, 75), N=100)    
        #x = line(x0=39.5, y0=40, dx=1, dy=1, L=20, N=4)    
        #x = [array([0.39, 0.4]), array([0.6, 0.61]), array([0.49, 0.5])]
        #x = random_circle((50, 50), 50, N=1000)
        
    x = comm.bcast(x, root=0) # with a random function one must compute points on 0 and bcast to get the same values on all procs
        
    lp = LagrangianParticles(V)
    t0 = time.time()    
    lp.add_particles(x)
    #lp.add_particles_ring(x)
    print comm.Get_rank(), ' time ', time.time() - t0
    dt = 1.
    t = 0
    #lp.scatter()
    t0 = time.time()
    while t <= 628:
        t = t + dt
        lp.step(u, dt)  
        if t % 157 == 0:
            if lp.myrank == 0: 
                print 'Plotting at ', t
            lp.scatter()    
            lp.total_number_of_particles()
            
    lp.total_number_of_particles()
    print 'Computing time ', time.time() - t0
    show()            
        
    return lp, u

def profiler():
    import cProfile
    import pstats
    cProfile.run('main()', 'mainprofile')
    p = pstats.Stats('mainprofile')
    p.sort_stats('time').print_stats(50)
    
if __name__=='__main__':
    #profiler()
    from dolfin import *
    lp, u = main()
    mf = MeshFunction('uint', lp.mesh, 2)
    mf.set_all(0)
    for cell in lp.cellparticles.iterkeys():
        mf[cell] += 1
        
    plot(mf)
    
