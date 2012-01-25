__author__ = "Mikael Mortensen <mikaem@math.uio.no>"
__date__ = "2012-01-7"
__copyright__ = "Copyright (C) 2011 " + __author__
__license__  = "GNU GPL version 3 or any later version"
"""
This module contains functionality for Lagrangian tracking of particles 
"""
from dolfin import Point, Cell, cells, Rectangle, FunctionSpace, VectorFunctionSpace, interpolate, Expression, parameters
import ufc
from numpy import linspace, pi, zeros, array, ndarray, squeeze, load, sin, cos, arcsin, arctan, dot as ndot
from pylab import scatter, show
from copy import copy, deepcopy
import time
from mpi4py import MPI as nMPI

comm = nMPI.COMM_WORLD
parameters["form_compiler"]["optimize"]     = True
parameters["form_compiler"]["cpp_optimize"] = True

class SingleParticle:
    
    def __init__(self, x, c):
        self.position = x
        self.cell = c
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
            self.cell = None

class LP:
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
                                   
        # Allocate a list to hold all particles
        self.particles = []
        
        # Allocate some MPI-stuff
        self.num_processes = comm.Get_size()
        self.myrank = comm.Get_rank()
        self.all_processes = range(self.num_processes)
        self.other_processes = range(self.num_processes)
        self.other_processes.remove(self.myrank)
        self.my_escaped_particles = zeros(self.num_processes, dtype='I')
        self.tot_escaped_particles = zeros(self.num_processes, dtype='I')
        self.particle0 = SingleParticle(zeros(self.mesh.geometry().dim()), None)
        self.verbose = False
        self.parallel_time = 0
        self.parallel_time0 = 0
        
    def add_particles(self, list_of_particles):
        for particle in list_of_particles: # particle or array
            c = self.locate(particle)
            if not c == -1:
                if isinstance(particle, ndarray):
                    self.particles.append(SingleParticle(particle, Cell(self.mesh, c)))
                else:
                    particle.cell = Cell(self.mesh, c)
                    self.particles.append(particle)

    def step(self, u, dt):
        escaped_particles = []
        received = []
        myrank = self.myrank
        for i, particle in enumerate(self.particles):
            point = Point(*particle.position)
            cell  = particle.cell
            # Check first if particle is still inside the same cell
            if cell.intersects(point):
                pass
            else:
                found = False 
                # Check neighbor cells
                for neighbor in cells(cell): 
                    if neighbor.intersects(point):
                        particle.cell = Cell(self.mesh, neighbor.index())
                        found = True
                        break
                # Do a completely new search if not found by now
                if not found:  
                    c = self.locate(point)
                    if c == -1: # If particle is no longer on processor then it needs to be relocated on one of the other_processes
                        escaped_particles.append(i)
                    else:
                        particle.cell = Cell(self.mesh, c)

        # With MPI the particles may travel between processors.
        # Capture these traveling particles here and find new homes
        t0 = time.time()
        escaped_particles.reverse()
        list_of_escaped_particles = []
        for i in escaped_particles:
            p0 = self.particles.pop(i)
            list_of_escaped_particles.append(p0)
        self.my_escaped_particles[myrank] = len(list_of_escaped_particles)        
        # Create a list of how many particles escapes from each processor
        self.parallel_time0 += time.time() - t0
        
        #for proc in self.other_processes:
            #comm.send(self.my_escaped_particles[myrank], dest=proc)
            #self.tot_escaped_particles[proc] = comm.recv(source=proc)
        self.tot_escaped_particles = comm.allreduce(self.my_escaped_particles)
            
        self.parallel_time += time.time() - t0
        # Print for debugging etc.
        if self.verbose:
            print 'Escaped ', myrank, list_of_escaped_particles
            if self.myrank==0:
                print 'Total escaped = ', self.tot_escaped_particles
        
        # Send the escaping particles to the other processes
        # For now send to all other processes. Could send only to neighbors using, e.g., info in parallel_data().shared_vertices()
        for particle in list_of_escaped_particles:
            particle.send(self.other_processes)
            if self.verbose: print myrank, ' sending ', particle.position, ' to ', self.other_processes
                
        # Receive the particles escaping from other processors
        received = []        
        for j in self.other_processes:
            for i in range(self.tot_escaped_particles[j]):
                self.particle0.recv(j)
                received.append(deepcopy(self.particle0))
                
        # Relocate particles
        self.add_particles(received)
        
        
        # Get particle velocities and move
        for particle in self.particles:
            cell = particle.cell
            x = particle.position
            u.restrict(self.coefficients, self.element, cell, self.ufc_cell)
            self.element.evaluate_basis_all(self.basis_matrix, x, cell)
            du = ndot(self.coefficients, self.basis_matrix)
            particle.prm['velocity'] = du # Just for fun remember the velocity. Could use this for higher order schemes
            x[:] = x[:] + dt*du[:]

    def locate(self, x):
        if isinstance(x, Point):
            point = x
        elif isinstance(x, ndarray):
            point = Point(*x)
        elif isinstance(x, SingleParticle):
            point = Point(*x.position)
        return self.mesh.intersected_cell(point)
            
    def scatter(self):
        """Put all particles on processor 0 for scatter plotting"""
        my_particles = zeros(self.num_processes, dtype='I')
        all_particles = zeros(self.num_processes, dtype='I')
        my_particles[self.myrank] = len(self.particles)
        comm.Allreduce(my_particles, all_particles) 
        if self.myrank > 0: # Send all particles from processes > 0 to 0
            for p in self.particles:
                p.send(0)
        else:               # Receive on proc 0
            recv = [copy(p.position) for p in self.particles]
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
    disk_length = width + 2.*sloth_length + 2.*radius*(pi-theta)
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
    
def main():
    mesh = Rectangle(0, 0, 100, 100, 50, 50)
    V = VectorFunctionSpace(mesh, 'CG', 2)
    u = interpolate(Expression(('pi/314.*(50.-x[1])', 
                                'pi/314.*(x[0]-50.)')), V)
    #u = interpolate(Expression(('1.', '0.')), Vv)
    u.gather() # Required for parallel
             
    x = zalesak(center=(50, 75), N=100)    
    #x = line(x0=39.5, y0=40, dx=1, dy=1, L=20, N=4)    
    #x = [array([0.39, 0.4]), array([0.6, 0.61]), array([0.49, 0.5])]
    lp = LP(V)
    lp.add_particles(x)
    dt = 0.5
    t = 0
    lp.scatter()
    t0 = time.time()
    while t <= 628:
        t = t + dt
        lp.step(u, dt)  
        if t % 157 == 0:
            if lp.myrank == 0: 
                print 'Plotting at ', t
            lp.scatter()
    #if lp.myrank == 0:
        
    print 'Computing time = ', time.time() - t0
        
    print "mpi time ", lp.parallel_time
    print "mpi time0 ", lp.parallel_time0

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
    for p in lp.particles:
        mf[p.cell.index()] += 1
        
    #plot(mf)
    #pointsources = []
    #for p in lp.particles:
        #pointsources.append(PointSource(, -1))
    
