

from scipy import *
from numpy import fft 
from scipy.interpolate import splrep, splev


def integrate_against_gauss(f, i, scale, dt):
  sum = 0 
  width = 10
  l = len(f)/30  
  for k in range(i-l, i+l): 
    if k >= len(f): k = k - len(f)
    if k < 0: k = k + len(f)
    sum += f[k]*exp(-0.001*(k-i)**2)  
  return sum/scale

def smooth_flow(a, b, c, dt, m):
  t = arange(0, c, dt)
  f = t*0  

  for i in range(len(t)): 
    ti = t[i]/c 
    if ti < a/(a+b):
      f[i] = b*sin(3.14*ti*(a+b)/(a))
    else: 
      tii = ti-a/((a+b))
      f[i] = -a*sin(3.14*tii*(a+b)/(b))

  h = t*0  
  h[:] = 1 

  hh = integrate_against_gauss(h, 100, 1, dt)

  FF = zeros(len(t)*m)
  for i in range(m): 
    FF[i*len(t):len(t)*(i+1)] = f[:]  

  g = FF*0  
  for i in range(m*len(t)): 
    g[i] = integrate_against_gauss(FF, i, hh, dt)

  g_sum = sum(g)/len(g)
  g -= g_sum 

  print "g_max ", max(g), " g_min ", min(g), " sum(g) ", sum(g), "g[0] ", g[0], "g[-1] ", g[-1] 

  g = g*260/max(g)
  
  return g


def create_spline(g, m, c, dt):
  tt = arange(0, c*m, dt)
  spline_func = splrep(tt, g)
  return spline_func


if __name__ == '__main__': 
  import cPickle
  a = 2.5     
  b = 10.0 
  c = 0.7
  dt = 0.001
  m = 2 
  smooth_func = smooth_flow(a, b, c, dt, m)
  spline_func = create_spline(smooth_func, m, c, dt)

  #change resolution  
  tt = arange(0, c*m, dt/30)
  ff = 0*tt
  for i in range(len(tt)): 
    ff[i] =  splev(tt[i], spline_func)

  import pylab
  pylab.plot(tt,ff)
  #pylab.hold(True)
  #pylab.show()
  f = open('csf_pressure.spl')
  cPickle.dump(spline_func, f)


