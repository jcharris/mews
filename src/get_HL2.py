import numpy as np
from scipy.optimize import fsolve

# wave gauges
wgx = np.array([6, 11, 12, 13, 14, 15, 16, 17])

# bathymetry
g = 9.80665
nx = 101
x = np.linspace(6,17,nx)
h = 0*x+0.1
h[x<12]=0.4-(x[x<12]-6)/20
h[x>14]=0.1+(x[x>14]-14)/10

#import matplotlib.pyplot as plt
#plt.plot(x,-h,'.')
#plt.axis([6, 17, -0.4, 0])
#plt.show()

# wave liimts
fp = 0.4
fmin = fp/4
fmax = fp*4

# group velocities
cg_min = 0*x
T = 1/fmax
for i in range(nx):
    L = fsolve(lambda L:2*np.pi/T**2-g/L*np.tanh(h[i]*2*np.pi/L),1)[0]
    cp = L/T
    kh = 2*np.pi/L*h[i]
    cg_min[i] = cp/2*(1+kh*(1-np.tanh(kh)**2)/np.tanh(kh))

cg_max = 0*x
T = 1/fmin
for i in range(nx):
    L = fsolve(lambda L:2*np.pi/T**2-g/L*np.tanh(h[i]*2*np.pi/L),1)[0]
    cp = L/T
    kh = 2*np.pi/L*h[i]
    cg_max[i] = cp/2*(1+kh*(1-np.tanh(kh)**2)/np.tanh(kh))

# integrate
xc = (x[1:]+x[:-1])/2
dx = xc[1]-xc[0]
cg_minc = (cg_min[1:]+cg_min[:-1])/2
cg_maxc = (cg_max[1:]+cg_max[:-1])/2
nwg = len(wgx)
tmin = np.zeros((nwg,nwg))
tmax = np.zeros((nwg,nwg))
for i in range(nwg):
    for j in range(nwg):
        tmin[i,j] = dx*np.sum(1/cg_minc[np.logical_and(xc>wgx[i],xc<wgx[j])])
        tmax[i,j] = dx*np.sum(1/cg_maxc[np.logical_and(xc>wgx[i],xc<wgx[j])])

# windowing
dt = 0.1
print('H')
print(np.round(tmax/dt))
print('L')
print(np.round(tmin/dt))
