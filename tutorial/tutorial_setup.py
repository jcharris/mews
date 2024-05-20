import numpy as np
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
import pandas as pd
from scipy.optimize import fsolve

dx = 2

# time-series parameters
fs = 10
Ns = 3600
t = np.arange(Ns)/fs

# wave parameters
Hs = 0.01
Tp = 2.5
h = 0.2
gamma = 3.3
g = 9.80665

# synthetic spectra
dt = 1/fs ; df = fs/Ns ; fp = 1/Tp ; f = np.linspace(df,4*fp,int(4*fp/df))
sigma = f*0 ; sigma[f>fp]=0.09 ; sigma[f<=fp]=0.7
alpha = 5*(Hs*fp*fp/g)**2*(1-0.287*np.log(gamma))*np.pi**4
beta = np.exp(-0.5*(((f/fp)-1)/sigma)**2)
S = alpha*g**2/(2*np.pi)**4*f**(-5)*np.exp(-1.25*(f/fp)**(-4))*gamma**beta

# compute amplitude, phase, and wavelength
a = np.sqrt(2*S*df)
ph = 2*np.pi*np.random.random(len(f))
k = 0*f
for i in range(len(f)):
    k[i]=2*np.pi/fsolve(lambda L:2*np.pi*f[i]**2-g/L*np.tanh(h*2*np.pi/L),1)[0]

# compute time-series
x = 0
z0 = 0*t
for ai,fi,ki,ei in zip(a,f,k,ph):
    z0 += ai*np.cos(2*np.pi*fi*t-ki*x+ei)

x = dx
z1 = 0*t
for ai,fi,ki,ei in zip(a,f,k,ph):
    z1 += ai*np.cos(2*np.pi*fi*t-ki*x+ei)

# save synthetic data to csv file
df = pd.DataFrame(index=np.arange(Ns),columns=['t','z0','z1'])
df['t']=t
df['z0']=z0
df['z1']=z1
df.to_csv('../data/tutorial_data.csv',index=False)

# plot setup
plt.rcdefaults()
fig = plt.figure(figsize=(10,6))
gs = GridSpec(2,1, figure=fig)
ax1 = fig.add_subplot(gs[0,0])
ax2 = fig.add_subplot(gs[1,0],sharex=ax1)

ax1.plot(df['t'].values,df['z0'].values,'k')
ax1.set_ylabel('z (m)')
plt.setp(ax1.get_xticklabels(), visible=False)

ax2.plot(df['t'].values,df['z1'].values,'k')
ax2.set_ylabel('z (m)')
ax2.set_xlabel('t (s)')

ax1.grid()
ax2.grid()
ax1.set_title('x = 0m')
ax2.set_title('x = '+str(dx)+'m')
ax2.set_xlim([np.min(df['t'].values),np.max(df['t'].values)])

plt.show()

# propose reasonable values of H and L
Tmin = Tp / 4
Tmax = 4 * Tp
Lmin = fsolve(lambda L:2*np.pi/Tmin**2-g/L*np.tanh(h*2*np.pi/L),1)[0]
Lmax = fsolve(lambda L:2*np.pi/Tmax**2-g/L*np.tanh(h*2*np.pi/L),1)[0]
kmin = 2*np.pi/Lmin
kmax = 2*np.pi/Lmax
cp_min = np.sqrt(g/kmin*np.tanh(kmin*h))
cp_max = np.sqrt(g/kmax*np.tanh(kmax*h))
cg_min = cp_min/2*(1+kmin*h*(1-np.tanh(kmin*h)**2)/np.tanh(kmin*h))
cg_max = cp_max/2*(1+kmax*h*(1-np.tanh(kmax*h)**2)/np.tanh(kmax*h))
H = x/cg_max/dt
L = x/cg_min/dt
print('H: ',H)
print('L: ',L)
