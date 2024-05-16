import numpy as np
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
from scipy.optimize import fsolve

# https://www.researchgate.net/publication/259138192_Uncertainties_in_the_design_of_support_structures_and_foundations_for_offshore_wind_turbines

# https://pureadmin.qub.ac.uk/ws/portalfiles/portal/240362382/STG_Hydro_2017_Monopiles.pdf

fig = plt.figure(figsize=(8,4))
gs = GridSpec(1,3,figure=fig,width_ratios=[5,1,5])
ax1 = fig.add_subplot(gs[0,0])

kh = np.linspace(0.01,10,100)
ax1.loglog(kh*np.tanh(kh)/4/np.pi/np.pi,
           0.88/4/np.pi/np.pi*np.tanh(kh)*np.tanh(0.89*kh),'k') # Miche

ax1.loglog(kh*np.tanh(kh)/4/np.pi/np.pi,
           0.88/4/np.pi/np.pi*np.tanh(kh)*np.tanh(0.89*kh)/4,'k:')

Dm = np.array([25.0, 25.0, 25.0, 20.8, 20.8,
               16.7, 16.7, 16.7, 16.7, 16.7])/1000
Hs = np.array([68.3, 75.5, 85.9, 75.3, 80.4,
               66.1, 68.0, 66.0, 63.7, 54.5])/1000
Tp = np.array([0.845, 0.930, 1.014, 1.026, 1.080,
               1.028, 1.097, 1.159, 1.215, 1.242])
h = 0.7;
g = 9.81;

L = np.array([fsolve(lambda L:2*np.pi/T**2-g/L*np.tanh(h*2*np.pi/L),1)[0]
              for T in Tp]) # linear dispersion

ax1.loglog(h/g/Tp/Tp,Hs/g/Tp/Tp,'.')

ax1.loglog(np.tanh(np.pi/10)/np.pi/40*np.ones((2,)),[0.00005,0.00184],'--k')
ax1.loglog(np.tanh(np.pi)/np.pi/4*np.ones((2,)),[0.00005,0.022],'--k')
ax1.text(.001,6e-5,'SHALLOW',ha='center',va='center',fontsize=8)
ax1.text(.13,6e-5,'DEEP',ha='center',va='center',fontsize=8)
ax1.text(.01,.01,'BREAKING',ha='center',va='center',fontsize=8,rotation=45)

ax1.set_xlim([0.0005,0.2])
ax1.set_ylim([0.00005,0.05])
ax1.set_xlabel(r'$h/(gT^2)$')
ax1.set_ylabel(r'$H/(gT^2)$')

ax2 = fig.add_subplot(gs[0,2])

ax2.loglog(np.pi*Dm/L,Hs/Dm,'.')

ax2.loglog([0.01,10],[1/.01*np.pi*.14,1/10*np.pi*.14],'k')

ax2.set_xlim([0.01,10])
ax2.set_ylim([0.01,100])
ax2.set_xlabel(r'$\pi D / \lambda$')
ax2.set_ylabel(r'$H / D$')

ax2.text(1,3,'DEEP WATER\n BREAKING',ha='center',va='center',fontsize=8)
ax2.text(.02,10,'LARGE\n DRAG',ha='center',va='center',fontsize=8)
ax2.text(.02,.1,'LARGE\n INERTIA',ha='center',va='center',fontsize=8)
ax2.text(2,.03,'DIFFRACTION\n REGION',ha='center',va='center',fontsize=8)

ax1.set_title('(a)',y=-0.3)
ax2.set_title('(b)',y=-0.3)
fig.set_tight_layout(True)
plt.savefig('../../figs/wsi_regimes.png')
plt.show()
