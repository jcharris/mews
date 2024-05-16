import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import patches
from matplotlib.gridspec import GridSpec

fig = plt.figure(figsize=(8,4))
gs = GridSpec(2, 5, figure=fig, width_ratios=[1,3,3,3,1])
ax1 = fig.add_subplot(gs[0, 1:4])

ax1.plot([0,6,12,14,17,18.95,37.7],[-0.4,-0.4,-0.1,-0.1,-0.4,-0.4,0.35],'k')
ax1.plot([2,28.95],[0,0],'b')
x=np.linspace(0,2,11)
ax1.plot(x,-0.02*np.sin(x*np.pi),'b')

ax1.plot([6,11,12,13,14,15,16,17],np.array([0,0,0,0,0,0,0,0])+0.05,
         ls="", marker="v", ms=10, color='k',clip_on=False)

IN = 'WG1'
OUT = 'WG8'
file = '../data/sln.dat'
df = pd.read_csv(file, delimiter='\s+', header=None, skiprows=2)
df = df.rename(columns = {0:'t',
                          1:'WG1',2:'WG2',3:'WG3',4:'WG4',
                          5:'WG5',6:'WG6',7:'WG7',8:'WG8'})

ax2 = fig.add_subplot(gs[1, 0:2])
ax2.plot(np.array(df['t'])-120,np.array(df[IN]),'b')
ax2.set_xlim([0,7.5])
ax2.set_ylim([-3,3])

ax2.spines['left'].set_position('zero')
ax2.spines['right'].set_visible(False)
ax2.spines['bottom'].set_position('zero')
ax2.spines['top'].set_visible(False)
ax2.xaxis.set_ticks_position('bottom')
ax2.yaxis.set_ticks_position('left')
ax2.set_xticks([])
ax2.set_yticks([])
ax2.plot((1), (0), ls="", marker=">", ms=5, color="k",
        transform=ax2.get_yaxis_transform(), clip_on=False)
ax2.plot((0), (1), ls="", marker="^", ms=5, color="k",
        transform=ax2.get_xaxis_transform(), clip_on=False)

ax3 = fig.add_subplot(gs[1, 3:5])
ax3.plot(np.array(df['t'])-120,np.array(df[OUT]),'b')
ax3.set_xlim([0,7.5])
ax3.set_ylim([-3,3])

ax3.spines['left'].set_position('zero')
ax3.spines['right'].set_visible(False)
ax3.spines['bottom'].set_position('zero')
ax3.spines['top'].set_visible(False)
ax3.xaxis.set_ticks_position('bottom')
ax3.yaxis.set_ticks_position('left')
ax3.set_xticks([])
ax3.set_yticks([])
ax3.plot((1), (0), ls="", marker=">", ms=5, color="k",
        transform=ax3.get_yaxis_transform(), clip_on=False)
ax3.plot((0), (1), ls="", marker="^", ms=5, color="k",
        transform=ax3.get_xaxis_transform(), clip_on=False)

ax1.set_xlim([0,35])
ax1.set_ylim([-0.4,+0.2])
ax1.set_xlabel('x (m)')
ax1.set_ylabel('z (m)')
ax1.text(22, 0, 'MWL', ha='left', va='bottom')
ax2.text(.01, .01, 'WG 1', ha='left', va='bottom', transform=ax2.transAxes)
ax3.text(.01, .01, 'WG 8', ha='left', va='bottom', transform=ax3.transAxes)

ax2.text(.99, .6, 't', ha='center', va='bottom', transform=ax2.transAxes)
ax3.text(.99, .6, 't', ha='center', va='bottom', transform=ax3.transAxes)
ax2.text(-.05, .8, r'$\eta$', ha='center', va='bottom', transform=ax2.transAxes)
ax3.text(-.05, .8, r'$\eta$', ha='center', va='bottom', transform=ax3.transAxes)

xyA = [6,0]
xyB = [0,0]
arrow = patches.ConnectionPatch(
    xyA,xyB,coordsA=ax1.transData,coordsB=ax2.transData,
    color="black",arrowstyle="-|>",mutation_scale=10,linewidth=1)
fig.patches.append(arrow)

xyA = [17,0]
xyB = [ 0,0]
arrow = patches.ConnectionPatch(
    xyA,xyB,coordsA=ax1.transData,coordsB=ax3.transData,
    color="black",arrowstyle="-|>",mutation_scale=10,linewidth=1)
fig.patches.append(arrow)

plt.savefig('../../figs/bar.png')

plt.show()
