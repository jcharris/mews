import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec

fig = plt.figure(figsize=(10,4))
gs = GridSpec(1,3,figure=fig,width_ratios=[2,1,2])
ax1 = fig.add_subplot(gs[0,0])

ax1.set_xlim([-1,5])
ax1.set_ylim([-1,5])

ax1.text(5, -0.5, 'x', ha='right', va='top')
ax1.text(-0.5, 5, 't', ha='right', va='top')

ax1.text(4.5, 0.5, '...', ha='right', va='bottom')
ax1.text(4.5, -0.5, '...', ha='right', va='bottom')

ax1.text(3, -0.5, 'ICs', ha='center', va='top')
ax1.text(-0.5, 2, 'BCs', ha='right', va='center')

ax1.annotate("", xy=(0.5, 0.5), xytext=(-0.5, -0.5),
             arrowprops=dict(arrowstyle="->"))
ax1.annotate("", xy=(0.5, 0.5), xytext=(+0.5, -0.5),
             arrowprops=dict(arrowstyle="->"))
ax1.annotate("", xy=(0.5, 0.5), xytext=(+1.5, -0.5),
             arrowprops=dict(arrowstyle="->"))
ax1.annotate("", xy=(0.5, 0.5), xytext=(+2.5, -0.5),
             arrowprops=dict(arrowstyle="->"))

ax1.plot([-0.5,0.5,1.5,2.5,3.5],[-0.5,-0.5,-0.5,-0.5,-0.5],'ok',ms=5)
ax1.plot([0.5,1.5,2.5,3.5],[0.5,0.5,0.5,0.5],'ok',ms=5,fillstyle='none')
ax1.plot([-0.5,-0.5,-0.5,-0.5],[0.5,1.5,2.5,3.5],'ok',ms=5)

ax1.spines['left'].set_position('zero')
ax1.spines['right'].set_visible(False)
ax1.spines['bottom'].set_position('zero')
ax1.spines['top'].set_visible(False)
ax1.xaxis.set_ticks_position('bottom')
ax1.yaxis.set_ticks_position('left')
ax1.set_xticks([])
ax1.set_yticks([])
ax1.plot((1), (0), ls="", marker=">", ms=5, color="k",
        transform=ax1.get_yaxis_transform(), clip_on=False)
ax1.plot((0), (1), ls="", marker="^", ms=5, color="k",
        transform=ax1.get_xaxis_transform(), clip_on=False)


ax2 = fig.add_subplot(gs[0,2])

ax2.set_xlim([-1,5])
ax2.set_ylim([-1,5])

ax2.annotate('H', xy=(5, 4), xytext=(5.5, 4),
             ha='left', va='center',
             bbox=dict(boxstyle='square', fc='white', color='k'),
             arrowprops=dict(arrowstyle='-[, widthB=2, lengthB=1.5',
                             lw=2.0, color='k'))

ax2.annotate('L', xy=(0, 1.5), xytext=(-0.5, 1.5),
             ha='right', va='center',
             bbox=dict(boxstyle='square', fc='white', color='k'),
             arrowprops=dict(arrowstyle='-[, widthB=4, lengthB=1.5',
                             lw=2.0, color='k'))

ax2.text(5, -0.5, 'x', ha='right', va='top')
ax2.text(-0.5, 5, 't', ha='right', va='top')

ax2.plot([0.5,0.5,0.5],[0.5,1.5,2.5],'ok',ms=5)
ax2.plot([4.5,4.5],[3.5,4.5],'ok',ms=5,fillstyle='none')
ax2.plot([4.5,4.5,4.5],[0.5,1.5,2.5],'ok',ms=5)

ax2.annotate("", xy=(3.5, 3.325), xytext=(1.5, 2),
             arrowprops=dict(arrowstyle="->"))

ax2.plot([0.5,4.5],[0.5,3.5],'--k')
ax2.plot([0.5,4.5],[2.5,4.5],'--k')
ax2.text(1.5,3.5,r'$c_{g,max}$')
ax2.text(2,1.25,r'$c_{g,min}$')

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

ax1.set_title('(a)',y=-0.1)
ax2.set_title('(b)',y=-0.1)

plt.savefig('../../figs/definition.png')

plt.show()
