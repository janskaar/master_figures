import os, h5py
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

top_dir = os.path.join('../data/brunel_four_states/')
sims = os.listdir(os.path.join(top_dir, 'nest_output'))
sims.sort()
sims = [sims[1], sims[0], sims[2], sims[3]]


rasters = []
frates = []

tstart = 400.
tstop = 500.
n_neurons = 100



for sim in sims:
    espikes = np.loadtxt(os.path.join(top_dir, 'nest_output/', sim + '/', 'brunel-py-EX-12502-0.gdf'))
    espikes = espikes[np.logical_and(espikes[:,1] >= tstart, espikes[:,1] <= tstop)]
    espikes = espikes[np.argsort(espikes[:,0])]

    ispikes = np.loadtxt(os.path.join(top_dir, 'nest_output/', sim + '/', 'brunel-py-IN-12503-0.gdf'))
    ispikes = ispikes[np.logical_and(ispikes[:,1] >= tstart, ispikes[:,1] <= tstop)]
    ispikes = ispikes[np.argsort(ispikes[:,0])]

    spikes = np.concatenate((espikes, ispikes), axis=0)
    frate, bins = np.histogram(spikes[:,1], np.arange(400, 500.1, 0.1)-0.05)
    uq = np.unique(spikes[:,0])
    nids = np.random.choice(uq, size=n_neurons, replace=False)
    chosen = np.where(np.isin(spikes[:,0], nids))
    uq, index = np.unique(spikes[chosen][:,0], return_index=True)
    rasters.append(np.split(spikes[chosen], index[1:]))
    frates.append(frate)

gs = gridspec.GridSpec(9,9)
gs.update(hspace=0.00, wspace=0.00)

fig = plt.gcf()
fig.set_size_inches([7,7])
ax1 = plt.subplot(gs[:3,:4])
ax1.set_xticks([])
ax2 = plt.subplot(gs[3,:4])
for i, r in enumerate(rasters[0]):
    ax1.scatter(r[:,1], np.ones_like(r[:,1])*i, marker='|', color='black', s=5.0, linewidth=0.8)
ax2.bar(bins[1:] - 0.05, height=frates[0], width=0.15, color='black')
ax1.set_yticks([])
ax2.set_yticks([0,6000])
ax2.set_xticks([400, 425, 450])
ax1.set_xlim(399, 451)
ax2.set_xlim(399, 451)
# ax2.set_xlabel('$t$ (ms)')

ax3 = plt.subplot(gs[:3,5:])
ax3.set_xticks([])
ax4 = plt.subplot(gs[3,5:])
for i, r in enumerate(rasters[1]):
    ax3.scatter(r[:,1], np.ones_like(r[:,1])*i, marker='|', color='black', s=5.0, linewidth=0.8)
ax4.bar(bins[1:] - 0.05, height=frates[1], width=0.1, color='black')
ax3.set_yticks([])
ax4.set_yticks([0,400])
ax4.set_xticks([400, 450, 500])
ax3.set_xlim(tstart-1, tstop+1)
ax4.set_xlim(tstart-1, tstop+1)
# ax4.set_xlabel('$t$ (ms)')

ax5 = plt.subplot(gs[5:8,:4])
ax5.set_xticks([])
ax6 = plt.subplot(gs[8,:4])
for i, r in enumerate(rasters[2]):
    ax5.scatter(r[:,1], np.ones_like(r[:,1])*i, marker='|', color='black', s=5.0, linewidth=0.8)
ax6.bar(bins[1:] - 0.05, height=frates[2], width=0.1, color='black')
ax5.set_yticks([])
ax6.set_yticks([0,80])
ax6.set_xticks([400, 450, 500])
ax5.set_xlim(tstart-1, tstop+1)
ax6.set_xlim(tstart-1, tstop+1)
ax6.set_xlabel('$t$ (ms)')

ax7 = plt.subplot(gs[5:8,5:])
ax7.set_xticks([])
ax8 = plt.subplot(gs[8,5:])
for i, r in enumerate(rasters[3]):
    ax7.scatter(r[:,1], np.ones_like(r[:,1])*i, marker='|', color='black', s=5.0, linewidth=0.8)
ax8.bar(bins[1:] - 0.05, height=frates[3], width=0.1, color='black')
ax7.set_yticks([])
ax8.set_yticks([0,300])
ax8.set_xticks([400, 450, 500])
ax7.set_xlim(tstart-1, tstop+1)
ax8.set_xlim(tstart-1, tstop+1)
ax8.set_xlabel('$t$ (ms)')

titlefont = {'fontsize': 10}
ax1.set_title('$\eta$ = 2.0, g = 3.0', fontdict=titlefont)
ax3.set_title('$\eta$ = 4.0, g = 6.0', fontdict=titlefont)
ax5.set_title('$\eta$ = 0.9, g = 4.5', fontdict=titlefont)
ax7.set_title('$\eta$ = 2.0, g = 5.0', fontdict=titlefont)

# fig.show()
fig.savefig('brunel_four_states2', dpi=400)
