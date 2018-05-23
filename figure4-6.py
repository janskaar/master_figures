import h5py, os
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

def clear_axis(ax):
    ax.spines['right'].set_visible(False)
    # ax.spines['bottom'].set_visible(False)
    ax.spines['top'].set_visible(False)
    # ax.set_xticks([])
    # ax.set_yticks([])
    # ax.spines['right'].set_visible(False)



low_g_dirs = [os.path.join('/home/jan-eirik/master-data/hybrid_brunel33/'),
              os.path.join('/home/jan-eirik/master-data/hybrid_brunel34/'),
              os.path.join('/home/jan-eirik/master-data/hybrid_brunel35/')]

apr_lfps = []
sim_lfps = []
exhists = []
inhists = []
hists = []

for path in low_g_dirs:
    sim = os.listdir(os.path.join(path, 'hybrid_output'))
    if len(sim) != 1:
        raise ValueError('WRONG DIRECTORY')
    simpath = os.path.join('hybrid_output', sim[0], sim[0]+'LFPsum.h5')
    with h5py.File(os.path.join(path, simpath)) as f:
        lfp = f['data'][:,100:]
        lfp -= np.mean(lfp, axis=1)[:,None]
        sim_lfps.append(lfp)
    aprpath = os.path.join('nest_output', sim[0], 'LFP_firing_rate.h5')
    with h5py.File(os.path.join(path, aprpath)) as f:
        lfp = f['data'][:,100:]
        lfp -= np.mean(lfp, axis=1)[:,None]
        apr_lfps.append(lfp)
        hists.append(f['ex_hist'][100:] + f['in_hist'][100:])
        inhists.append(f['in_hist'][:])
        exhists.append(f['ex_hist'][:])


fig, ax = plt.subplots(3, figsize=[5,3], sharex=True)
fig.subplots_adjust(hspace=1.0, bottom=0.2)

for i in range(3):
    clear_axis(ax[i])
    ax[i].plot(np.arange(100, 200), sim_lfps[i][4][100:200], color='black', linewidth=0.8)
    ax[i].plot(np.arange(100, 200), apr_lfps[i][4][101:201], color='orange', linewidth=0.8)

ax[0].set_title('$g$ = 3.2', fontdict={'fontsize': 8})
ax[1].set_title('$g$ = 3.4', fontdict={'fontsize': 8})
ax[2].set_title('$g$ = 3.6', fontdict={'fontsize': 8})

ax[0].set_yticks([0, 1])
ax[1].set_yticks([-0.25, 0.25])
ax[2].set_yticks([-0.25, 0.25])

ax[0].set_yticklabels([0, 1], fontdict={'fontsize': 8})
ax[1].set_yticklabels([-0.25, 0.25], fontdict={'fontsize': 8})
ax[2].set_yticklabels([-0.25, 0.25], fontdict={'fontsize': 8})

ax[0].set_xticks([100, 150, 200])
ax[1].set_xticks([100, 150, 200])
ax[2].set_xticks([100, 150, 200])
ax[2].set_xticklabels([100, 150, 200], fontdict={'fontsize': 8})
ax[2].set_xlabel('$t$ (ms)', fontdict={'fontsize': 8})

ax[0].set_ylabel('mV', fontdict={'fontsize': 8}, rotation=0, labelpad=18)
ax[1].set_ylabel('mV', fontdict={'fontsize': 8}, rotation=0, labelpad=3.0)
ax[2].set_ylabel('mV', fontdict={'fontsize': 8}, rotation=0, labelpad=2.0)



fig.savefig('../plots/lfp_apprx_fails', dpi=200)
