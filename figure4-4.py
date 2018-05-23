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

top_dir = os.path.join('../data/hybridlfp_poplfp/hybrid_brunel54')

# low_g_dirs = [os.path.join('/home/jan-eirik/master-data/hybrid_brunel33/'),
#               os.path.join('/home/jan-eirik/master-data/hybrid_brunel34/'),
#               os.path.join('/home/jan-eirik/master-data/hybrid_brunel35/')]



with h5py.File(os.path.join(top_dir, 'nest_output/e885685a5cd76be067e58e8a42a3ea74/LFP_firing_rate.h5')) as f:
    poplfp = f['data'][:,100:]
    ex_hist = f['ex_hist'][100:]
    in_hist = f['in_hist'][100:]
    avg_frate = f['average_firing_rate'][:]

with h5py.File(os.path.join(top_dir, 'hybrid_output/e885685a5cd76be067e58e8a42a3ea74/e885685a5cd76be067e58e8a42a3ea74LFPsum.h5')) as f:
    hybridlfp = f['data'][:,100:]

with h5py.File(os.path.join('../data', 'kernel/', 'L4E_53rpy1_cut_L4I_oi26rbc1_new.h5')) as f:
    exkernels = f['ex'][:]
    inkernels = f['in'][:]


apr_lfps = []
sim_lfps = []
hists = []

# for path in low_g_dirs:
#     sim = os.listdir(os.path.join(path, 'hybrid_output'))
#     if len(sim) != 1:
#         raise ValueError('WRONG DIRECTORY')
#     simpath = os.path.join('hybrid_output', sim[0], sim[0]+'LFPsum.h5')
#     with h5py.File(os.path.join(path, simpath)) as f:
#         lfp = f['data'][:,100:]
#         lfp -= np.mean(lfp, axis=1)[:,None]
#         sim_lfps.append(lfp)
#     aprpath = os.path.join('nest_output', sim[0], 'LFP_firing_rate.h5')
#     with h5py.File(os.path.join(path, aprpath)) as f:
#         lfp = f['data'][:,100:]
#         lfp -= np.mean(lfp, axis=1)[:,None]
#         apr_lfps.append(lfp)
#         hists.append(f['ex_hist'][100:] + f['in_hist'][100:])


poplfp -= np.mean(poplfp, axis=1)[:,None]
hybridlfp -= np.mean(hybridlfp, axis=1)[:,None]

gs = gridspec.GridSpec(7, 2, wspace=0.3, hspace=0.3)

ax1 = plt.subplot(gs[0, 0])
clear_axis(ax1)
ax1.plot(np.arange(100, 301), ex_hist, linewidth=0.8, color='orange')
ax1.set_xlim(100, 300)
ax1.set_yticks([np.mean(ex_hist[100:])])
ax1.set_xticks([100, 200, 300])
ax1.set_title('excitatory spikes', fontdict={'fontsize': 9})
ax1.set_xlabel('$t$ (ms)', fontdict={'fontsize': 8})
ax1.tick_params(axis='both', which='both', labelsize=8)

ax2 = plt.subplot(gs[0, 1])
clear_axis(ax2)
ax2.plot(np.arange(100, 301), in_hist, linewidth=0.8, color='royalblue')
ax2.set_xlim(100, 300)
ax2.set_yticks([int(np.mean(in_hist[100:]))])
ax2.set_xticks([100, 200, 300])
ax2.set_title('inhibitory spikes', fontdict={'fontsize': 9})
ax2.set_xlabel('$t$ (ms)', fontdict={'fontsize': 8})
ax2.tick_params(axis='both', which='both', labelsize=8)

ax3 = plt.subplot(gs[2:, 0])
clear_axis(ax3)
ax3.spines['bottom'].set_visible(True)
for i in range(6):
    ax3.plot(np.arange(101, 302), hybridlfp[i]*6 + 5 -i, color='black', linewidth=0.7)
    ax3.plot(np.arange(100, 301), poplfp[i]*6 + 5 - i, color='orange', linewidth=0.7)
ax3.set_xlim(100,300)
ax3.set_yticks([hybridlfp[i,0]*6 + 5 - i for i in range(6)][::-1])
ax3.set_yticklabels(['ch. %d'%(i+1) for i in range(6)][::-1])
ax3.set_xticks([100, 200, 300])
ax3.tick_params(axis='y', which='both', length=0)
ax3.set_title('predicted LFPs', fontdict={'fontsize': 9})
ax3.set_xlabel('$t$ (ms)', fontdict={'fontsize': 8})
ax3.tick_params(axis='both', which='both', labelsize=8)
ax3.plot([296,296], [2.25, 2.85], color='black', linewidth=1.5)
ax3.text(298, 2.47, '0.1 mV', fontdict={'fontsize': 8})

ax4 = plt.subplot(gs[2:, 1])
clear_axis(ax4)
ax4.spines['bottom'].set_visible(True)
maxkernel = np.max([np.max(exkernels), np.max(inkernels)*4])
for i in range(6):
    ax4.plot(np.arange(-100, 100), exkernels[i]/maxkernel + 10 - 2*i, color='orange', linewidth=0.7)
    ax4.plot(np.arange(-100, 100), inkernels[i]*4/maxkernel + 10 - 2*i, color='royalblue', linewidth=0.7)

ax4.plot([45,45], [4.8, 5.514], color='black', linewidth=1.5)
ax4.text(47, 5.0, '0.1 $\mathrm{\mu V}$', fontdict={'fontsize': 8})
ax4.set_xlim(-100, 100)
ax4.set_yticks([2*i for i in range(6)])
ax4.set_yticklabels(['ch. %d'%(i+1) for i in range(6)][::-1])
ax4.set_xticks([-100, 0, 100])
ax4.tick_params(axis='y', which='both', length=0)
ax4.set_title('kernels', fontdict={'fontsize': 9})
ax4.tick_params(axis='both', which='both', labelsize=8)
ax4.set_xlabel('$ \\tau $ (ms)', fontdict={'fontsize': 8})

# plt.show()
# plt.gcf().savefig('../plots/lfp_kernels', dpi=200)

fig2, axes2 = plt.subplots(ncols=2, nrows=6, sharex=True, sharey='row', figsize=[5,8])
fig2.subplots_adjust(left=0.2)

combined_kernel1 = exkernels*4/5 + inkernels*4*1/5
combined_kernel2 = exkernels*4/5 + inkernels*5.4*1/5

for ax in axes2.flatten():
    ax.tick_params(axis='both', which='both', labelsize=10)

axes2[0,0].set_yticks([0,-0.05])
axes2[-1,0].set_xlabel('$ \\tau $ (ms)', fontdict={'fontsize': 10})
axes2[-1,1].set_xlabel('$ \\tau $ (ms)', fontdict={'fontsize': 10})
axes2[0,0].set_title('$g = 4.0$', fontdict={'fontsize': 10})
axes2[0,1].set_title('$g = 5.4$', fontdict={'fontsize': 10})

axes2[-1,0].set_xticks([100,150])
axes2[-1,1].set_xticks([100,150])

for i, ax in enumerate(axes2[:,0]):
    ax.set_ylabel('ch. %d'%(i+1), rotation=0, fontdict={'fontsize': 10})
    ax.set_yticks([])
    ax.yaxis.set_label_coords(-0.2, 0.5)

for i in range(6):
    axes2[i,0].plot(np.arange(100,150), combined_kernel1[i,100:150]*1000, 'black')
    axes2[i,1].plot(np.arange(100,150), combined_kernel2[i,100:150]*1000, 'black')









1
