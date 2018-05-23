import os, h5py
import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import periodogram, welch
from ConvNet.input_functions import read_ids_parameters
import tensorflow as tf
import matplotlib.gridspec as gridspec

def clear_axis(ax):
    ax.spines['right'].set_visible(False)
    ax.spines['top'].set_visible(False)

colors = plt.cm.tab10.colors
## Load IDS and parameters
top_dir = os.path.join('C:\\Users\\JanEirik\\Documents\\TensorFlow\\data\\hybrid_brunel47\\nest_output')
savedir = os.path.join('..\\plots\\')

ids1, labels1 = read_ids_parameters(os.path.join(top_dir, 'info.txt'))
ids2, labels2 = read_ids_parameters(os.path.join(top_dir, 'info1.txt'))
ids3, labels3 = read_ids_parameters(os.path.join(top_dir, 'info2.txt'))

labels = np.array(labels1 + labels2 + labels3)
ids = np.array(ids1 + ids2 + ids3, dtype=str)


## Sort them into 8x8x8x6 matrix, (eta, g, j, sim number)
etas = np.array([1.6, 1.8, 2.0, 2.2, 2.4, 2.6, 2.8, 3.0], dtype=np.float32)
gs = np.array([4.0, 4.2, 4.4, 4.6, 4.8, 5.0, 5.2, 5.4], dtype=np.float32)
js = np.array([0.06, 0.08, 0.1, 0.12, 0.14, 0.16, 0.18, 0.2], dtype=np.float32)


idarr = np.chararray([8, 8, 8, 12], itemsize=32, unicode=True)
filled = np.zeros([8,8,8], dtype=np.int64)

## Fill IDs
for j, i in enumerate(ids):
    etapos = np.argwhere(etas == labels[j][0])
    gpos = np.argwhere(gs == labels[j][1])
    jpos = np.argwhere(js == labels[j][2])
    idarr[etapos, gpos, jpos, filled[etapos, gpos, jpos]] = i
    filled[etapos, gpos, jpos] += 1

## Get LFPs and calculate average PSDs for each parameter combination
psd_arr = None
avg_psd_arr = None
neweta = None
lfp_arr = np.zeros([8, 8, 8, 12, 6, 1852])
lfp_filled = np.zeros([8, 8, 8], dtype=np.int64)
for (eta, g, j), _ in np.ndenumerate(idarr[:,:,:,0]):
    if neweta != eta:
        print(eta)
    ids = idarr[eta,g,j]
    avg_psd = None
    for i in ids:
        psd = []
        with h5py.File(os.path.join(top_dir, i, 'LFP_firing_rate.h5')) as f:
            lfp = f['data'][:]
            lfp = lfp[:,150:] - np.mean(lfp[:,150:], axis=1)[:,None]
        lfp_arr[eta, g, j, lfp_filled[eta, g, j]] = lfp
        for ch in lfp:
            freqs, psd_ch = welch(ch, fs=1000, nperseg=256, detrend=False)
            psd.append(psd_ch)
        psd = np.array(psd)
        if psd_arr is None:
            psd_arr = np.zeros([8,8,8,12]+list(np.shape(psd)))
        psd_arr[eta, g, j, lfp_filled[eta, g, j]] = psd
        lfp_filled[eta, g, j] += 1
    neweta = eta

avg_psd_arr = np.mean(psd_arr, axis=3)


## PARAMTER g EDGE PLOTS
fig1, ax1 = plt.subplots(ncols=3,nrows=7, figsize=[7,9], sharex=True, sharey=True, gridspec_kw={'height_ratios': [3,3,3,1,3,3,3]})
for ax in ax1.flatten():
    clear_axis(ax)
    ax.tick_params(axis='both', which='both', labelsize=8)
for ax in ax1[3,:]:
    ax.axis('off')

ax1[0,0].set_xlim(4,200)
# ax1[0,0].set_ylim(1e-7, 1e-1)
ax1.flatten()[0].set_title('J = 0.06', fontdict={'fontsize': 8})
ax1.flatten()[1].set_title('J = 0.1', fontdict={'fontsize': 8})
ax1.flatten()[2].set_title('J = 0.2', fontdict={'fontsize': 8})

ax1.flatten()[2].set_ylabel('$\eta$ = 3.0', rotation=0, fontdict={'fontsize': 8})
ax1.flatten()[2].yaxis.set_label_coords(1.1, 0.5)
ax1.flatten()[5].set_ylabel('$\eta$ = 2.2', rotation=0, fontdict={'fontsize': 8})
ax1.flatten()[5].yaxis.set_label_coords(1.1, 0.5)
ax1.flatten()[8].set_ylabel('$\eta$ = 1.6', rotation=0, fontdict={'fontsize': 8})
ax1.flatten()[8].yaxis.set_label_coords(1.1, 0.5)

# ax1.flatten()[6].set_xlabel('Hz', fontdict={'fontsize': 8})
# ax1.flatten()[7].set_xlabel('Hz', fontdict={'fontsize': 8})
# ax1.flatten()[8].set_xlabel('Hz', fontdict={'fontsize': 8})

ax1.flatten()[0].set_ylabel('$\mathrm{mV^2/Hz}$', fontdict={'fontsize': 8})
ax1.flatten()[3].set_ylabel('$\mathrm{mV^2/Hz}$', fontdict={'fontsize': 8})
ax1.flatten()[6].set_ylabel('$\mathrm{mV^2/Hz}$', fontdict={'fontsize': 8})

for i in range(8):
    ax1.flatten()[0].loglog(freqs[:50], avg_psd_arr[7,i,0,4,:50], color=plt.cm.Greys(7/14 + i/14), linewidth=0.8)
    ax1.flatten()[1].loglog(freqs[:50], avg_psd_arr[7,i,2,4,:50], color=plt.cm.Greys(7/14 + i/14), linewidth=0.8)
    ax1.flatten()[2].loglog(freqs[:50], avg_psd_arr[7,i,7,4,:50], color=plt.cm.Greys(7/14 + i/14), linewidth=0.8)
    ax1.flatten()[3].loglog(freqs[:50], avg_psd_arr[3,i,0,4,:50], color=plt.cm.Greys(7/14 + i/14), linewidth=0.8)
    ax1.flatten()[4].loglog(freqs[:50], avg_psd_arr[3,i,2,4,:50], color=plt.cm.Greys(7/14 + i/14), linewidth=0.8)
    ax1.flatten()[5].loglog(freqs[:50], avg_psd_arr[3,i,7,4,:50], color=plt.cm.Greys(7/14 + i/14), linewidth=0.8)
    ax1.flatten()[6].loglog(freqs[:50], avg_psd_arr[0,i,0,4,:50], color=plt.cm.Greys(7/14 + i/14), linewidth=0.8)
    ax1.flatten()[7].loglog(freqs[:50], avg_psd_arr[0,i,2,4,:50], color=plt.cm.Greys(7/14 + i/14), linewidth=0.8)
    ax1.flatten()[8].loglog(freqs[:50], avg_psd_arr[0,i,7,4,:50], color=plt.cm.Greys(7/14 + i/14), linewidth=0.8)

ax1.flatten()[12].set_title('$g$ = 4.0', fontdict={'fontsize': 8})
ax1.flatten()[13].set_title('$g$ = 4.6', fontdict={'fontsize': 8})
ax1.flatten()[14].set_title('$g$ = 5.4', fontdict={'fontsize': 8})

ax1.flatten()[14].set_ylabel('$\eta$ = 3.0', rotation=0, fontdict={'fontsize': 8})
ax1.flatten()[14].yaxis.set_label_coords(1.15, 0.5)
ax1.flatten()[17].set_ylabel('$\eta$ = 2.2', rotation=0, fontdict={'fontsize': 8})
ax1.flatten()[17].yaxis.set_label_coords(1.15, 0.5)
ax1.flatten()[20].set_ylabel('$\eta$ = 1.6', rotation=0, fontdict={'fontsize': 8})
ax1.flatten()[20].yaxis.set_label_coords(1.15, 0.5)

ax1.flatten()[18].set_xlabel('Hz', fontdict={'fontsize': 8})
ax1.flatten()[19].set_xlabel('Hz', fontdict={'fontsize': 8})
ax1.flatten()[20].set_xlabel('Hz', fontdict={'fontsize': 8})

ax1.flatten()[12].set_ylabel('$\mathrm{mV^2/Hz}$', fontdict={'fontsize': 8})
ax1.flatten()[15].set_ylabel('$\mathrm{mV^2/Hz}$', fontdict={'fontsize': 8})
ax1.flatten()[18].set_ylabel('$\mathrm{mV^2/Hz}$', fontdict={'fontsize': 8})


for i in range(8):
    ax1.flatten()[12].loglog(freqs[:50], avg_psd_arr[7,0,i,4,:50], color=plt.cm.Greys(7/14 + i/14), linewidth=0.8)
    ax1.flatten()[13].loglog(freqs[:50], avg_psd_arr[7,3,i,4,:50], color=plt.cm.Greys(7/14 + i/14), linewidth=0.8)
    ax1.flatten()[14].loglog(freqs[:50], avg_psd_arr[7,7,i,4,:50], color=plt.cm.Greys(7/14 + i/14), linewidth=0.8)
    ax1.flatten()[15].loglog(freqs[:50], avg_psd_arr[3,0,i,4,:50], color=plt.cm.Greys(7/14 + i/14), linewidth=0.8)
    ax1.flatten()[16].loglog(freqs[:50], avg_psd_arr[3,3,i,4,:50], color=plt.cm.Greys(7/14 + i/14), linewidth=0.8)
    ax1.flatten()[17].loglog(freqs[:50], avg_psd_arr[3,7,i,4,:50], color=plt.cm.Greys(7/14 + i/14), linewidth=0.8)
    ax1.flatten()[18].loglog(freqs[:50], avg_psd_arr[0,0,i,4,:50], color=plt.cm.Greys(7/14 + i/14), linewidth=0.8)
    ax1.flatten()[19].loglog(freqs[:50], avg_psd_arr[0,3,i,4,:50], color=plt.cm.Greys(7/14 + i/14), linewidth=0.8)
    ax1.flatten()[20].loglog(freqs[:50], avg_psd_arr[0,7,i,4,:50], color=plt.cm.Greys(7/14 + i/14), linewidth=0.8)








# fig1.savefig(os.path.join(savedir, 'lfps_varying_g_j'), dpi=200)


# PARAMTER eta EDGE PLOTS
fig2, ax2 = plt.subplots(ncols=3,nrows=3, figsize=[7,5], sharex=True, sharey=True)
for ax in ax2.flatten():
    clear_axis(ax)
    ax.tick_params(axis='both', which='both', labelsize=8)

ax2[0,0].set_xlim(4,200)
ax2[0,0].set_ylim(1e-7, 1)
ax2.flatten()[0].set_title('J = 0.06', fontdict={'fontsize': 8})
ax2.flatten()[1].set_title('J = 0.1', fontdict={'fontsize': 8})
ax2.flatten()[2].set_title('J = 0.2', fontdict={'fontsize': 8})

ax2.flatten()[2].set_ylabel('$\eta$ = 3.0', rotation=0, fontdict={'fontsize': 8})
ax2.flatten()[2].yaxis.set_label_coords(1.1, 0.5)
ax2.flatten()[5].set_ylabel('$\eta$ = 2.2', rotation=0, fontdict={'fontsize': 8})
ax2.flatten()[5].yaxis.set_label_coords(1.1, 0.5)
ax2.flatten()[8].set_ylabel('$\eta$ = 1.6', rotation=0, fontdict={'fontsize': 8})
ax2.flatten()[8].yaxis.set_label_coords(1.1, 0.5)

ax2.flatten()[6].set_xlabel('Hz', fontdict={'fontsize': 8})
ax2.flatten()[7].set_xlabel('Hz', fontdict={'fontsize': 8})
ax2.flatten()[8].set_xlabel('Hz', fontdict={'fontsize': 8})

ax2.flatten()[0].set_ylabel('$\mathrm{mV^2/Hz}$', fontdict={'fontsize': 8})
ax2.flatten()[3].set_ylabel('$\mathrm{mV^2/Hz}$', fontdict={'fontsize': 8})
ax2.flatten()[6].set_ylabel('$\mathrm{mV^2/Hz}$', fontdict={'fontsize': 8})

for i in range(8):
    ax2.flatten()[0].loglog(freqs[:50], avg_psd_arr[7,i,0,4,:50], color=plt.cm.Greys(7/14 + i/14), linewidth=0.8)
    ax2.flatten()[1].loglog(freqs[:50], avg_psd_arr[7,i,2,4,:50], color=plt.cm.Greys(7/14 + i/14), linewidth=0.8)
    ax2.flatten()[2].loglog(freqs[:50], avg_psd_arr[7,i,7,4,:50], color=plt.cm.Greys(7/14 + i/14), linewidth=0.8)
    ax2.flatten()[3].loglog(freqs[:50], avg_psd_arr[3,i,0,4,:50], color=plt.cm.Greys(7/14 + i/14), linewidth=0.8)
    ax2.flatten()[4].loglog(freqs[:50], avg_psd_arr[3,i,2,4,:50], color=plt.cm.Greys(7/14 + i/14), linewidth=0.8)
    ax2.flatten()[5].loglog(freqs[:50], avg_psd_arr[3,i,7,4,:50], color=plt.cm.Greys(7/14 + i/14), linewidth=0.8)
    ax2.flatten()[6].loglog(freqs[:50], avg_psd_arr[0,i,0,4,:50], color=plt.cm.Greys(7/14 + i/14), linewidth=0.8)
    ax2.flatten()[7].loglog(freqs[:50], avg_psd_arr[0,i,2,4,:50], color=plt.cm.Greys(7/14 + i/14), linewidth=0.8)
    ax2.flatten()[8].loglog(freqs[:50], avg_psd_arr[0,i,7,4,:50], color=plt.cm.Greys(7/14 + i/14), linewidth=0.8)

fig2.savefig(os.path.join(savedir, 'lfps_varying_g'), dpi=200)



## PARAMTER j EDGE PLOTS
fig3, ax3 = plt.subplots(ncols=3,nrows=3, figsize=[7,5], sharex=True, sharey=True)
for ax in ax3.flatten():
    clear_axis(ax)
    ax.tick_params(axis='both', which='both', labelsize=8)


ax3.flatten()[0].set_title('$g$ = 4.0', fontdict={'fontsize': 8})
ax3.flatten()[1].set_title('$g$ = 4.6', fontdict={'fontsize': 8})
ax3.flatten()[2].set_title('$g$ = 5.4', fontdict={'fontsize': 8})

ax3.flatten()[2].set_ylabel('$\eta$ = 3.0', rotation=0, fontdict={'fontsize': 8})
ax3.flatten()[2].yaxis.set_label_coords(1.15, 0.5)
ax3.flatten()[5].set_ylabel('$\eta$ = 2.2', rotation=0, fontdict={'fontsize': 8})
ax3.flatten()[5].yaxis.set_label_coords(1.15, 0.5)
ax3.flatten()[8].set_ylabel('$\eta$ = 1.6', rotation=0, fontdict={'fontsize': 8})
ax3.flatten()[8].yaxis.set_label_coords(1.15, 0.5)

ax3.flatten()[6].set_xlabel('Hz', fontdict={'fontsize': 8})
ax3.flatten()[7].set_xlabel('Hz', fontdict={'fontsize': 8})
ax3.flatten()[8].set_xlabel('Hz', fontdict={'fontsize': 8})

ax3.flatten()[0].set_ylabel('$\mathrm{mV^2/Hz}$', fontdict={'fontsize': 8})
ax3.flatten()[3].set_ylabel('$\mathrm{mV^2/Hz}$', fontdict={'fontsize': 8})
ax3.flatten()[6].set_ylabel('$\mathrm{mV^2/Hz}$', fontdict={'fontsize': 8})


for i in range(8):
    ax3.flatten()[0].loglog(freqs[:50], avg_psd_arr[7,0,i,4,:50], color=plt.cm.Greys(7/14 + i/14), linewidth=0.8)
    ax3.flatten()[1].loglog(freqs[:50], avg_psd_arr[7,3,i,4,:50], color=plt.cm.Greys(7/14 + i/14), linewidth=0.8)
    ax3.flatten()[2].loglog(freqs[:50], avg_psd_arr[7,7,i,4,:50], color=plt.cm.Greys(7/14 + i/14), linewidth=0.8)
    ax3.flatten()[3].loglog(freqs[:50], avg_psd_arr[3,0,i,4,:50], color=plt.cm.Greys(7/14 + i/14), linewidth=0.8)
    ax3.flatten()[4].loglog(freqs[:50], avg_psd_arr[3,3,i,4,:50], color=plt.cm.Greys(7/14 + i/14), linewidth=0.8)
    ax3.flatten()[5].loglog(freqs[:50], avg_psd_arr[3,7,i,4,:50], color=plt.cm.Greys(7/14 + i/14), linewidth=0.8)
    ax3.flatten()[6].loglog(freqs[:50], avg_psd_arr[0,0,i,4,:50], color=plt.cm.Greys(7/14 + i/14), linewidth=0.8)
    ax3.flatten()[7].loglog(freqs[:50], avg_psd_arr[0,3,i,4,:50], color=plt.cm.Greys(7/14 + i/14), linewidth=0.8)
    ax3.flatten()[8].loglog(freqs[:50], avg_psd_arr[0,7,i,4,:50], color=plt.cm.Greys(7/14 + i/14), linewidth=0.8)

# fig3.savefig(os.path.join(savedir, 'lfps_varying_j'), dpi=200)

fig4, ax4 = plt.subplots(3, sharex=True, figsize=[5,3])
fig4.subplots_adjust(hspace=0.7, bottom=0.12)
for ax in ax4.flatten():
    clear_axis(ax)
    ax.tick_params(axis='both', which='both', labelsize=8)

offset = 300
length = 600
ax4[0].plot(np.arange(offset, offset+length), lfp_arr[3, 0, 0, 0, 4, offset:offset+length], color=plt.cm.Greys(7/14 + i/14), linewidth=0.8)
ax4[0].set_title('$\eta$ = %.1f, $g$ = %.1f, $J$ = %.2f'%(2.2, 4.0, 0.06), fontdict={'fontsize': 8})
ax4[0].set_yticks([-0.5, 0, 0.5])
ax4[0].set_xlim(300, 900)
ax4[0].set_ylabel('mV', rotation=0, fontdict={'fontsize': 8})
ax4[1].plot(np.arange(offset, offset+length), lfp_arr[7, 3, 3, 0, 4, offset:offset+length], color=plt.cm.Greys(7/14 + i/14), linewidth=0.8)
ax4[1].set_title('$\eta$ = %.1f, $g$ = %.1f, $J$ = %.2f'%(3.0, 4.6, 0.12), fontdict={'fontsize': 8})
ax4[1].set_yticks([-1.0, 0, 1.0])
ax4[1].set_xlim(300, 900)
ax4[1].set_ylabel('mV', rotation=0, fontdict={'fontsize': 8})
ax4[2].plot(np.arange(offset, offset+length), lfp_arr[3, 7, 7, 0, 4, offset:offset+length], color=plt.cm.Greys(7/14 + i/14), linewidth=0.8)
ax4[2].set_title('$\eta$ = %.1f, $g$ = %.1f, $J$ = %.2f'%(2.2, 5.4, 0.2), fontdict={'fontsize': 8})
ax4[2].set_yticks([-1.0, 0, 1.0])
ax4[2].set_xlim(300, 900)
ax4[2].set_xlabel('ms', fontdict={'fontsize': 8})
ax4[2].set_ylabel('mV', rotation=0, fontdict={'fontsize': 8})

# fig4.savefig(os.path.join(savedir, 'lfps_examples'), dpi=200)
