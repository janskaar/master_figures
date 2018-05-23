import h5py, os
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec


def read_ids_parameters(path):
    with open(path, 'r') as f:
        info = f.readlines()
    del info[0]
    del info[0]
    labels = []
    ids = []
    for i, e in enumerate(info):
        if 'eta = ' in e:
            j = e[-5:-2]
            g = e[15:18]
            eta = e[6:9]
        else:
            ids.append(e.replace('\n', ''))
            label = np.array([float(eta), float(g), float(j)], dtype=np.float32)
            labels.append(label)
    return ids, labels

def clear_axis(ax):
    ax.set_xticks([])
    ax.set_yticks([])


top_dir = os.path.join('C:\\Users\\JanEirik\\Documents\\TensorFlow\\data\\hybrid_brunel52\\nest_output')
savedir = os.path.join('..\\plots\\')

ids, labels = read_ids_parameters(os.path.join(top_dir, 'info.txt'))

etas = np.array([1.6, 1.8, 2.0, 2.2, 2.4, 2.6, 2.8, 3.0], dtype=np.float32)
gs = np.array([4.0, 4.2, 4.4, 4.6, 4.8, 5.0, 5.2, 5.4], dtype=np.float32)
js = np.array([0.06, 0.08, 0.1, 0.12, 0.14, 0.16, 0.18, 0.2], dtype=np.float32)

cvs = np.zeros([8, 8, 8])
cv_vars = np.zeros([8,8,8])
frates = np.zeros([8, 8, 8])
frate_vars = np.zeros([8,8,8])
corrs = np.zeros([8, 8, 8])
corr_vars = np.zeros([8,8,8])
rasters = []

full_frates = None
full_corrs = None
full_cvs = None
pearsons3 = None
hists = None
firing_rates = None

for i, lab in enumerate(labels):
    eta = np.nonzero(etas == lab[0])
    g = np.nonzero(gs == lab[1])
    j = np.nonzero(js == lab[2])
    with h5py.File(os.path.join(top_dir, ids[i], 'LFP_firing_rate.h5'), 'r') as f:
        if full_frates is None:
            full_frates = np.zeros([8,8,8,len(f['frates'])])
            full_cvs = np.zeros([8,8,8,len(f['cvs'])])
            full_corrs = np.zeros([8,8,8,len(f['pearsons'])])
            pearsons3 = np.zeros([8,8,8,len(f['pearsons3'])])
            hists = np.zeros([8,8,8, len(f['ex_hist'])])
            firing_rates = np.zeros([8,8,8, len(f['average_firing_rate'])])
        firing_rates[eta, g, j] = f['average_firing_rate'][:]
        full_frates[eta, g, j] = f['frates'][:]
        full_cvs[eta, g, j] = f['cvs'][:]
        full_corrs[eta, g, j] = f['pearsons'][:]
        pearsons3[eta, g, j] = f['pearsons3'][:]
        hists[eta, g, j] = f['ex_hist'][:] + f['in_hist'][:]
        cvs[eta, g, j] = np.mean(f['cvs'][:])
        cv_vars[eta, g, j] = np.var(f['cvs'][:])
        frates[eta, g, j] = np.mean(f['frates'][:])
        frate_vars[eta, g, j] = np.var(f['frates'][:])
        corrs[eta, g, j] = np.mean(f['pearsons'][:])
        corr_vars[eta, g, j] = np.var(f['pearsons'][:])
        rasters.append(f['rasters'][:])



# FIRING RATE AND CV PLOTS
fig1, ax1 = plt.subplots(ncols=5, nrows=4, figsize=[5.0,5.4],
                         gridspec_kw={'width_ratios':[4,4,1,4,4]})
for ax in ax1[:,2]:
    ax.axis('off')

fig1.subplots_adjust(hspace=0.0, wspace=0.1, right=0.9, left=0.1)
for i, ax in enumerate(ax1[:,:2].flatten()):
    clear_axis(ax)
    ax.set_title('$J$ = %.2f'%(js[i]), fontdict={'fontsize': 6})
    ax.title.set_position([0.5,0.95])

for i, ax in enumerate(ax1[:,:2].flatten()):
    im1 = ax.imshow(frates[:,:,i]*1000, vmin=0.0, vmax=170, origin='lower')

for i, ax in enumerate(ax1[:,3:].flatten()):
    clear_axis(ax)
    ax.set_title('$J$ = %.2f'%(js[i]), fontdict={'fontsize': 6})
    ax.title.set_position([0.5,0.95])

for i, ax in enumerate(ax1[:,3:].flatten()):
    im2 = ax.imshow(cvs[:,:,i], vmin=0.0, vmax=1.3, origin='lower')

tax1 = fig1.add_axes([0.1, 0.9, 0.3, 0.02])
tax2 = fig1.add_axes([0.6, 0.9, 0.3, 0.02])
tax1.axis('off')
tax2.axis('off')
tax1.text(0.35,0, 'Firing rates')
tax2.text(0.3,0, 'CVs')

cax1 = fig1.add_axes([0.065, 0.4, 0.02,0.2])
cax1.tick_params(labelsize=6)
cbar1 = plt.colorbar(im1, cax=cax1, ticks=[0, 80, 160])
cax1.yaxis.set_ticks_position('left')
cax1.yaxis.set_label_position('left')
cax1.yaxis.set_label_coords(-2.5, 0.5)
cax1.set_ylabel('Hz', fontdict={'fontsize': 6}, rotation=0)


ax1[3,0].set_xticks(range(8)[1::2])
ax1[3,0].set_yticks(range(8))
ax1[3,0].set_xticklabels(gs[1::2], fontdict={'fontsize': 6})
ax1[3,0].set_yticklabels(etas, fontdict={'fontsize': 6})
ax1[3,0].set_ylabel('$\eta$', fontdict={'fontsize': 8}, rotation=0)
ax1[3,0].set_xlabel('$g$', fontdict={'fontsize': 8})

cax2 = fig1.add_axes([0.915, 0.4, 0.02,0.2])
cax2.tick_params(labelsize=6)
cbar2 = plt.colorbar(im2, cax=cax2, ticks=[0, 0.6, 1.2])


# fig1.savefig(os.path.join(savedir, 'firing_rates_cvs'), dpi=200, bbox_inches='tight')


## CORRELATION PLOTS
pearsonmeans = np.mean(pearsons3, axis=3)
pearsonsds = np.std(pearsons3, axis=3)


fig2, ax2 = plt.subplots(ncols=2, nrows=4, figsize=[2.6,5.4])

fig2.subplots_adjust(hspace=0.1, wspace=0.1, right=0.95, left=0.25)
for i, ax in enumerate(ax2.flatten()):
    clear_axis(ax)
    ax.set_title('$J$ = %.2f'%(js[i]), fontdict={'fontsize': 6})
    ax.title.set_position([0.5,0.95])

for i, ax in enumerate(ax2.flatten()):
    im1 = ax.imshow(pearsonmeans[:,:,i], vmin=0.0, vmax=0.07, origin='lower')

cax1 = fig2.add_axes([0.15, 0.4, 0.04,0.2])
cax1.tick_params(labelsize=6)
cbar1 = plt.colorbar(im1, cax=cax1)
cbar1.set_ticks([0.01,0.04,0.07])
cax1.yaxis.set_ticks_position('left')
cax1.yaxis.set_label_position('left')


ax2[3,0].set_xticks(range(8)[1::2])
ax2[3,0].set_yticks(range(8))
ax2[3,0].set_xticklabels(gs[1::2], fontdict={'fontsize': 6})
ax2[3,0].set_yticklabels(etas, fontdict={'fontsize': 6})
ax2[3,0].set_ylabel('$\eta$', fontdict={'fontsize': 8}, rotation=0)
ax2[3,0].set_xlabel('$g$', fontdict={'fontsize': 8})
fig2.suptitle('Pearson correlation coeff.')




### RASTER PLOTS
# raster_labels = [np.array([2.2,4.0,0.06], dtype=np.float32),
#                  np.array([3.0,4.6,0.12], dtype=np.float32),
#                  np.array([2.2,5.4,0.20], dtype=np.float32)]
#
# t_start = 600
# t_stop = 800
# barbins = np.arange(t_start, t_stop)
#
# histlist = []
# for lab in raster_labels:
#     eta = np.squeeze(np.nonzero(etas == lab[0]))
#     g = np.squeeze(np.nonzero(gs == lab[1]))
#     j = np.squeeze(np.nonzero(js == lab[2]))
#     histlist.append(hists[eta, g, j, t_start:t_stop])
#
# raster_index = [np.nonzero(np.all((lab == labels), axis=1)) for lab in raster_labels]
# rs = [rasters[i[0][0]] for i in raster_index]
#
# fig = plt.figure(figsize=[4,12])
# gspec = GridSpec(14,1, hspace=0.0, wspace=0.0)
#
# r1ax = fig.add_subplot(gspec[0:3,0])
# r1ax.set_xticks([])
# r1ax.set_yticks([])
# r1ax.tick_params(axis='both', which='both', labelsize=8)
# r1ax.set_xlim(t_start, t_stop)
# r1ax.set_title('$\eta$ = %.1f, $g$ = %.1f, $J$ = %.2f'%(tuple(raster_labels[0])), fontdict={'fontsize': 8})
# for i, r in enumerate(rs[0][:50]):
#     r_i = r[np.nonzero(np.logical_and(r < t_stop, r > t_start))]
#     r1ax.scatter(r_i, np.ones_like(r_i)*i, marker='|', color='black', s=5.0, linewidth=0.8)
#
# hist1ax = fig.add_subplot(gspec[3,0])
# hist1ax.bar(barbins, height=histlist[0], width=1.0, color='black')
# hist1ax.set_xticks([t_start, (t_stop-t_start)/2 +t_start, t_stop])
# hist1ax.set_xlim(t_start, t_stop)
# hist1ax.tick_params(axis='both', which='both', labelsize=8)
#
# r2ax = fig.add_subplot(gspec[5:8,0])
# r2ax.set_xticks([])
# r2ax.set_yticks([])
# r2ax.tick_params(axis='both', which='both', labelsize=8)
# r2ax.set_xlim(t_start, t_stop)
# r2ax.set_title('$\eta$ = %.1f, $g$ = %.1f, $J$ = %.2f'%(tuple(raster_labels[1])), fontdict={'fontsize': 8})
# for i, r in enumerate(rs[1][:50]):
#     r_i = r[np.nonzero(np.logical_and(r < t_stop, r > t_start))]
#     r2ax.scatter(r_i, np.ones_like(r_i)*i, marker='|', color='black', s=5.0, linewidth=0.8)
# hist2ax = fig.add_subplot(gspec[8,0])
# hist2ax.bar(barbins, height=histlist[1], width=1.0, color='black')
# hist2ax.set_xticks([t_start, (t_stop-t_start)/2 +t_start, t_stop])
# hist2ax.set_xlim(t_start, t_stop)
# hist2ax.tick_params(axis='both', which='both', labelsize=8)
#
# r3ax = fig.add_subplot(gspec[10:13,0])
# r3ax.set_xticks([])
# r3ax.set_yticks([])
# r3ax.tick_params(axis='both', which='both', labelsize=8)
# r3ax.set_xlim(t_start, t_stop)
# r3ax.set_title('$\eta$ = %.1f, $g$ = %.1f, $J$ = %.2f'%(tuple(raster_labels[2])), fontdict={'fontsize': 8})
# for i, r in enumerate(rs[2][:50]):
#     r_i = r[np.nonzero(np.logical_and(r < t_stop, r > t_start))]
#     r3ax.scatter(r_i, np.ones_like(r_i)*i, marker='|', color='black', s=5.0, linewidth=0.8)
#
# hist3ax = fig.add_subplot(gspec[13,0])
# hist3ax.bar(barbins, height=histlist[2], width=1.0, color='black')
# hist3ax.set_xticks([t_start, (t_stop-t_start)/2 +t_start, t_stop])
# hist3ax.set_xlim(t_start, t_stop)
# hist3ax.tick_params(axis='both', which='both', labelsize=8)
# hist3ax.set_xlabel('$t$ (ms)', fontdict={'fontsize': 8})
#
# fig.savefig(os.path.join(savedir, 'example_rasters'), dpi=200, bbox_inches='tight')

# fig4, ax4 = plt.subplots(3, sharex=True, figsize=[4,9])
# fig4.subplots_adjust(hspace=0.2)
# for k in range(3):
#     ax4[k].set_xticks([300, 400, 500])
#     ax4[k].set_yticks([])
#     ax4[k].tick_params(axis='both', which='both', labelsize=8)
#     ax4[k].set_xlim(300,500)
#     ax4[k].set_title('$\eta$ = %.1f, $g$ = %.1f, $J$ = %.2f'%(tuple(raster_labels[k])), fontdict={'fontsize': 8})
#     for i, r in enumerate(rs[k][:50]):
#         r_i = r[np.nonzero(np.logical_and(r < t_stop, r > t_start))]
#         ax4[k].scatter(r_i, np.ones_like(r_i)*i, marker='|', color='black', s=4.0, linewidth=0.7)
# ax4[-1].set_xlabel('ms', fontdict={'fontsize': 8})


# fig4.savefig(os.path.join(savedir, 'example_rasters'), dpi=200, bbox_inches='tight')
