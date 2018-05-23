import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
from ConvNet.convnet import inference_shallow, accuracy
from ConvNet.input_functions import load_datasets
import os

test_path = os.path.join('C:/Users/JanEirik/Documents/TensorFlow/tfrecords_data/final/300ms/test_data_ngj_augmented.tfrecord')
training_path = os.path.join('C:/Users/JanEirik/Documents/TensorFlow/tfrecords_data/final/300ms/training_data_ngj_augmented.tfrecord')
validation_path = os.path.join('C:/Users/JanEirik/Documents/TensorFlow/tfrecords_data/final/300ms/validation_data_ngj_augmented.tfrecord')
etapath = os.path.join('C:/Users/JanEirik/Documents/TensorFlow/network_saves/eta_75_momentumgd/best/model.ckpt')
gpath = os.path.join('C:/Users/JanEirik/Documents/TensorFlow/network_saves/g_90_momentumgd/best/model.ckpt')
jpath = os.path.join('C:/Users/JanEirik/Documents/TensorFlow/network_saves/j_90_momentumgd/best/model.ckpt')

## Keep dtype float32 to match TF
etas = np.array([1.6, 1.8, 2.0, 2.2, 2.4, 2.6, 2.8, 3.0], dtype=np.float32)
gs = np.array([4.0, 4.2, 4.4, 4.6, 4.8, 5.0, 5.2, 5.4], dtype=np.float32)
js = np.array([0.06, 0.08, 0.1, 0.12, 0.14, 0.16, 0.18, 0.2], dtype=np.float32)

PARAMETER = 0
ntrain_per_label = 11*9
ntest_per_label = 11*2
nval_per_label = 11*1
softmax=True

x, y, train_init, test_init, val_init = load_datasets(test_path, training_path, validation_path, batch_size=400, length=300, one_hot=True)
x = tf.transpose(x, perm=[0, 3, 2, 1])

phase_train = tf.placeholder(tf.bool, name='phase_train')
keep_prob = tf.placeholder(tf.float32, name='keep_prob')

output = inference_shallow(x, phase_train, keep_prob, softmax=True)


global_vars = tf.global_variables()
loading_vars = global_vars[:3]
del(global_vars[:3])


saver = tf.train.Saver(var_list = global_vars)

sess = tf.Session()
saver.restore(sess, gpath)
sess.run(tf.variables_initializer(loading_vars))



def get_accuracy_classification(path, init, param, n_per_label, loading_vars):
    etas = np.array([1.6, 1.8, 2.0, 2.2, 2.4, 2.6, 2.8, 3.0], dtype=np.float32)
    gs = np.array([4.0, 4.2, 4.4, 4.6, 4.8, 5.0, 5.2, 5.4], dtype=np.float32)
    js = np.array([0.06, 0.08, 0.1, 0.12, 0.14, 0.16, 0.18, 0.2], dtype=np.float32)
    accuracy_arr = np.zeros((len(etas), len(gs), len(js)), dtype=np.float64)
    npass_arr = np.zeros([len(etas), len(gs), len(js)], dtype=np.int64)
    with tf.Session() as sess:
        saver.restore(sess, path)
        sess.run(tf.variables_initializer(loading_vars))
        sess.run(init)
        while True:
            try:
                predictions, labels = sess.run([output,y], feed_dict={phase_train:False, keep_prob:1.0})
                for i, lab in enumerate(labels):
                    etapos = np.argmax(lab[0])
                    gpos = np.argmax(lab[1])
                    jpos = np.argmax(lab[2])
                    accuracy_arr[etapos, gpos, jpos] += np.equal(np.argmax(predictions[i]), np.argmax(labels[i, param])).astype(np.float64)
                    npass_arr[etapos, gpos, jpos] += 1
            except tf.errors.OutOfRangeError:
                break
    print(np.min(npass_arr))
    print(np.max(npass_arr))
    accuracy_arr /= npass_arr
    return accuracy_arr

# eta_train_accuracy = get_accuracy_classification(etapath, train_init, 1, 9*11, loading_vars)
eta_test_accuracy = get_accuracy_classification(etapath, test_init, 0, 9*11, loading_vars)
# eta_val_accuracy = get_accuracy_classification(etapath, val_init, 1, 9*11, loading_vars)

# g_train_accuracy = get_accuracy_classification(etapath, train_init, 1, 9*11, loading_vars)
g_test_accuracy = get_accuracy_classification(gpath, test_init, 1, 9*11, loading_vars)
# g_val_accuracy = get_accuracy_classification(etapath, val_init, 1, 9*11, loading_vars)

# j_train_accuracy = get_accuracy_classification(jpath, train_init, 1, 9*11, loading_vars)
j_test_accuracy = get_accuracy_classification(jpath, test_init, 2, 9*11, loading_vars)
# j_val_accuracy = get_accuracy_classification(jpath, val_init, 1, 9*11, loading_vars)



def get_errors(path, init, param, n_per_label):
    etas = np.array([1.6, 1.8, 2.0, 2.2, 2.4, 2.6, 2.8, 3.0], dtype=np.float32)
    gs = np.array([4.0, 4.2, 4.4, 4.6, 4.8, 5.0, 5.2, 5.4], dtype=np.float32)
    js = np.array([0.06, 0.08, 0.1, 0.12, 0.14, 0.16, 0.18, 0.2], dtype=np.float32)
    error_arr = np.zeros([len(etas), len(gs), len(js), n_per_label])
    npass_arr = np.zeros([len(etas), len(gs), len(js)], dtype=np.int64)
    with tf.Session() as sess:
        saver.restore(sess, path)
        sess.run(init)
        while True:
            try:
                predictions, labels = sess.run([output,y], feed_dict={phase_train:False, keep_prob:1.0})
                error = np.absolute(np.subtract(predictions[:,0], labels[:,param]))
                for i, p in enumerate(labels):
                    etapos = np.where(etas == p[0])
                    gpos = np.where(gs == p[1])
                    jpos = np.where(js == p[2])
                    error_arr[etapos, gpos, jpos, npass_arr[etapos, gpos, jpos]] = error[i]
                    npass_arr[etapos, gpos, jpos] += 1
            except tf.errors.OutOfRangeError:
                break
        return error_arr





def clear_axis(ax):
    ax.set_xticks([])
    ax.set_yticks([])

# eta_test_errors = get_errors(etapath, test_init, 0, 12)
# eta_val_errors = get_errors(etapath, val_init, 0, 6)
# g_test_errors = get_errors(gpath, test_init, 1, 12)
# g_val_errors = get_errors(gpath, val_init, 1, 6)
# j_test_errors = get_errors(jpath, test_init, 2, 12)
# j_val_errors = get_errors(jpath, val_init, 2, 6)
#
# eta_means = np.mean(eta_test_errors, axis=-1) + np.mean(eta_val_errors, axis=-1)
# eta_stds = np.std(np.concatenate((eta_test_errors, eta_val_errors), axis=-1), axis=-1)
# g_means = np.mean(g_test_errors, axis=-1) + np.mean(g_val_errors, axis=-1)
# g_stds = np.std(np.concatenate((g_test_errors, g_val_errors), axis=-1), axis=-1)
# j_means = np.mean(j_test_errors, axis=-1) + np.mean(j_val_errors, axis=-1)
# j_stds = np.std(np.concatenate((j_test_errors, j_val_errors), axis=-1), axis=-1)





fig1, ax1 = plt.subplots(ncols=2, nrows=4, figsize=[3.0,6.0])
fig1.subplots_adjust(hspace=0.2, wspace=0.1, right=0.9, left=0.15)
fig1.suptitle('$\eta$ prediction accuracy', x=0.5, y=0.95, fontdict={'fontsize': 8})

for i, ax in enumerate(ax1.flatten()):
    clear_axis(ax)
    ax.set_title('$J$ = %.2f'%(js[i]), fontdict={'fontsize': 6})
    ax.title.set_position([0.5,0.95])
    im = ax.imshow(eta_test_accuracy[:,:,i], vmax=1.0, vmin=0, origin='lower')

ax1[3,0].set_xticks(range(8)[1::2])
ax1[3,0].set_yticks(range(8))
ax1[3,0].set_xticklabels(gs[1::2], fontdict={'fontsize': 5.5})
ax1[3,0].set_yticklabels(etas, fontdict={'fontsize': 5.5})
ax1[3,0].set_ylabel('$\eta$', fontdict={'fontsize': 7}, rotation=0)
ax1[3,0].set_xlabel('$g$', fontdict={'fontsize': 7})
cax1 = fig1.add_axes([0.12, 0.4, 0.02,0.2])
cax1.tick_params(labelsize=6)
cbar1 = plt.colorbar(im, cax=cax1)
cbar1.set_ticks([0, 1.0])
cax1.yaxis.set_ticks_position('left')
cax1.yaxis.set_label_position('left')
cax1.yaxis.set_label_coords(-2.5, 0.5)
# fig1.savefig('../plots/eta_accuracy_classification_new', dpi=200, bbox_inches='tight')



fig2, ax2 = plt.subplots(ncols=2, nrows=4, figsize=[3.0,6.0])
fig2.subplots_adjust(hspace=0.2, wspace=0.1, right=0.9, left=0.15)
fig2.suptitle('$g$ prediction accuracy', x=0.5, y=0.95, fontdict={'fontsize': 8})

for i, ax in enumerate(ax2.flatten()):
    clear_axis(ax)
    ax.set_title('$J$ = %.2f'%(js[i]), fontdict={'fontsize': 6})
    ax.title.set_position([0.5,0.95])
    im = ax.imshow(g_test_accuracy[:,:,i], vmax=1.0, vmin=0, origin='lower')

ax2[3,0].set_xticks(range(8)[1::2])
ax2[3,0].set_yticks(range(8))
ax2[3,0].set_xticklabels(gs[1::2], fontdict={'fontsize': 6})
ax2[3,0].set_yticklabels(etas, fontdict={'fontsize': 6})
ax2[3,0].set_ylabel('$\eta$', fontdict={'fontsize': 6}, rotation=0)
ax2[3,0].set_xlabel('$g$', fontdict={'fontsize': 6})
cax2 = fig2.add_axes([0.12, 0.4, 0.02,0.2])
cax2.tick_params(labelsize=6)
cbar2 = plt.colorbar(im, cax=cax2)
cbar2.set_ticks([0, 1.0])
cax2.yaxis.set_ticks_position('left')
cax2.yaxis.set_label_position('left')
cax2.yaxis.set_label_coords(-2.5, 0.5)
fig2.savefig('../plots/g_accuracy_classification_new', dpi=200, bbox_inches='tight')

fig3, ax3 = plt.subplots(ncols=2, nrows=4, figsize=[3.0,6.0])
fig3.subplots_adjust(hspace=0.2, wspace=0.1, right=0.9, left=0.15)
fig3.suptitle('$J$ prediction accuracy', x=0.5, y=0.95, fontdict={'fontsize': 8})

for i, ax in enumerate(ax3.flatten()):
    clear_axis(ax)
    ax.set_title('$J$ = %.2f'%(js[i]), fontdict={'fontsize': 6})
    ax.title.set_position([0.5,0.95])
    im = ax.imshow(j_test_accuracy[:,:,i], vmax=1.0, vmin=0., origin='lower')

ax3[3,0].set_xticks(range(8)[1::2])
ax3[3,0].set_yticks(range(8))
ax3[3,0].set_xticklabels(gs[1::2], fontdict={'fontsize': 6})
ax3[3,0].set_yticklabels(etas, fontdict={'fontsize': 5.5})
ax3[3,0].set_ylabel('$\eta$', fontdict={'fontsize': 6}, rotation=0)
ax3[3,0].set_xlabel('$g$', fontdict={'fontsize': 6})
cax3 = fig3.add_axes([0.12, 0.4, 0.02,0.2])
cax3.tick_params(labelsize=6)
cbar3 = plt.colorbar(im, cax=cax3)
cbar3.set_ticks([0, 1.0])
cax3.yaxis.set_ticks_position('left')
cax3.yaxis.set_label_position('left')
cax3.yaxis.set_label_coords(-2.5, 0.5)
# fig3.savefig('../plots/j_accuracy_classification_new', dpi=200, bbox_inches='tight')



fig4, ax4 = plt.subplots(ncols=1, nrows=3, figsize=[3,7], sharex=True, sharey=True)
fig4.subplots_adjust(hspace=0.3, right=0.76, left=0.15)


lw = 2.0
for i, acc in enumerate((eta_test_accuracy, g_test_accuracy, j_test_accuracy)):
    ax4[i].plot([np.mean(acc[j,:,:]) for j in range(8)], color='r', label='$\eta$', linewidth=lw)
    ax4[i].plot([np.mean(acc[:,j,:]) for j in range(8)], color='orange', label='g', linewidth=lw)
    ax4[i].plot([np.mean(acc[:,:,j]) for j in range(8)], color='royalblue', label='J', linewidth=lw)
    ax4[i].set_ylim(0.5,1.0)
    ax4[i].set_xlim(0,7)
    ax4[i].tick_params(axis='both', which='both', labelsize=8)
    ax4[i].set_yticks([0.6, 0.8, 1.0])
    ax4[i].set_xticks([0, 1, 2, 3, 4, 5, 6, 7])
    ax4[i].set_xticklabels([1, 2, 3, 4, 5, 6, 7, 8])
    ax4[i].set_ylabel('accuracy', fontdict={'fontsize': 9})

ax4[2].set_xlabel('relative param. value', fontdict={'fontsize': 8})
ax4[0].set_title('$\eta$ predictions', fontdict={'fontsize': 9})
ax4[1].set_title('$g$ predictions', fontdict={'fontsize': 9})
ax4[2].set_title('$J$ predictions', fontdict={'fontsize': 9})
handles, labels = ax4[0].get_legend_handles_labels()
fig4.legend(handles, labels, loc='right')
# fig4.savefig('../plots/accuracy_curves_new', dpi=200, bbox_inches='tight')

















1
