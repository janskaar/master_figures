import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
from ConvNet.convnet import inference_shallow
import os

modelpath = os.path.join('C:/Users/JanEirik/Documents/TensorFlow/network_saves/g_90_momentumgd/best/model.ckpt')

im = tf.placeholder(tf.float32, shape=(None, 1, 300, 6))

phase_train = tf.placeholder(tf.bool, name='phase_train')
keep_prob = tf.placeholder(tf.float32, name='keep_prob')
output = inference_shallow(im, phase_train, keep_prob, softmax=True)
saver = tf.train.Saver()
with tf.Session() as sess:
    saver.restore(sess, modelpath)
    filters = sess.run(tf.get_default_graph().get_tensor_by_name('ConvReluBN1/kernel:0'))

fig1, ax1 = plt.subplots(ncols=11, nrows=11, sharex=True, sharey=True)
for a in ax1.flatten():
    a.axis('off')
for i in range(121):
    ax1.flatten()[i].plot(filters[0,:,2,i], 'black')
ax1[-1,0].axis('on')
ax1[-1,0].set_xlabel('$\\tau$ (ms)')
ax1[-1,0].set_yticks([])
ax1[-1,0].set_xticks([0,11])
ax1[-1,0].spines['top'].set_visible(False)
ax1[-1,0].spines['right'].set_visible(False)
ax1[-1,0].spines['left'].set_visible(False)

fig1.suptitle('First layer filters, ch. 3')









fig2, ax2 = plt.subplots(ncols=10, sharex=True, sharey=True, figsize=[9,5])
top = (0.3+9)*0.05 + 0.04

for i in range(10):
    for j in range(6):
        ax2[i].plot(filters[0,:,j,i+100]+ top - (0.3+j)*0.05, 'black')

ax2[0].set_ylim(0.21, 0.53)
ax2[0].set_yticks([top - (0.35+j)*0.05 for j in range(6)])
ax2[0].set_yticklabels(['ch. %i'%(i) for i in range(1,7)])
for i in range(10):
    ax2[i].set_xticks([])
for i in range(1,10):
    ax2[i].tick_params(which='both', length=0)






















1
