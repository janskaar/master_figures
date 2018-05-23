import tensorflow as tf
from tensorboard.backend.event_processing import event_accumulator
import os
import numpy as np
import matplotlib.pyplot as plt



def get_loss_accuracies(accumulator, classification=False, j=False, smooth=False):
    def smooth_func(scalars, weight):
        last = scalars[0]
        smoothed = list()
        for point in scalars:
            smoothed_val = last * weight + (1 - weight) * point
            smoothed.append(smoothed_val)
            last = smoothed_val
        return smoothed
    a = 'accuracy_summaries/'
    te = 'test_accuracy'
    tr = 'training_accuracy'
    va = 'validation_accuracy'
    values = {'training_loss': None,
              'test_loss': None,
              'validation_loss': None}
    if classification:
        values.update({a + te: None,
                       a + tr: None,
                       a + va: None})
    else:
        values.update({a + te + ('_0.02' if j else '_0.2'): None,
                       a + te + ('_0.01' if j else '_0.1'): None,
                       a + te + ('_0.005' if j else '_0.05'): None,
                       a + tr + ('_0.02' if j else '_0.2'): None,
                       a + tr + ('_0.01' if j else '_0.1'): None,
                       a + tr + ('_0.005' if j else '_0.05'): None,
                       a + va + ('_0.02' if j else '_0.2'): None,
                       a + va + ('_0.01' if j else '_0.1'): None,
                       a + va + ('_0.005' if j else '_0.05'): None})
    for key in values.keys():
        values[key] = [scalar.value for scalar in accumulator.Scalars(key)]
    values['steps'] = [scalar.step for scalar in accumulator.Scalars('test_loss')]
    for key in values.keys():
        if a in key:
            values[key.replace(a, '')] = values.pop(key)
    if smooth is not False:
        values['training_loss'] = smooth_func(values['training_loss'], smooth)
        values['test_loss'] = smooth_func(values['training_loss'], smooth)
        values['validation_loss'] = smooth_func(values['training_loss'], smooth)
        values['training_accuracy'] = smooth_func(values['training_accuracy'], smooth)
        values['test_accuracy'] = smooth_func(values['test_accuracy'], smooth)
        values['validation_accuracy'] = smooth_func(values['validation_accuracy'], smooth)
    return values


### NEW_CLASSIFICATION
eta_events = os.path.join('../data/training_data/eta/')
g_events = os.path.join('../data/training_data/g/')
j_events = os.path.join('../data/training_data/j/')


g_ea = event_accumulator.EventAccumulator(g_events)
g_ea.Reload()
g_values = get_loss_accuracies(g_ea, classification=True, smooth=False)

eta_ea = event_accumulator.EventAccumulator(eta_events)
eta_ea.Reload()
eta_values = get_loss_accuracies(eta_ea, classification=True, smooth=False)

j_ea = event_accumulator.EventAccumulator(j_events)
j_ea.Reload()
j_values = get_loss_accuracies(j_ea, j=True, classification=True, smooth=False)


fig1, ax1 = plt.subplots(ncols=2, nrows=2, figsize=[8,6], sharey=True)
fig1.subplots_adjust(wspace=0.2, hspace=0.4, left=0.1, right=0.9, top=0.9, bottom=0.2)
twax1 = [ax.twinx() for ax in ax1.flatten()]
for ax in ax1.flatten():
    ax.tick_params(axis='both', which='both', labelsize=8)
for ax in twax1:
    ax.tick_params(axis='both', which='both', labelsize=8)

lw = 2.0
titles = ('$\eta$', '$g$', '$J$')
for i, values in enumerate([eta_values, g_values, j_values]):
    ax1.flatten()[i].plot(values['steps'], values['training_loss'], color='r', linewidth=lw, label='Training loss')
    ax1.flatten()[i].plot(values['steps'], values['test_loss'], color='royalblue', linewidth=lw, label='Test loss')
    # ax1[i].plot(values['steps'], values['validation_loss'], color='royalblue', linewidth=lw)
    ax1.flatten()[i].set_ylim(0, 3.0)
    twax1[i].plot(values['steps'], values['test_accuracy'], color='orange', linewidth=lw, label='Test accuracy')
    twax1[i].set_ylim(0, 1)
    ax1.flatten()[i].set_title(titles[i], fontdict={'fontsize': 12})

    ax1.flatten()[i].set_xlim(0, values['steps'][-1])

handles1, labels1 = ax1[1,0].get_legend_handles_labels()
handles2, labels2 = twax1[1].get_legend_handles_labels()
handles = handles2 + handles1[::-1]
labels = labels2 + labels1[::-1]
fig1.legend(handles, labels, bbox_to_anchor=(0.8, 0.4))
ax1[1,0].set_xlabel('step', fontdict={'fontsize': 10})
ax1[0,1].set_xlabel('step', fontdict={'fontsize': 10})
ax1[0,0].set_ylabel('loss', fontdict={'fontsize': 10})
ax1[1,0].set_ylabel('loss', fontdict={'fontsize': 10})
twax1[1].set_ylabel('accuracy', fontdict={'fontsize': 10})
twax1[2].set_ylabel('accuracy', fontdict={'fontsize': 10})
ax1[1,1].axis('off')
twax1[-1].axis('off')


fig1.savefig('../plots/training_curves_classification', dpi=200, bbox_inches='tight')




1
