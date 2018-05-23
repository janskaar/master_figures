import nest
import matplotlib.pyplot as plt
import numpy as np

tauMem = 20.0
E_L = 0.0
CMem = 1.0
theta = 20.0
V_reset = 0.0
V_m = 0.0
delay = 1.5
t_ref = 2.0
CE = 1000

nest.ResetKernel()
nest.SetKernelStatus({'resolution':0.1, 'print_time':True})
neuron = nest.Create('iaf_psc_delta')
nest.SetStatus(neuron, {"C_m": CMem,
                        "tau_m": tauMem,
                        "t_ref": t_ref,
                        "E_L": E_L,
                        "V_reset": V_reset,
                        "V_m": V_m,
                        "V_th": theta,
                        })
multimeter = nest.Create('voltmeter')
e_spike_gen = nest.Create('spike_generator')
nest.SetStatus(e_spike_gen, {'spike_times': [10.]})
nest.Connect(e_spike_gen, neuron)
nest.Connect(multimeter, neuron)
nest.Simulate(111.)

Vm = nest.GetStatus(multimeter, 'events')[0]['V_m']
times = nest.GetStatus(multimeter, 'events')[0]['times']
times -= 10.

fig, ax = plt.subplots(1, figsize=[2,2])
fig.subplots_adjust(left=0.3, bottom=0.3)
ax.plot(times, Vm, 'black')
ax.set_xticks([0, 50, 100])
ax.set_yticks([0, 0.5, 1.0])
ax.tick_params(axis='both', which='major', labelsize=4)
ax.tick_params(axis='both', which='minor', labelsize=4)
ax.set_ylabel('mV', fontsize=4)
ax.set_xlabel('ms', fontsize=4)

fig.savefig('../plots/single_excitatory_spike', dpi=120)
