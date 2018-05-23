from example_plotting import plot_individual_morphologies
import os, LFPy
import matplotlib.pyplot as plt
from matplotlib.collections import PolyCollection
import numpy as np

stretched_ex = os.path.join('./morphologies/L4E_53rpy1_cut.hoc')
stretched_in = os.path.join('./morphologies/L4I_oi26rbc1.hoc')

scellParams = dict(
    #excitory cells
    EX = dict(
        morphology = stretched_ex
    ),
    #inhibitory cells
    IN = dict(
        morphology = stretched_in
        ))

populationParams = dict(
    EX = dict(
        z_min = -450,
        z_max = -350
        ),

    IN = dict(
        z_min = -450,
        z_max = -350
        ))

r = np.sqrt(1000**2/np.pi)

icell = LFPy.Cell(morphology=stretched_in)
icell.set_pos(-200, 0, -400)

ecell = LFPy.Cell(morphology=stretched_ex)
ecell.set_pos(200, 0, -400)
ecell.set_rotation(z=-0.05*np.pi)


iverts = []
for x, z in icell.get_idx_polygons():
    iverts.append(list(zip(x, z )))

everts = []
for x, z in ecell.get_idx_polygons():
    everts.append(list(zip(x, z)))


ipoly = PolyCollection(iverts, facecolors='black')
epoly = PolyCollection(everts, facecolors='black')

ax = plt.gca()
fig = plt.gcf()
fig.set_size_inches([5, 8])

t1 = np.linspace(-np.pi, 0, 50)
t2 = np.linspace(0, np.pi, 50)

# ax.plot(r*np.cos(t1), 20*np.sin(t1) - 450, color='black', linewidth=0.7)
# ax.plot(r*np.cos(t2), 20*np.sin(t2) - 450, ':', color='black', linewidth=0.7)
#
# ax.plot(r*np.cos(t1), 20*np.sin(t1) - 350, color='black', linewidth=0.5)
# ax.plot(r*np.cos(t2), 20*np.sin(t2) - 350, color='black', linewidth=0.5)

ax.plot([-r, -r], [-450, -350], color='black', linewidth=0.5)
ax.plot([r, r], [-450, -350], color='black', linewidth=0.5)

# ax.plot([r, 270], [-350, -350], '--', color='black', linewidth=0.5)
# ax.plot([r, 270], [-450, -450], '--', color='black', linewidth=0.5)
#
# ax.text(280, -355, 'z = -350 $\mathrm{\mu m}$')
# ax.text(280, -455, 'z = -450 $\mathrm{\mu m}$')

ax.plot([-r, 0], [-500, -500], '--', color='black', linewidth=0.5)
ax.plot([-400, -280], [-540, -500], color='black', linewidth=0.5)

ax.text(-480, -560, 'r = %.0f $\mathrm{\mu m}$'%r)

ax.plot(r*np.cos(t1), 25*np.sin(t1) - 500, color='black', linewidth=0.5)
ax.plot(r*np.cos(t2), 25*np.sin(t2) - 500, ':', color='black', linewidth=0.5)
ax.plot(r*np.cos(t1), 25*np.sin(t1) - 300, color='black', linewidth=0.5)
ax.plot(r*np.cos(t2), 25*np.sin(t2) - 300, ':', color='black', linewidth=0.5)
ax.plot([-r, -r], [-500, -300], color='black', linewidth=0.5)
ax.plot([r, r], [-500, -300], color='black', linewidth=0.5)

ax.plot(r*np.cos(t1), 25*np.sin(t1) - 0, color='black', linewidth=0.5)
ax.plot(r*np.cos(t2), 25*np.sin(t2) - 0, color='black', linewidth=0.5)
ax.plot([-r, -r], [0, -300], color='black', linewidth=0.5)
ax.plot([r, r], [0, -300], color='black', linewidth=0.5)





ax.text(-r-350, -310, 'z = -300 $\mathrm{\mu m}$')
ax.text(-r-350, -510, 'z = -500 $\mathrm{\mu m}$')
ax.text(-r-280, -5, 'z = 0 $\mathrm{\mu m}$')

elec_z = [0, -100, -200, -300, -400, -500]
elec_x = [0, 0, 0, 0, 0, 0]

ax.scatter(elec_x, elec_z, marker='o', color='black', s=10)

ax.plot([-20, -20], [0, -100], '--', color='black', linewidth=0.5)
ax.plot([-190, -20], [-70, -50], color='black', linewidth=0.5)
ax.plot([-20, -3], [0, 0], '--', color='black', linewidth=0.5)
ax.plot([-20, -3], [-100, -100], '--', color='black', linewidth=0.5)

ax.text(-380, -90, 'd = 100 $\mathrm{\mu m}$')

ax.add_collection(ipoly)
ax.add_collection(epoly)
ax.set_xlim(-r-200,r+200)
ax.set_ylim(-580, 70)
ax.set_xticks([])
ax.set_yticks([])
ax.axis('off')

# plt.show()
fig.savefig('../plots/morphologies_column', dpi=120, bbox_inches='tight')
