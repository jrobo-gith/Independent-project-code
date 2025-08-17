import numpy as np
import matplotlib.pyplot as plt
import json
from glob_var.deviation_from_steady_state import magnitude_of_deviation as mod
import matplotlib as mpl

try:
    with open('../../glob_var/global_variables.json') as f:
        GV = json.load(f)
except FileNotFoundError:
    print("Global variables json not found!")

def num_to_range(num, inMin, inMax):
  return (float(num - inMin) / float(inMax - inMin))

def calculate_deformation(data, steady_states_Q):
    errors = np.zeros((data.shape[0], data.shape[1]))
    for i in range(data.shape[0]):
        for j in range(data.shape[1]):
            errors[i, j] = mod(steady_state=steady_states_Q[i], startup_flow=data[i, j, :], scalar=True)

    Z = np.zeros((data.shape[0], data.shape[1]))
    min_error = min(errors.flatten())
    max_error = max(errors.flatten())
    for i in range(data.shape[0]):
        for j in range(data.shape[1]):
            Z[i, j] = num_to_range(errors[i, j], min_error, max_error)

    return Z

newtonian = np.load("newtonian.npy")
thinning = np.load("thinning.npy")
thickening = np.load("thickening.npy")

newtonian_SS = np.load("../steady_state_all_rheologies/data/newtonian_ss.npy")
thinning_SS = np.load("../steady_state_all_rheologies/data/thinning_SS.npy")
thickening_SS = np.load("../steady_state_all_rheologies/data/thickening_SS.npy")

newt_Z = calculate_deformation(newtonian, newtonian_SS)
thin_Z = calculate_deformation(thinning, thinning_SS)
thic_Z = calculate_deformation(newtonian, thickening_SS)

assert newt_Z.shape == thin_Z.shape == thic_Z.shape, print("Shapes are not equal!", newt_Z.shape, thin_Z.shape, thic_Z.shape)

x = np.linspace(0, 0.5, newt_Z.shape[0])
y = np.linspace(0.1, 0.95, newt_Z.shape[1])
X, Y = np.meshgrid(x, y)

fig, ax = plt.subplots(ncols=3, figsize=(15, 10))
plt.subplots_adjust(wspace=0.05)

ax[0].pcolormesh(X, Y, newt_Z, cmap=GV['cmap'])
ax[0].set_title("Newtonian Fluid\n$t=29$", fontsize=16)
ax[0].set_ylabel("Volume Flux $(Q)$", fontsize=14)

ax[1].pcolormesh(X, Y, thin_Z, cmap=GV['cmap'])
ax[1].set_yticklabels([])
ax[1].set_title("Shear-thinning Fluid\n$t=28$", fontsize=16)

ax[2].pcolormesh(X, Y, thic_Z, cmap=GV['cmap'])
ax[2].set_yticklabels([])
ax[2].set_title("Shear-thickening Fluid\n$t=39$", fontsize=16)

fig.text(0.5, 0.05, "Disjoint Pressure Strength $(A)$", ha="center", fontsize=14, rotation='horizontal')

c = np.arange(0, 1.1, 0.1)
norm = mpl.colors.Normalize(vmin=c.min(), vmax=c.max())
cmap = mpl.cm.ScalarMappable(norm=norm, cmap=GV['cmap'])
cmap.set_array([])

cbar_ax = fig.add_axes([0.92, 0.11, 0.03, 0.77])
cb = fig.colorbar(cmap, cax=cbar_ax, ticks=np.arange(0, 1.1, 0.1))
cb.set_label("Deformation Strength", fontsize=14)
cb.ax.tick_params(labelsize=14)

fig.savefig("clumped_heatmap.png", bbox_inches='tight')
