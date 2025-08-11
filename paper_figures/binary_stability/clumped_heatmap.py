import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
import json

try:
    with open('../../glob_var/global_variables.json') as f:
        GV = json.load(f)
except FileNotFoundError:
    print("Global variables json not found!")

def compute_Z(data):
    Z = np.zeros((data.shape[0], data.shape[1]))
    for i in range(data.shape[0]):
        for j in range(data.shape[1]):
            if np.any(data[i, j, :] < 0.05):
                # Unstable
                Z[i, j] = 0
            else:
                # Stable
                Z[i, j] = 1

    return Z

newtonian_data = np.load("newtonian.npy")
thinning_data = np.load("thinning.npy")
thickening_data = np.load("thickening.npy")

newt_Z = compute_Z(newtonian_data)
thin_Z = compute_Z(thinning_data)
thic_Z = compute_Z(thickening_data)

assert newt_Z.shape == thin_Z.shape == thic_Z.shape, print("Z shapes are not equal!", newt_Z.shape, thin_Z.shape, thic_Z.shape)
x = np.linspace(0, 0.5, newt_Z.shape[0])
y = np.linspace(0.1, 0.95, newt_Z.shape[1])
X, Y = np.meshgrid(x, y)


fig, ax = plt.subplots(ncols=3, figsize=(15, 10))
plt.subplots_adjust(wspace=0.05)

ax[0].pcolormesh(X, Y, newt_Z, cmap=GV['cmap'])
ax[0].set_title("Newtonian Fluid", fontsize=16)
ax[0].set_ylabel("Volume Flux $(Q)$", fontsize=14)

ax[1].pcolormesh(X, Y, thin_Z, cmap=GV['cmap'])
ax[1].set_title("Shear-thinning Fluid", fontsize=16)
ax[1].set_yticks([])

ax[2].pcolormesh(X, Y, thic_Z, cmap=GV['cmap'])
ax[2].set_title("Shear-thickening Fluid", fontsize=16)
ax[2].set_yticks([])

fig.text(0.5, 0.05, "Disjoint Pressure Strength $(A)$", ha="center", fontsize=14, rotation='horizontal')

c = np.arange(0, 1.1, 0.1)
norm = mpl.colors.Normalize(vmin=c.min(), vmax=c.max())
cmap = mpl.cm.ScalarMappable(norm=norm, cmap=GV['cmap'])
cmap.set_array([])

cbar_ax = fig.add_axes([0.92, 0.11, 0.03, 0.77])
cb = fig.colorbar(cmap, cax=cbar_ax, ticks=np.arange(0, 1.1, 0.1))
cb.set_label("Stability", fontsize=14)
cb.ax.tick_params(labelsize=14)

fig.savefig("clumped_heatmap.png", bbox_inches='tight')
