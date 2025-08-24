import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import matplotlib as mpl


def make_heatmap(x, y, Z, dir):

    X, Y = np.meshgrid(x, y)

    fig, ax = plt.subplots(nrows=3, figsize=(12,12))
    fig.subplots_adjust(hspace=0.1)

    axes = []
    [axes.append(ax[i]) for i in range(len(ax))]

    norm = mpl.colors.Normalize(vmin=np.min(Z), vmax=np.max(Z[1]))
    cbar = fig.colorbar(cm.ScalarMappable(norm=norm, cmap='plasma_r'), ax=axes)
    cbar.set_label("Time $(s)$", fontsize=14, rotation="vertical")
    
    for i, PLOT in enumerate(axes):
        PLOT.pcolormesh(X, Y, Z[i], cmap='plasma_r', vmin=np.min(Z[1]), vmax=np.max(Z[1]))
        PLOT.set_xlabel("Disjoint Pressure Strength $(A)$", fontsize=14)
        PLOT.set_ylabel("Volume Flux $(Q)$", fontsize=14)
        if i != len(axes)-1:
            PLOT.set_xticklabels([])
            PLOT.set_xlabel("")

    fig.text(0.02, 0.72, "Newtonian", rotation="vertical", fontsize=16)
    fig.text(0.02, 0.43, "Shear-thinning", rotation="vertical", fontsize=16)
    fig.text(0.02, 0.16, "Shear-thickening", rotation="vertical", fontsize=16)

    fig.savefig(f"{dir}.png", bbox_inches="tight")