import numpy as np
import matplotlib.pyplot as plt
import json

try:
    with open('../../glob_var/global_variables.json') as f:
        GV = json.load(f)
except FileNotFoundError:
    print("Global variables json not found!")

def plot_binary(data, directory, title):
    Z = np.zeros((data.shape[0], data.shape[1]))
    for i in range(data.shape[0]):
        for j in range(data.shape[1]):
            if np.any(data[i, j, :] < 0.05):
                # Unstable
                Z[i, j] = 0
            else:
                # Stable
                Z[i, j] = 1

    x = np.linspace(0, 0.5, Z.shape[0])
    y = np.linspace(0.1, 0.95, Z.shape[1])
    X, Y = np.meshgrid(x, y)
    fig, ax = plt.subplots(figsize=(10, 10))
    c = ax.pcolormesh(X, Y, Z, cmap=GV['cmap'])
    fig.colorbar(c, ax=ax, label="Stable/Unstable")
    ax.set_ylabel("Volume Flux $(Q)$", fontsize=14)
    ax.set_xlabel("Disjoint Pressure Strength $(A)$", fontsize=14)
    ax.set_title("Heatmap of Binary Stability with a " + title + " rheology", fontsize=16)
    ax.set_xticks(np.arange(0, x[-1]+0.1, 0.1))
    ax.set_yticks(np.arange(y[0], y[-1], 0.1))
    fig.savefig(directory)

newtonian_data = np.load("newtonian.npy")
plot_binary(newtonian_data, "binary_stability_newtonian.png", "Newtonian")

thinning_data = np.load("thinning.npy")
plot_binary(thinning_data, "binary_stability_thinning.png", "Shear-Thinning")

thickening_data = np.load("thickening.npy")
plot_binary(thickening_data, "binary_stability_thickening.png", "Shear-Thickening")
