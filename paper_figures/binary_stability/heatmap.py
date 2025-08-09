import numpy as np
import matplotlib.pyplot as plt

def plot_binary(data):
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
    c = ax.pcolormesh(X, Y, Z, cmap='winter')
    fig.colorbar(c, ax=ax, label="Binary Stability")
    ax.set_ylabel("Volume Flux $(Q)$")
    ax.set_xlabel("Disjoint Pressure Strength $(A)$")
    ax.set_title("Heatmap of Binary Stability")
    fig.savefig("binary_stability_newtonian.png")

newtonian_data = np.load("newtonian.npy")
plot_binary(newtonian_data)
