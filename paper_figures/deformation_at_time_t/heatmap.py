import numpy as np
import matplotlib.pyplot as plt

def plot_deformation(x, y, Z):

    X, Y = np.meshgrid(x, y)
    fig, ax = plt.subplots(figsize=(10, 10))
    c = ax.pcolormesh(X, Y, Z, cmap='winter')
    fig.colorbar(c, ax=ax, label="Deformation at time t")
    ax.set_xlabel("Volume Flux $(Q)$")
    ax.set_ylabel("Disjoint Pressure Strength $(A)$")
    ax.set_title("Heatmap of ")
    fig.show()