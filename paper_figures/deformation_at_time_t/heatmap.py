import numpy as np
from scipy.integrate import solve_ivp
import matplotlib.pyplot as plt
import json
from glob_var.deviation_from_steady_state import magnitude_of_deviation as mod
from glob_var.FVM.stop_events import unstable, steady_state
from glob_var.FVM.FVM_RHS import FVM_RHS
from non_newtonian_thin_film_solve.individual_files.power_law_dp import make_step as PL_DP_make_step

try:
    with open('../../glob_var/global_variables.json') as f:
        GV = json.load(f)
except FileNotFoundError:
    print("Global variables json not found!")

def num_to_range(num, inMin, inMax):
  return (float(num - inMin) / float(inMax - inMin))

def plot_deformation(data, directory, title, steady_states_Q, t):
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

    x = np.linspace(0, 0.5, Z.shape[0])
    y = np.linspace(0.1, 0.95, Z.shape[1])
    X, Y = np.meshgrid(x, y)
    fig, ax = plt.subplots(figsize=(10, 10))
    c = ax.pcolormesh(X, Y, Z, cmap=GV['cmap'])
    fig.colorbar(c, ax=ax, label="Deformation Strength")
    ax.set_ylabel("Volume Flux $(Q)$", fontsize=14)
    ax.set_xlabel("Disjoint Pressure Strength $(A)$", fontsize=14)
    ax.set_title(f"Heatmap of Deformation at time {t}\nwith a " + title + " rheology", fontsize=16)
    ax.set_xticks(np.arange(0, x[-1]+0.1, 0.1))
    ax.set_yticks(np.arange(y[0], y[-1], 0.1))
    fig.savefig(directory)

newtonian = np.load("newtonian.npy")
thinning = np.load("thinning.npy")
thickening = np.load("thickening.npy")

newtonian_SS = np.load("../steady_state_all_rheologies/data/newtonian_ss.npy")

plot_deformation(data=newtonian, directory="deformation_newtonian.png", title="Newtonian", steady_states_Q=newtonian_SS, t=39)
# plot_deformation(data=thinning, directory="deformation_thinning.png", title="Thinning", steady_state=thinning_SS)
# plot_deformation(data=thickening, directory="deformation_thickening.png", title="Thickening", steady_state=thickening_SS)
