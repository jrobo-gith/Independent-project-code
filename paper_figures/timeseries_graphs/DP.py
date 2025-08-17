import numpy as np
import matplotlib.pyplot as plt
import json
import matplotlib as mpl

A = 0.15

try:
    with open('../../glob_var/global_variables.json') as f:
        GV = json.load(f)
except FileNotFoundError:
    print("Global variables json not found!")

newt_sol_y = np.load("../term_tracker/newtonian_collector/sol_y_dp.npy")
thin_sol_y = np.load("../term_tracker/thinning_collector/sol_y_dp.npy")
thic_sol_y = np.load("../term_tracker/thickening_collector/sol_y_dp.npy")

newt_sol_t = np.load("../term_tracker/newtonian_collector/T_dp.npy")
thin_sol_t = np.load("../term_tracker/thinning_collector/T_dp.npy")
thic_sol_t = np.load("../term_tracker/thickening_collector/T_dp.npy")

max_t = max([newt_sol_t[-1], thin_sol_t[-1], thic_sol_t[-1]])

c = np.arange(0, max_t)
norm = mpl.colors.Normalize(vmin=c.min(), vmax=c.max())
cmap = mpl.cm.ScalarMappable(norm=norm, cmap=GV['cmap'])
cmap.set_array([])

fig, ax = plt.subplots(nrows=3, figsize=(15, 15))
fig.subplots_adjust(hspace=0.05)

num_intervals = 5

LW = 2.5

def plot_lines(axis, sol_y, sol_t):
    rgb_step = int(max(sol_t) / num_intervals)
    step = int(sol_y.shape[1]/num_intervals)
    intermediate_array = sol_y[:, ::step]
    axis.plot(GV['x'], sol_y[:, 0], c=cmap.to_rgba(0.01), linewidth=LW)
    for i in range(1, num_intervals):
        axis.plot(GV['x'], intermediate_array[:, i], c=cmap.to_rgba(i * rgb_step), linewidth=LW)
    axis.plot(GV['x'], sol_y[:, -1], c=cmap.to_rgba(sol_t[-1]), linewidth=LW)

plot_lines(ax[0], newt_sol_y, newt_sol_t)
plot_lines(ax[1], thin_sol_y, thin_sol_t)
plot_lines(ax[2], thic_sol_y, thic_sol_t)

ax[0].grid(True)
ax[0].set_ylabel("Newtonian Fluid", fontsize=14)
ax[0].set_xticks(np.arange(0, GV['L'] + 1, 2))
ax[0].set_xticklabels([])
ax[0].set_yticks(np.arange(0, GV['h0']+0.1, 0.1))
ax[0].tick_params(axis='y', labelsize=14)

ax[1].grid(True)
ax[1].set_ylabel("Shear-thinning Fluid", fontsize=14)
ax[1].set_xticks(np.arange(0, GV['L'] + 1, 2))
ax[1].set_xticklabels([])
ax[1].set_yticks(np.arange(0, GV['h0']+0.1, 0.1))
ax[1].tick_params(axis='y', labelsize=14)

ax[2].grid(True)
ax[2].set_ylabel("Shear-thickening Fluid", fontsize=14)
ax[2].set_xticks(np.arange(0, GV['L'] + 1, 2))
ax[2].set_xlabel("Surface Length $(x)$", fontsize=14)
ax[2].set_yticks(np.arange(0, GV['h0']+0.1, 0.1))
ax[2].tick_params(axis='y', labelsize=14)
ax[2].tick_params(axis='x', labelsize=14)

fig.text(0.05, 0.5, "Film Height $(h)$", rotation='vertical', fontsize=16, va='center', ha='center')

cbar_ax = fig.add_axes([0.92, 0.11, 0.03, 0.77])
cb = fig.colorbar(cmap, cax=cbar_ax, ticks=np.arange(0, max_t + 1, 5))
cb.set_label("Time (s)", fontsize=16)
cb.ax.tick_params(labelsize=14)

fig.suptitle(f"Timeseries graph showing evolution of three rheologies in time\nwith disjoining pressure $(A={A})$", fontsize=18, y=0.92)
fig.savefig("DP.png", bbox_inches='tight')