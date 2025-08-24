import numpy as np
import matplotlib.pyplot as plt
import json
import matplotlib as mpl

try:
    with open('../../../glob_var/global_variables.json') as f:
        GV = json.load(f)
except FileNotFoundError:
    print("Global variables json not found!")

term = "non-linear"

newt = np.load("newtonian_collector/NLT.npy")[:, :, 1:298]
newt_dp = np.load("newtonian_collector/NLT_dp.npy")[:, :, 1:298]

thin = np.load("thinning_collector/NLT.npy", allow_pickle=True)[:, :, 1:298]
thin_dp = np.load("thinning_collector/NLT_dp.npy")[:, :, 1:298]

thic = np.load("thickening_collector/NLT.npy")[:, :, 1:298]
thic_dp = np.load("thickening_collector/NLT_dp.npy")[:, :, 1:298]

T_newt = np.load("newtonian_collector/T.npy")
T_newt_dp = np.load("newtonian_collector/T_dp.npy")

T_thin = np.load("thinning_collector/T.npy")
T_thin_dp = np.load("thinning_collector/T_dp.npy")

T_thic = np.load("thickening_collector/T.npy")
T_thic_dp = np.load("thickening_collector/T_dp.npy")

## COMPUTE q for each
newt_diff = ((newt[:, 1, :] - newt[:, 0, :]) / GV['dx']).transpose(1, 0)
newt_diff_dp = ((newt_dp[:, 1, :] - newt_dp[:, 0, :]) / GV['dx']).transpose(1, 0)

thin_diff = ((thin[:, 1, :] - thin[:, 0, :]) / GV['dx']).transpose(1, 0)
thin_diff_dp = ((thin_dp[:, 1, :] - thin_dp[:, 0, :]) / GV['dx']).transpose(1, 0)

thic_diff = ((thic[:, 1, :] - thic[:, 0, :]) / GV['dx']).transpose(1, 0)
thic_diff_dp = ((thic_dp[:, 1, :] - thic_dp[:, 0, :]) / GV['dx']).transpose(1, 0)

max_t = max([T_newt[-1], T_thin[-1], T_thic[-1], T_newt_dp[-1], T_thin_dp[-1], T_thic_dp[-1]])
c = np.arange(0, max_t)
norm = mpl.colors.Normalize(vmin=c.min(), vmax=c.max())
cmap = mpl.cm.ScalarMappable(norm=norm, cmap=GV['cmap'])
cmap.set_array([])

fig, ax = plt.subplots(nrows=3, ncols=2, figsize=(15, 15))
fig.subplots_adjust(hspace=0.1)

num_intervals = 5

LW = 2.5

def plot_lines(axis, sol_y, sol_t):
    step = int(len(sol_t)/num_intervals)
    rgb_step = int(max(sol_t) / num_intervals)
    intermediate_array = sol_y[:, ::step]
    axis.plot(GV['x'][1:298], sol_y[:, 0], c=cmap.to_rgba(0.01), linewidth=LW)
    for i in range(1, num_intervals):
        axis.plot(GV['x'][1:298], intermediate_array[:, i], c=cmap.to_rgba(i * rgb_step), linewidth=LW)
    axis.plot(GV['x'][1:298], sol_y[:, -1], c=cmap.to_rgba(sol_t[-1]), linewidth=LW)

plot_lines(ax[0, 0], newt_diff, T_newt)
plot_lines(ax[0, 1], newt_diff_dp, T_newt_dp)

plot_lines(ax[1, 0], thin_diff, T_thin)
plot_lines(ax[1, 1], thin_diff_dp, T_thin_dp)

plot_lines(ax[2, 0], thic_diff, T_thic)
plot_lines(ax[2, 1], thic_diff_dp, T_thic_dp)

ax[0, 0].grid(True)
ax[0, 0].set_ylabel("Newtonian Fluid no DP", fontsize=14)
ax[0, 0].set_xticks(np.arange(0, GV['L'] + 1, 2))
ax[0, 0].set_xticklabels([])
# ax[0, 0].set_yticks(np.arange(0, GV['h0']+0.1, 0.1))
ax[0, 0].tick_params(axis='y', labelsize=14)
ax[0, 0].set_title("(A)", fontsize=16)

ax[0, 1].grid(True)
ax[0, 1].set_ylabel("Newtonian Fluid, $DP=0.15$", fontsize=14)
ax[0, 1].set_xticks(np.arange(0, GV['L'] + 1, 2))
ax[0, 1].set_xticklabels([])
# ax[0, 1].set_yticks(np.arange(0, GV['h0']+0.1, 0.1))
ax[0, 1].tick_params(axis='y', labelsize=14)
ax[0, 1].set_title("(B)", fontsize=16)

ax[1, 0].grid(True)
ax[1, 0].set_ylabel("Shear-thinning Fluid no DP", fontsize=14)
ax[1, 0].set_xticks(np.arange(0, GV['L'] + 1, 2))
ax[1, 0].set_xticklabels([])
# ax[1, 0].set_yticks(np.arange(0, GV['h0']+0.1, 0.1))
ax[1, 0].tick_params(axis='y', labelsize=14)

ax[1, 1].grid(True)
ax[1, 1].set_ylabel("Shear-thinning Fluid $DP=0.15$", fontsize=14)
ax[1, 1].set_xticks(np.arange(0, GV['L'] + 1, 2))
ax[1, 1].set_xticklabels([])
# ax[1, 1].set_yticks(np.arange(0, GV['h0']+0.1, 0.1))
ax[1, 1].tick_params(axis='y', labelsize=14)

ax[2, 0].grid(True)
ax[2, 0].set_ylabel("Shear-thickening Fluid no DP", fontsize=14)
ax[2, 0].set_xticks(np.arange(0, GV['L'] + 1, 2))
ax[2, 0].set_xlabel("Surface Length $(x)$", fontsize=14)
# ax[2, 0].set_yticks(np.arange(0, GV['h0']+0.1, 0.1))
ax[2, 0].tick_params(axis='y', labelsize=14)
ax[2, 0].tick_params(axis='x', labelsize=14)

ax[2, 1].grid(True)
ax[2, 1].set_ylabel("Shear-thickening Fluid $DP=0.15$", fontsize=14)
ax[2, 1].set_xticks(np.arange(0, GV['L'] + 1, 2))
ax[2, 1].set_xlabel("Surface Length $(x)$", fontsize=14)
# ax[2, 1].set_yticks(np.arange(0, GV['h0']+0.1, 0.1))
ax[2, 1].tick_params(axis='y', labelsize=14)
ax[2, 1].tick_params(axis='x', labelsize=14)

ax[0, 0].set_ylim(-1, 0.2)
ax[1, 0].set_ylim(-1, 0.2)
ax[2, 0].set_ylim(-1, 0.2)

ax[0, 1].set_ylim(-0.8, 0.5)
ax[1, 1].set_ylim(-0.8, 0.5)
ax[2, 1].set_ylim(-0.8, 0.5)

fig.text(0.02, 0.5, "Film Height $(h)$", rotation='vertical', fontsize=16, va='center', ha='center')

cbar_ax = fig.add_axes([0.92, 0.11, 0.03, 0.77])
cb = fig.colorbar(cmap, cax=cbar_ax, ticks=np.arange(0, max_t + 1, 5))
cb.set_label("Time (s)", fontsize=16)
cb.ax.tick_params(labelsize=14)

fig.suptitle(f"Timeseries graph showing evolution of the {term} term in time", fontsize=18, y=0.92)

fig.savefig("graphs/NLT_diff.png", bbox_inches='tight')