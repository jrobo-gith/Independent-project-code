import numpy as np
import matplotlib.pyplot as plt
import json
import matplotlib as mpl

from scipy.integrate import solve_ivp
from glob_var.FVM.FVM_RHS import FVM_RHS
from glob_var.FVM.stop_events import unstable, steady_state
from non_newtonian_thin_film_solve.individual_files.power_law_startup import make_step as PL_make_step

try:
    with open('../../glob_var/global_variables.json') as f:
        GV = json.load(f)
except FileNotFoundError:
    print("Global variables json not found!")

unstable.terminal = True
steady_state.terminal = True

h_initial = np.ones(GV['N']) * GV['h0']
t_span = GV['t-span']
A = 0.0

print("Newtonian")
n = 1.0
args = [PL_make_step, GV['dx'], None, GV['Q'], A, n, False, GV['N']]
newt_sol = solve_ivp(fun=FVM_RHS, y0=h_initial, t_span=t_span, args=(args,), method='BDF', rtol=1e-6, atol=1e-8, events=[unstable, steady_state])

print("Thinning")
n = 0.8
args = [PL_make_step, GV['dx'], None, GV['Q'], A, n, False, GV['N']]
thin_sol = solve_ivp(fun=FVM_RHS, y0=h_initial, t_span=t_span, args=(args,), method='BDF', rtol=1e-6, atol=1e-8, events=[unstable, steady_state])

print("Thickening")
n = 1.2
args = [PL_make_step, GV['dx'], None, GV['Q'], A, n, False, GV['N']]
thic_sol = solve_ivp(fun=FVM_RHS, y0=h_initial, t_span=t_span, args=(args,), method='BDF', rtol=1e-6, atol=1e-8, events=[unstable, steady_state])

print("Plotting")
max_t = max([newt_sol.t[-1], thin_sol.t[-1], thic_sol.t[-1]])

c = np.arange(0, max_t)
norm = mpl.colors.Normalize(vmin=c.min(), vmax=c.max())
cmap = mpl.cm.ScalarMappable(norm=norm, cmap=GV['cmap'])
cmap.set_array([])

fig, ax = plt.subplots(nrows=3, figsize=(15, 15))
fig.subplots_adjust(hspace=0.0)

num_intervals = 5
rgb_step = int(max_t/num_intervals)

LW = 2.5

def plot_lines(axis, sol):
    step = int(len(sol.t)/num_intervals)
    intermediate_array = sol.y[:, ::step]
    print(intermediate_array.shape)
    axis.plot(GV['x'], sol.y[:, 0], c=cmap.to_rgba(0.01), linewidth=LW)
    for i in range(1, num_intervals):
        axis.plot(GV['x'], intermediate_array[:, i], c=cmap.to_rgba(i * rgb_step), linewidth=LW)
    axis.plot(GV['x'], sol.y[:, -1], c=cmap.to_rgba(sol.t[-1]), linewidth=LW)

plot_lines(ax[0], newt_sol)
plot_lines(ax[1], thin_sol)
plot_lines(ax[2], thic_sol)

ax[0].grid(True)
ax[0].set_ylabel("NEWTONIAN\n\nFilm Height $(y)$", fontsize=14)
ax[0].set_xticks(np.arange(0, GV['L'] + 1, 2))
ax[0].set_xticklabels([])
ax[0].set_yticks(np.arange(0, GV['h0'], 0.1))
ax[0].tick_params(axis='y', labelsize=14)

ax[1].grid(True)
ax[1].set_ylabel("SHEAR-THINNING\n\nFilm Height $(y)$", fontsize=14)
ax[1].set_xticks(np.arange(0, GV['L'] + 1, 2))
ax[1].set_xticklabels([])
ax[1].set_yticks(np.arange(0, GV['h0'], 0.1))
ax[1].tick_params(axis='y', labelsize=14)

ax[2].grid(True)
ax[2].set_ylabel("SHEAR-THICKENING\n\nFilm Height $(y)$", fontsize=14)
ax[2].set_xticks(np.arange(0, GV['L'] + 1, 2))
ax[2].set_xlabel("Surface Length $(x)$", fontsize=14)
ax[2].set_yticks(np.arange(0, GV['h0'], 0.1))
ax[2].tick_params(axis='y', labelsize=14)
ax[2].tick_params(axis='x', labelsize=14)

cbar_ax = fig.add_axes([0.92, 0.11, 0.03, 0.77])
cb = fig.colorbar(cmap, cax=cbar_ax, ticks=np.arange(0, max_t + 1, 5))
cb.set_label("Time (s)", fontsize=16)
cb.ax.tick_params(labelsize=14)

fig.suptitle(f"Timeseries graph showing evolution of three rheologies in time", fontsize=18, y=0.92)

fig.savefig("no_DP.png")