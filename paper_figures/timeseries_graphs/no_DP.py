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

print("Newtonian")
n = 1.0
args = [PL_make_step, GV['dx'], None, GV['Q'], None, n, False, GV['N']]
newt_sol = solve_ivp(fun=FVM_RHS, y0=h_initial, t_span=t_span, args=(args,), method='BDF', rtol=1e-6, atol=1e-8, events=[unstable, steady_state])

print("Thinning")
n = 0.8
args = [PL_make_step, GV['dx'], None, GV['Q'], None, n, False, GV['N']]
thin_sol = solve_ivp(fun=FVM_RHS, y0=h_initial, t_span=t_span, args=(args,), method='BDF', rtol=1e-6, atol=1e-8, events=[unstable, steady_state])

print("Thickening")
n = 1.2
args = [PL_make_step, GV['dx'], None, GV['Q'], None, n, False, GV['N']]
thic_sol = solve_ivp(fun=FVM_RHS, y0=h_initial, t_span=t_span, args=(args,), method='BDF', rtol=1e-6, atol=1e-8, events=[unstable, steady_state])

print("Plotting")
max_t = max([newt_sol.t[-1], thin_sol.t[-1], thic_sol.t[-1]])

c = np.arange(0, max_t)
norm = mpl.colors.Normalize(vmin=c.min(), vmax=c.max())
cmap = mpl.cm.ScalarMappable(norm=norm, cmap=mpl.cm.cool)
cmap.set_array([])

fig, ax = plt.subplots(ncols=3, figsize=(10, 10))

num_intervals = 5
step = int(len(thic_sol.t)/num_intervals)
rgb_step = int(max_t/num_intervals)

LW = 2.5

def plot_lines(axis, sol):
    intermediate_array = sol.y[:, ::step]
    axis.plot(GV['x'], sol.y[:, 0], c=cmap.to_rgba(0.01), linewidth=LW)
    for i in range(1, num_intervals):
        axis.plot(GV['x'], intermediate_array[i], c=cmap.to_rgba(i * rgb_step), linewidth=LW)
    axis.plot(GV['x'], sol.y[:, -1], c=cmap.to_rgba(sol.t[-1]), linewidth=LW)

plot_lines(ax[0], newt_sol)
plot_lines(ax[1], thin_sol)
plot_lines(ax[2], thic_sol)

fig.colorbar(cmap, ticks=np.arange(0, max_t, 10), label="Test")
ax[0].grid(True)
ax[0].set_xlabel("Surface Length $(x)$")
ax[0].set_ylabel("Film Height $(y)$")

ax[1].grid(True)
ax[1].set_xlabel("Surface Length $(x)$")
ax[1].set_ylabel("Film Height $(y)$")

ax[2].grid(True)
ax[2].set_xlabel("Surface Length $(x)$")
ax[2].set_ylabel("Film Height $(y)$")

ax[0].set_title("NEWTONIAN")
ax[0].set_title("SHEAR-THINNING")
ax[0].set_title("SHEAR-THICKENING")
fig.savefig("no_DP.png")