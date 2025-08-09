##### Used [gridspec](https://matplotlib.org/stable/api/_as_gen/matplotlib.gridspec.GridSpec.html) when making these types of plots

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import json
from scipy.integrate import solve_ivp
from non_newtonian_thin_film_solve.individual_files.power_law_startup import make_step as PL_make_step
from glob_var.FVM.stop_events import unstable, steady_state
from glob_var.FVM.FVM_RHS import FVM_RHS
from glob_var.deviation_from_steady_state import magnitude_of_deviation as mod


try:
    with open('../../glob_var/global_variables.json') as f:
        GV = json.load(f)
except FileNotFoundError:
    print("Global variables json not found!")

t_span = GV['t-span']

## CALCULATE NEWTONIAN STEADY STATE
# print("Calculating steady state...")
# 
# linear_solution_Q_steady = []  # Store linear solutions varying Q
# non_linear_solution_Q_steady = []  # Store non-linear solutions varying Q
# 
# for i in range(len(GV['Q-list'])):
#     linear_solution_Q_steady.append(bdf_solver_N(q=GV['Q-list'][i], L=GV['L'], linear=True).y[0])
#     non_linear_solution_Q_steady.append(bdf_solver_N(q=GV['Q-list'][i], L=GV['L'], linear=False).y[0])

## CALCULATE STARTUP FLOW
print("Calculating startup flow...")

linear_solution_Q = []  # Store linear solutions varying Q
non_linear_solution_Q = []  # Store non-linear solutions varying Q

for i in range(len(GV['Q-list'])):
    print(f"Q = {GV['Q-list'][i]}")
    h_initial = np.ones(GV['N']) * GV['h0']

    n = 0.8

    args = [PL_make_step, GV['dx'], None, GV['Q-list'][i], None, n, True, GV['N']]
    linear_solution_Q.append(
        solve_ivp(fun=FVM_RHS, args=(args,), y0=h_initial, t_span=t_span, method='BDF', rtol=1e-6, atol=1e-8,
                  events=[unstable, steady_state]).y[:, -1])

    args = [PL_make_step, GV['dx'], None, GV['Q-list'][i], None, n, False, GV['N']]
    non_linear_solution_Q.append(
        solve_ivp(fun=FVM_RHS, args=(args,), y0=h_initial, t_span=t_span, method='BDF', rtol=1e-6, atol=1e-8,
                  events=[unstable, steady_state]).y[:, -1])

L_list = [5, 20, 40, 80]

linear_solution = []
non_linear_solution = []

for L in L_list:
    print(f"L = {L}")
    h_initial = np.ones(GV['N']) * GV['h0']
    dx = L / GV['N']

    args = [PL_make_step, dx, None, GV['Q'], None, n, True, GV['N']]
    linear_solution.append(
        solve_ivp(fun=FVM_RHS, args=(args,), y0=h_initial, t_span=t_span, method='BDF', rtol=1e-6, atol=1e-8,
                  events=[unstable, steady_state]).y[:, -1])

    args = [PL_make_step, dx, None, GV['Q'], None, n, False, GV['N']]
    non_linear_solution.append(
        solve_ivp(fun=FVM_RHS, args=(args,), y0=h_initial, t_span=t_span, method='BDF', rtol=1e-6, atol=1e-8,
                  events=[unstable, steady_state]).y[:, -1])

# linear_error_Q = []
# non_linear_error_Q = []
#
# for i in range(len(GV['Q-list'])):
#     linear_error_Q.append(mod(linear_solution_Q_steady[i], linear_solution_Q[i], scalar=False))
#     non_linear_error_Q.append(mod(non_linear_solution_Q_steady[i], non_linear_solution_Q[i], scalar=False))

## PLOT EVERYTHING
print("Plotting...")

fig = plt.figure(figsize=(20, 10))

gs = gridspec.GridSpec(2, 4, figure=fig, width_ratios=[1.3, 1, 0.85, 0.85])
large_left = fig.add_subplot(gs[:, 0])
error = fig.add_subplot(gs[:, 1])

top_left = fig.add_subplot(gs[0, 2])
top_right = fig.add_subplot(gs[0, 3])
bottom_left = fig.add_subplot(gs[1, 2])
bottom_right = fig.add_subplot(gs[1, 3])

[large_left.plot(GV['x'], linear_solution_Q[i], color=GV['colors'][i], linestyle='--') for i in range(len(GV['Q-list']))]
[large_left.plot(GV['x'], non_linear_solution_Q[i], color=GV['colors'][i], label=f"$Q={GV['Q-list'][i]}$", linestyle='-') for i in range(len(GV['Q-list']))]
large_left.legend(loc='lower right')
large_left.grid(True)
large_left.set_title(f"Complete BVP varying flux Q at\nsurface length $L={GV['L']}$", fontsize=16)
large_left.set_xlabel("Surface Length $(x)$", fontsize=14)
large_left.set_ylabel("Film Height $(y)$", fontsize=14)
large_left.set_ylim(0, 1)

top_left.set_title(f"$L={L_list[0]}$", fontsize=16)
top_left.plot(np.linspace(0, L_list[0], GV['N']), linear_solution[0], linestyle='--', color='k')
top_left.plot(np.linspace(0, L_list[0], GV['N']), non_linear_solution[0], linestyle='-', color='k')
top_left.grid(True)
top_left.set_ylim(0, 1)

top_right.set_title(f"$L={L_list[1]}$", fontsize=16)
top_right.plot(np.linspace(0, L_list[1], GV['N']), linear_solution[1], linestyle='--', color='k')
top_right.plot(np.linspace(0, L_list[1], GV['N']), non_linear_solution[1], linestyle='-', color='k')
top_right.grid(True)
top_right.set_ylim(0, 1)

bottom_left.set_title(f"$L={L_list[2]}$", fontsize=16)
bottom_left.plot(np.linspace(0, L_list[2], GV['N']), linear_solution[2], linestyle='--', color='k')
bottom_left.plot(np.linspace(0, L_list[2], GV['N']), non_linear_solution[2], linestyle='-', color='k')
bottom_left.grid(True)
bottom_left.set_ylim(0, 1)

bottom_right.set_title(f"$L={L_list[3]}$", fontsize=16)
bottom_right.plot(np.linspace(0, L_list[3], GV['N']), linear_solution[3], linestyle='--', color='k')
bottom_right.plot(np.linspace(0, L_list[3], GV['N']), non_linear_solution[3], linestyle='-', color='k')
bottom_right.grid(True)
bottom_right.set_ylim(0, 1)

# [error.semilogy(np.linspace(0, 16, 10_000), linear_error_Q[i], linestyle='--', linewidth=2,  color=GV['colors'][i]) for i in range(len(linear_error_Q))]
# [error.semilogy(np.linspace(0, 16, 10_000), non_linear_error_Q[i], linestyle='-', label=f"$Q={GV['Q-list'][i]}$", linewidth=2, color=GV['colors'][i]) for i in range(len(non_linear_error_Q))]
# error.grid(True)
# error.legend()
# error.set_ylabel("Absolute error", fontsize=14)
# error.set_xlabel("Surface Length $(x)$", fontsize=14)
# error.set_title("Absolute error against\nsteady state with varying Q", fontsize=16)

fig.suptitle("Validation graph varying flux Q with fixed L (left) the error against the steady state model (middle)\nand length scale L with fixed Q (right)", fontsize=20, y=1.055)
plt.tight_layout()
fig.savefig("graphs/thinning_startup_no_DP.png")