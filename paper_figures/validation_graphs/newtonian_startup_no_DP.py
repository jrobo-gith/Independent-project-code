##### Used [gridspec](https://matplotlib.org/stable/api/_as_gen/matplotlib.gridspec.GridSpec.html) when making these types of plots

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import json
from scipy.integrate import solve_ivp
from newtonian_thin_film_solve.individual_files.steady_state_central_differences import solver as bdf_solver_N
from newtonian_thin_film_solve.individual_files.startup_flow_FVM import make_step as newt_make_step
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
print("Calculating steady state...")

linear_solution_Q_steady = []  # Store linear solutions varying Q
non_linear_solution_Q_steady = []  # Store non-linear solutions varying Q

for i in range(len(GV['Q-list'])):
    linear_solution_Q_steady.append(bdf_solver_N(q=GV['Q-list'][i], L=GV['L'], linear=True).y[0])
    non_linear_solution_Q_steady.append(bdf_solver_N(q=GV['Q-list'][i], L=GV['L'], linear=False).y[0])

## CALCULATE STARTUP FLOW
print("Calculating startup flow...")

linear_solution_Q = []  # Store linear solutions varying Q
non_linear_solution_Q = []  # Store non-linear solutions varying Q

linear_times = []
non_linear_times = []

linear_temporal_errors = []
non_linear_temporal_errors = []

for i in range(len(GV['Q-list'])):
    h_initial = np.ones(GV['N']) * GV['h0']

    args = [newt_make_step, GV['dx'], 0, GV['Q-list'][i], None, 1.0, None, GV['N']]
    l_sol = solve_ivp(fun=FVM_RHS, args=(args,), y0=h_initial, t_span=t_span, method='BDF', rtol=1e-6, atol=1e-8,
                  events=[unstable, steady_state])
    linear_solution_Q.append(l_sol.y[:, -1])
    linear_times.append(l_sol.t)

    args = [newt_make_step, GV['dx'], 3, GV['Q-list'][i], None, 1.0, None, GV['N']]
    nl_sol = solve_ivp(fun=FVM_RHS, args=(args,), y0=h_initial, t_span=t_span, method='BDF', rtol=1e-6, atol=1e-8,
                  events=[unstable, steady_state])
    non_linear_solution_Q.append(nl_sol.y[:, -1])
    non_linear_times.append(nl_sol.t)

    ## Compute temporal error
    lte = []
    nlte = []
    for j in range(l_sol.y.shape[1]):
        lte.append(mod(steady_state=linear_solution_Q_steady[i], startup_flow=l_sol.y[:, j], scalar=True))
    for j in range(nl_sol.y.shape[1]):
        nlte.append(mod(steady_state=non_linear_solution_Q_steady[i], startup_flow=nl_sol.y[:, j], scalar=True))
    linear_temporal_errors.append(lte)
    non_linear_temporal_errors.append(nlte)

L_list = [5, 20, 40, 80]

linear_solution = []
non_linear_solution = []

for L in L_list:
    h_initial = np.ones(GV['N']) * GV['h0']
    dx = L / GV['N']

    args = [newt_make_step, dx, 0, GV['Q'], None, 1.0, None, GV['N']]
    linear_solution.append(
        solve_ivp(fun=FVM_RHS, args=(args,), y0=h_initial, t_span=t_span, method='BDF', rtol=1e-6, atol=1e-8,
                  events=[unstable, steady_state]).y[:, -1])

    args = [newt_make_step, dx, 3, GV['Q'], None, 1.0, None, GV['N']]
    non_linear_solution.append(
        solve_ivp(fun=FVM_RHS, args=(args,), y0=h_initial, t_span=t_span, method='BDF', rtol=1e-6, atol=1e-8,
                  events=[unstable, steady_state]).y[:, -1])

linear_error_Q = []
non_linear_error_Q = []

for i in range(len(GV['Q-list'])):
    linear_error_Q.append(mod(linear_solution_Q_steady[i], linear_solution_Q[i], scalar=False))
    non_linear_error_Q.append(mod(non_linear_solution_Q_steady[i], non_linear_solution_Q[i], scalar=False))

## PLOT EVERYTHING
print("Plotting...")

fig = plt.figure(figsize=(10, 15))

gs = gridspec.GridSpec(4, 2, figure=fig, height_ratios=[1.25, 1.25, 0.75, 0.75])
large_left = fig.add_subplot(gs[0, :])
temporal_error = fig.add_subplot(gs[1, 0])
error = fig.add_subplot(gs[1, 1])

top_left = fig.add_subplot(gs[2, 0])
top_right = fig.add_subplot(gs[2, 1])
bottom_left = fig.add_subplot(gs[3, 0])
bottom_right = fig.add_subplot(gs[3, 1])

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
top_left.set_ylabel("Film Height $(y)$", fontsize=14)

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
bottom_left.set_ylabel("Film Height $(y)$", fontsize=14)
bottom_left.set_xlabel("Surface Length $(x)$", fontsize=14)

bottom_right.set_title(f"$L={L_list[3]}$", fontsize=16)
bottom_right.plot(np.linspace(0, L_list[3], GV['N']), linear_solution[3], linestyle='--', color='k')
bottom_right.plot(np.linspace(0, L_list[3], GV['N']), non_linear_solution[3], linestyle='-', color='k')
bottom_right.grid(True)
bottom_right.set_ylim(0, 1)
bottom_right.set_xlabel("Surface Length $(x)$", fontsize=14)

[temporal_error.loglog(linear_times[i], linear_temporal_errors[i], linewidth=1.5, linestyle='--', color=GV['colors'][i]) for i in range(len(GV['Q-list']))]
[temporal_error.loglog(non_linear_times[i], non_linear_temporal_errors[i], linewidth=1.5, color=GV['colors'][i], label=f"$Q={GV['Q-list'][i]}$") for i in range(len(GV['Q-list']))]
temporal_error.set_xlim(1e-0, 1e2)
temporal_error.grid(True)
temporal_error.legend()
temporal_error.set_xlabel("Time steps $(\Delta t)$", fontsize=14)
temporal_error.set_ylabel("Absolute Error", fontsize=14)
temporal_error.set_title("Error of model varying volume flux over time", fontsize=16)

[error.semilogy(np.linspace(0, 16, 10_000), linear_error_Q[i], linestyle='--', linewidth=1.5,  color=GV['colors'][i]) for i in range(len(linear_error_Q))]
[error.semilogy(np.linspace(0, 16, 10_000), non_linear_error_Q[i], linestyle='-', label=f"$Q={GV['Q-list'][i]}$", linewidth=1.5, color=GV['colors'][i]) for i in range(len(non_linear_error_Q))]
error.grid(True)
error.legend()
error.set_ylabel("Absolute error", fontsize=14)
error.set_xlabel("Surface Length $(x)$", fontsize=14)
error.set_title("Absolute error against\nsteady state with varying Q", fontsize=16)

plt.tight_layout()
fig.savefig("graphs/newtonian_startup_no_DP.png")