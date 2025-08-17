import numpy as np
import json
from scipy.integrate import solve_ivp
from non_newtonian_thin_film_solve.individual_files.power_law_steady import solver as bdf_solver_NN
from non_newtonian_thin_film_solve.individual_files.power_law_startup import make_step
from glob_var.FVM.stop_events import unstable, steady_state
from glob_var.FVM.FVM_RHS import FVM_RHS
from paper_figures.validation_graphs.graph_generator import generate_validation_graph as gvg
from paper_figures.validation_graphs.error import temporal_error, spatial_error, interpolate

import matplotlib.pyplot as plt

try:
    with open('../../glob_var/global_variables.json') as f:
        GV = json.load(f)
except FileNotFoundError:
    print("Global variables json not found!")

t_span = GV['t-span']
n = 1.2

## CALCULATE THICKENING STEADY STATE
print("Calculating steady state...")

linear_solution_Q_steady = []  # Store linear solutions varying Q
non_linear_solution_Q_steady = []  # Store non-linear solutions varying Q

for i in range(len(GV['Q-list'])):
    print(f"LOADING STEADY STATE | LINEAR | Q = {GV['Q-list'][i]}")
    steady_linear_sol = bdf_solver_NN(q=GV['Q-list'][i], L=GV['L'], n=n, linear=True)
    linear_solution_Q_steady.append(steady_linear_sol.y[0])

    print(f"LOADING STEADY STATE | NON-LINEAR | Q = {GV['Q-list'][i]}")
    steady_nonlinear_sol = bdf_solver_NN(q=GV['Q-list'][i], L=GV['L'], n=n)
    non_linear_solution_Q_steady.append(steady_nonlinear_sol.y[0])

## CALCULATE STARTUP FLOW
print("Calculating startup flow...")

linear_solution_Q = []
non_linear_solution_Q = []
linear_temporal_errors = []
non_linear_temporal_errors = []

linear_temporal_x = []
non_linear_temporal_x = []

for i in range(len(GV['Q-list'])):
    print(f"LOADING STARTUP FLOW | LINEAR | Q = {GV['Q-list'][i]}")

    h_initial = np.ones(GV['N'])

    unstable.terminal = True
    steady_state.terminal = True
    args = [make_step, GV['dx'], 0, GV['Q-list'][i], None, n, True, GV['N']]
    l_sol = solve_ivp(fun=FVM_RHS, args=(args,), y0=h_initial, t_span=t_span, method='BDF', rtol=1e-6, atol=1e-8,
                  events=[unstable, steady_state])
    linear_solution_Q.append(l_sol.y[:, -1])

    print(f"LOADING STARTUP FLOW | NON-LINEAR | Q = {GV['Q-list'][i]}")

    unstable.terminal = True
    steady_state.terminal = True
    args = [make_step, GV['dx'], 3, GV['Q-list'][i], None, n, False, GV['N']]
    nl_sol = solve_ivp(fun=FVM_RHS, args=(args,), y0=h_initial, t_span=t_span, method='BDF', rtol=1e-6, atol=1e-8,
                  events=[unstable, steady_state])
    non_linear_solution_Q.append(nl_sol.y[:, -1])

    ## Compute temporal error
    lte = []
    nlte = []
    for j in range(l_sol.y.shape[1]):
        lte.append(temporal_error(steady=linear_solution_Q_steady[i], startup=l_sol.y[:, j]))
    for j in range(nl_sol.y.shape[1]):
        nlte.append(temporal_error(steady=non_linear_solution_Q_steady[i], startup=nl_sol.y[:, j]))
    linear_temporal_errors.append(lte)
    non_linear_temporal_errors.append(nlte)

    linear_temporal_x.append(l_sol.t)
    non_linear_temporal_x.append(nl_sol.t)

L_list = [5, 20, 40, 80]

linear_solution = []
non_linear_solution = []

for L in L_list:
    h_initial = np.ones(GV['N']) * GV['h0']
    dx = L / GV['N']

    print(f"LOADING LENGTH | LINEAR | L = {L}")

    args = [make_step, dx, 0, GV['Q'], None, n, True, GV['N']]
    linear_solution.append(
        solve_ivp(fun=FVM_RHS, args=(args,), y0=h_initial, t_span=t_span, method='BDF', rtol=1e-6, atol=1e-8,
                  events=[unstable, steady_state]).y[:, -1])

    print(f"LOADING LENGTH | NON-LINEAR | L = {L}")

    args = [make_step, dx, 3, GV['Q'], None, n, False, GV['N']]
    non_linear_solution.append(
        solve_ivp(fun=FVM_RHS, args=(args,), y0=h_initial, t_span=t_span, method='BDF', rtol=1e-6, atol=1e-8,
                  events=[unstable, steady_state]).y[:, -1])

linear_error_Q = []
non_linear_error_Q = []

linear_SS = []
non_linear_SS = []

for i in range(len(GV['Q-list'])):
    linear_error_Q.append(spatial_error(linear_solution_Q_steady[i], linear_solution_Q[i]))
    linear_SS.append(interpolate(linear_solution_Q_steady[i], linear_solution_Q[i]))
    non_linear_error_Q.append(spatial_error(non_linear_solution_Q_steady[i], non_linear_solution_Q[i]))
    non_linear_SS.append(interpolate(non_linear_solution_Q_steady[i], non_linear_solution_Q[i]))

gvg(L_arrays=[linear_solution, non_linear_solution], Q_arrays=[linear_solution_Q, non_linear_solution_Q],
    space_error_y_arrays=[linear_error_Q, non_linear_error_Q],
    time_error_x_arrays=[linear_temporal_x, non_linear_temporal_x],
    time_error_y_arrays=[linear_temporal_errors, non_linear_temporal_errors],
    steady_states_array=[linear_SS, non_linear_SS], directory="graphs/thickening_startup.png", rheology="Shear-Thickening")