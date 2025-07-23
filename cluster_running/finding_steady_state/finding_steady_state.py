import numpy as np
import os
import json

from scipy.integrate import solve_ivp
from scipy.interpolate import interp1d
import matplotlib.pyplot as plt

from newtonian_thin_film_solve.individual_files.steady_state_central_differences import solver
from glob_var.FVM.FVM_RHS import FVM_RHS
from non_newtonian_thin_film_solve.individual_files.power_law_dp import PL_DP_make_step

current_dir = os.path.dirname(__file__)
project_root = os.path.abspath(os.path.join(current_dir, '..', '..'))

try:
    with open(project_root + '/glob_var/global_variables.json') as f:
        GV = json.load(f)
except FileNotFoundError:
    print("Global variables json not found!")



if __name__ == '__main__':
    tolerance = 1e-4
    placeholder_t_min = 100

    # Get steady state solution for newtonian flow
    SS_sol = solver()
    f = interp1d(SS_sol.x, SS_sol.y[0, :])

    steady_state_newtonian_solution = f(GV['x'])

    saved_startup_sols = []
    saved_error_arrays = []
    saved_ts = []

    t = -10
    av_error = 10000
    while av_error > tolerance:
        t += 10
        print(f"Error is larger than tolerance: {np.round(av_error, 5)}, t is now {t}")
        saved_ts.append(t)
        n = 1.0
        t_span = (0, t)
        h_initial = np.ones(GV['N']) * GV['h0']
        args = [PL_DP_make_step, GV['dx'], None, GV['Q'], 0.0, n]

        sol = solve_ivp(fun=FVM_RHS, y0=h_initial, t_span=t_span, args=(args,), method='BDF', rtol=1e-6, atol=1e-8)
        saved_startup_sols.append(sol.y[:, -1])

        error = np.abs(sol.y[:, -1] - steady_state_newtonian_solution)
        av_error = np.mean(error)
        saved_error_arrays.append(error)

    print(f"Min t found! {t}")
    np.save('../data/min_t/min_t.npy', t)

    fig, ax = plt.subplots(nrows=1, ncols=2, figsize=(15, 10))

    ax[0].plot(GV['x'], steady_state_newtonian_solution, label="Steady State", linestyle='--', color='black')
    [ax[0].plot(GV['x'], saved_startup_sols[i], label=f"$t={saved_ts[i]}$") for i in range(len(saved_ts))]
    [ax[1].semilogy(GV['x'], saved_error_arrays[i], label=f"$t={saved_ts[i]}$") for i in range(len(saved_ts))]

    ax[0].legend()
    ax[1].legend()

    ax[0].grid(True)
    ax[1].grid(True)

    ax[0].set_title("Plot showing the films at varying time-steps", fontsize=14)
    ax[1].set_title("Plot showing the error between startup-flow and steady state\n at varying time-steps", fontsize=14)
    ax[0].set_xlabel("Length $(x)$")
    ax[1].set_xlabel("Length $(x)$")
    ax[0].set_ylabel("Film Height $(h)$")
    ax[1].set_ylabel("Abs Error")

    fig.suptitle(f"Figure showing a thin film evolving at different time scales,\ncompared with the steady state solution. Error < {tolerance} at $t={t}$", fontsize=14)

    fig.savefig('film_error_plot.png')


