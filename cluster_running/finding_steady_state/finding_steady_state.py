import numpy as np
import os
import json

from scipy.integrate import solve_ivp
from scipy.interpolate import interp1d

from newtonian_thin_film_solve.individual_files.steady_state_central_differences import solver
from glob_var.FVM.FVM_RHS import FVM_RHS
from non_newtonian_thin_film_solve.individual_files.power_law_dp import make_step as PL_DP_make_step

current_dir = os.path.dirname(__file__)
project_root = os.path.abspath(os.path.join(current_dir, '..', '..'))

try:
    with open(project_root + '/glob_var/global_variables.json') as f:
        GV = json.load(f)
except FileNotFoundError:
    print("Global variables json not found!")

if __name__ == '__main__':
    tolerance = 1e-4
    t_list = {} # Dictionary with key: Lengths, value: min_t
    t = -10
    for L in GV['L-list']:
        # Get steady state solution for newtonian flow
        SS_sol = solver(q=GV['Q'], L=L, linear=False)
        f = interp1d(SS_sol.x, SS_sol.y[0, :])

        new_x = np.linspace(0, L, GV['N'])
        steady_state_newtonian_solution = f(new_x)

        saved_startup_sols = []
        saved_error_arrays = []
        saved_ts = []

        av_error = 1
        while av_error > tolerance:
            t += 10
            print(f"Error is larger than tolerance: {np.round(av_error, 5)}, t is now {t}")
            saved_ts.append(t)
            n = 1.0
            t_span = (0, t)
            dx = L/GV['N']
            h_initial = np.ones(GV['N']) * GV['h0']
            args = [PL_DP_make_step, dx, None, GV['Q'], 0.0, n]

            sol = solve_ivp(fun=FVM_RHS, y0=h_initial, t_span=t_span, args=(args,), method='BDF', rtol=1e-6, atol=1e-8)
            saved_startup_sols.append(sol.y[:, -1])

            error = np.abs(sol.y[:, -1] - steady_state_newtonian_solution)
            av_error = np.mean(error)
            saved_error_arrays.append(error)

        print(f"Min t found! | L = {L} | t = {t} |\n")
        t_list[f'{L}'] = t

    file = '../data/min_t/min_ts.json'
    with open(file, 'w') as f:
        json.dump(t_list, f)

