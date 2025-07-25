## This file will generate data for three difference power law fluids, where n < 1.0, n = 1.0 and n > 1.0. This will
# give us a gradient of stable, unstable and deformed scenarios at a high resolution. The variables looked at are the
# strength of the disjoint pressure (A) and the volume flux (Q).

## Import function from individual files
from non_newtonian_thin_film_solve.individual_files.power_law_dp import make_step as PL_DP_make_step
from glob_var.FVM.FVM_RHS import FVM_RHS
from glob_var.FVM.check_for_success import check_for_success

import numpy as np
import os
import json
from scipy.integrate import solve_ivp
import multiprocessing as mp

current_dir = os.path.dirname(__file__)
project_root = os.path.abspath(os.path.join(current_dir, '..', '..'))

try:
    with open(project_root + '/glob_var/global_variables.json') as f:
        GV = json.load(f)
except FileNotFoundError:
    print("Global variables json not found!")

resolution = 1
Qs = np.linspace(0.9, 0.95, resolution)
As = np.logspace(-9, -8, resolution)

def heatmap(n):

    max_t = GV['t-span'][f"{GV['L']}"]
    t_span = (0, max_t)
    h_initial = np.ones(GV['N']) * GV['h0']

    data = np.zeros((len(As), len(Qs), GV['N']))
    table = np.zeros((len(As), len(Qs)))

    for j, Q in enumerate(Qs):
        for i, A in enumerate(As):
            print(f"LOADING | n: {n} | Q: {Q} | A: {A} |")
            args = [PL_DP_make_step, GV['dx'], None, Q, A, n]
            sol = solve_ivp(fun=FVM_RHS, y0=h_initial, t_span=t_span, args=(args,), method='BDF', rtol=1e-6, atol=1e-8)
            data[i, j, :] = sol.y[:, -1]
            table[i, j] = check_for_success(sol, Q)

    return data, table

if __name__ == "__main__":
    num_threads = 3         # Number of threads needed
    ns = [1.0, 1.0, 1.0]  # Rheologies to be tested
    data = np.zeros((len(ns), len(As), len(Qs), GV['N']))
    table = np.zeros((len(ns), len(As), len(Qs)))
    with mp.Pool(processes=num_threads) as pool:
         stuff1, stuff2, stuff3 = pool.map(heatmap, ns)

    data1, table1 = stuff1
    data2, table2 = stuff2
    data3, table3 = stuff3
    print(table1, table2, table3)