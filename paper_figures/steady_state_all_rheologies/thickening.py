import numpy as np
from scipy.integrate import solve_ivp
import json
import multiprocessing as mp
import time

try:
    with open('../../glob_var/global_variables.json') as f:
        GV = json.load(f)
except FileNotFoundError:
    print("Global variables json not found!")

from glob_var.FVM.stop_events import unstable, steady_state
from glob_var.FVM.FVM_RHS import FVM_RHS
from non_newtonian_thin_film_solve.individual_files.power_law_dp import make_step as PL_DP_make_step

resolution = 12
n = 1.2

all_Qs = np.linspace(0.1, 0.95, resolution)

def f(Qs):
    data = np.zeros((len(Qs), GV['N']))
    for i, Q in enumerate(Qs):
        print(f"THICKENING | STEADY STATE | Q = {Q}")
        h_initial = np.ones(GV['N']) * GV['h0']
        t_span = GV['t-span']
        args = [PL_DP_make_step, GV['dx'], None, Q, 0.0, n, True, GV['N']]
        unstable.terminal = True
        steady_state.terminal = True
        data[i, :] = solve_ivp(fun=FVM_RHS, y0=h_initial, t_span=t_span, args=(args,), method='BDF', rtol=1e-6, atol=1e-8, events=[unstable, steady_state]).y[:, -1]
    return data


if __name__ == "__main__":
    start = time.time()
    with mp.Pool(processes=6) as pool:
        dat = [[all_Qs[0:2]], [all_Qs[2:4]], [all_Qs[4:6]], [all_Qs[6:8]], [all_Qs[8:10]], [all_Qs[10:12]]]
        results = pool.starmap(f, dat)

    results = np.array(results)
    np.save("data/thickening_SS.npy", results.reshape(resolution, GV['N']))
    end = time.time()

    print(f"Time taken: {np.round(end - start, 2)}.")