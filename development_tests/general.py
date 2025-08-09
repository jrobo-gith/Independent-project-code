import numpy as np
import json
from scipy.integrate import solve_ivp
from glob_var.FVM.FVM_RHS import FVM_RHS
from non_newtonian_thin_film_solve.individual_files.power_law_dp import make_step
from glob_var.FVM.stop_events import  steady_state, unstable
import multiprocessing as mp
import time

with open("../glob_var/global_variables.json") as f:
    GV = json.load(f)

resolution = 4
all_Qs = np.linspace(0.1, 0.95, resolution)
all_As = np.linspace(0, 0.5, resolution)
overall_data = np.zeros((len(all_Qs), len(all_As), GV['N']))

n = 1.0
linear = False
h_initial = np.ones(GV['N'])
t_span = GV['t-span']
unstable.terminal = True
steady_state.terminal = True


def f(Qs, As):
    data = np.zeros((len(Qs), len(As), GV['N']))
    for i, Q in enumerate(Qs):
        for j, A in enumerate(As):
            print(f"LOADING ~ NEWTONIAN ~ Q = {np.round(Q, 2)} ~ A = {np.round(A, 2)}.")
            args = [make_step, GV['dx'], None, Q, A, n, linear, GV['N']]
            data[i, j, :] = solve_ivp(fun=FVM_RHS, y0=h_initial, args=(args,), t_span=t_span, method='BDF',
                                      rtol=1e-6, atol=1e-8, events=[unstable, steady_state]).y[:, -1]
    return data


if __name__ == "__main__":
    start = time.time()
    dat = [(all_Qs[0:1], all_As), (all_Qs[1:2], all_As), (all_Qs[2:3], all_As), (all_Qs[3:4], all_As)]
    with mp.Pool(processes=4) as pool:
        results = pool.starmap(f, dat)

    results = np.array(results)

    np.save("newtonian.npy",
            results.reshape(resolution, 1, GV['N']))
    end = time.time()

    print(f"Time taken: {np.round(end - start, 2)}.")