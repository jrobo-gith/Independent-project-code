import os
import sys
import gc

root_dir = os.getcwd()
sys.path.insert(0, root_dir)

import numpy as np
import json
from scipy.integrate import solve_ivp
from glob_var.FVM.FVM_RHS_tracked import make_ode
from glob_var.FVM.stop_events import unstable, steady_state_tracked
from non_newtonian_thin_film_solve.individual_files.power_law_dp_tracked import make_step as PL_DP_make_step

try:
    with open('../../glob_var/global_variables.json') as f:
        GV = json.load(f)
except FileNotFoundError:
    print("Global variables json not found!")

unstable.terminal = True
steady_state_tracked.terminal = True

max_t = 0

h_initial = np.ones(GV['N']) * GV['h0']
t_span = GV['t-span']

A = 0.0

print(f"Newtonian | A = {A} ")
newt_collector = [[], [], []]
FVM_RHS = make_ode(newt_collector)
n = 1.0
args = [PL_DP_make_step, GV['dx'], None, GV['Q'], A, n, False, GV['N']]
newt_sol = solve_ivp(fun=FVM_RHS, y0=h_initial, t_span=t_span, args=(args,), method='BDF', rtol=1e-6, atol=1e-8, events=[unstable, steady_state_tracked])

SGN, MAG, RAT = newt_collector
SGN = np.array(SGN)
MAG = np.array(MAG)

np.save("newtonian_collector/SGN.npy", SGN)
np.save("newtonian_collector/MAG.npy", MAG)
np.save("newtonian_collector/sol_t.npy", newt_sol.t)

del SGN, MAG, RAT, newt_sol
gc.collect()

print(f"Thinning | A = {A} ")
thin_collector = [[], [], []]
FVM_RHS = make_ode(thin_collector)
n = 0.8
args = [PL_DP_make_step, GV['dx'], None, GV['Q'], A, n, False, GV['N']]
thin_sol = solve_ivp(fun=FVM_RHS, y0=h_initial, t_span=t_span, args=(args,), method='BDF', rtol=1e-6, atol=1e-8, events=[unstable, steady_state_tracked])

SGN, MAG, RAT = thin_collector
SGN = np.array(SGN)
MAG = np.array(MAG)

np.save("thinning_collector/SGN.npy", SGN)
np.save("thinning_collector/MAG.npy", MAG)
np.save("thinning_collector/sol_t.npy", thin_sol.t)

del SGN, MAG, RAT, thin_sol
gc.collect()

print(f"Thickening | A = {A} ")
thic_collector = [[], [], []]
FVM_RHS = make_ode(thic_collector)
n = 1.2
args = [PL_DP_make_step, GV['dx'], None, GV['Q'], A, n, True, GV['N']]
thic_sol = solve_ivp(fun=FVM_RHS, y0=h_initial, t_span=t_span, args=(args,), method='BDF', rtol=1e-6, atol=1e-8, events=[unstable, steady_state_tracked])

SGN, MAG, RAT = thic_collector
SGN = np.array(SGN)
MAG = np.array(MAG)

np.save("thickening_collector/SGN.npy", SGN)
np.save("thickening_collector/MAG.npy", MAG)
np.save("thickening_collector/sol_t.npy", thic_sol.t)

del SGN, MAG, RAT, thic_sol
gc.collect()

A = 0.15

print(f"Newtonian | A = {A} ")
newt_collector_dp = [[], [], []]
FVM_RHS = make_ode(newt_collector_dp)
n = 1.0
args = [PL_DP_make_step, GV['dx'], None, GV['Q'], A, n, False, GV['N']]
newt_sol = solve_ivp(fun=FVM_RHS, y0=h_initial, t_span=t_span, args=(args,), method='BDF', rtol=1e-6, atol=1e-8, events=[unstable, steady_state_tracked])

SGN, MAG, RAT = newt_collector_dp
SGN = np.array(SGN)
MAG = np.array(MAG)
RAT = np.array(RAT)

np.save("newtonian_collector/SGN_DP.npy", SGN)
np.save("newtonian_collector/MAG_DP.npy", MAG)
np.save("newtonian_collector/RAT_DP.npy", RAT)
np.save("newtonian_collector/sol_t_DP.npy", newt_sol.t)

del SGN, MAG, RAT, newt_sol
gc.collect()

print(f"Thinning | A = {A} ")
thin_collector_dp = [[], [], []]
FVM_RHS = make_ode(thin_collector_dp)
n = 0.8
args = [PL_DP_make_step, GV['dx'], None, GV['Q'], A, n, False, GV['N']]
thin_sol = solve_ivp(fun=FVM_RHS, y0=h_initial, t_span=t_span, args=(args,), method='BDF', rtol=1e-6, atol=1e-8, events=[unstable, steady_state_tracked])

SGN, MAG, RAT = thin_collector_dp
SGN = np.array(SGN)
MAG = np.array(MAG)
RAT = np.array(RAT)

np.save("thinning_collector/SGN_DP.npy", SGN)
np.save("thinning_collector/MAG_DP.npy", MAG)
np.save("thinning_collector/RAT_DP.npy", RAT)
np.save("thinning_collector/sol_t_DP.npy", thin_sol.t)

del SGN, MAG, RAT, thin_sol
gc.collect()

print(f"Thickening | A = {A} ")
thic_collector_dp = [[], [], []]
FVM_RHS = make_ode(thic_collector_dp)
n = 1.2
args = [PL_DP_make_step, GV['dx'], None, GV['Q'], A, n, False, GV['N']]
thic_sol = solve_ivp(fun=FVM_RHS, y0=h_initial, t_span=t_span, args=(args,), method='BDF', rtol=1e-6, atol=1e-8, events=[unstable, steady_state_tracked])

SGN, MAG, RAT = thic_collector_dp
SGN = np.array(SGN)
MAG = np.array(MAG)
RAT = np.array(RAT)

np.save("thickening_collector/SGN_DP.npy", SGN)
np.save("thickening_collector/MAG_DP.npy", MAG)
np.save("thickening_collector/RAT_DP.npy", RAT)
np.save("thickening_collector/sol_t_DP.npy", thic_sol.t)

del SGN, MAG, RAT, thic_sol
gc.collect()