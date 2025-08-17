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
    with open('glob_var/global_variables.json') as f:
        GV = json.load(f)
except FileNotFoundError:
    print("Global variables json not found!")

unstable.terminal = True
steady_state_tracked.terminal = True

h_initial = np.ones(GV['N']) * GV['h0']
t_span = GV['t-span']

A = 0.0

print(f"Newtonian | A = {A} ")
newt_collector = [[], [], [], [], []]
FVM_RHS = make_ode(newt_collector)
n = 1.0
args = [PL_DP_make_step, GV['dx'], None, GV['Q'], A, n, False, GV['N']]
newt_sol = solve_ivp(fun=FVM_RHS, y0=h_initial, t_span=t_span, args=(args,), method='BDF', rtol=1e-6, atol=1e-8, events=[unstable, steady_state_tracked])

ADV, NLT, TOT, DPT, T = newt_collector
ADV = np.array(ADV)
NLT = np.array(NLT)
TOT = np.array(TOT)
DPT = np.array(DPT)
T = np.array(T)

np.save("paper_figures/term_tracker/newtonian_collector/ADV.npy", ADV)
np.save("paper_figures/term_tracker/newtonian_collector/NLT.npy", NLT)
np.save("paper_figures/term_tracker/newtonian_collector/TOT.npy", TOT)
np.save("paper_figures/term_tracker/newtonian_collector/DPT.npy", DPT)
np.save("paper_figures/term_tracker/newtonian_collector/T.npy", T)

del ADV, NLT, TOT, DPT, T
gc.collect()

print(f"Thinning | A = {A} ")
thin_collector = [[], [], [], [], []]
FVM_RHS = make_ode(thin_collector)
n = 0.8
args = [PL_DP_make_step, GV['dx'], None, GV['Q'], A, n, False, GV['N']]
thin_sol = solve_ivp(fun=FVM_RHS, y0=h_initial, t_span=t_span, args=(args,), method='BDF', rtol=1e-6, atol=1e-8, events=[unstable, steady_state_tracked])

ADV, NLT, TOT, DPT, T = thin_collector
ADV = np.array(ADV)
NLT = np.array(NLT)
TOT = np.array(TOT)
DPT = np.array(DPT)
T = np.array(T)

np.save("paper_figures/term_tracker/thinning_collector/ADV.npy", ADV)
np.save("paper_figures/term_tracker/thinning_collector/NLT.npy", NLT)
np.save("paper_figures/term_tracker/thinning_collector/TOT.npy", TOT)
np.save("paper_figures/term_tracker/thinning_collector/DPT.npy", DPT)
np.save("paper_figures/term_tracker/thinning_collector/T.npy", T)

del ADV, NLT, TOT, DPT, T
gc.collect()

print(f"Thickening | A = {A} ")
thic_collector = [[], [], [], [], []]
FVM_RHS = make_ode(thic_collector)
n = 1.2
args = [PL_DP_make_step, GV['dx'], None, GV['Q'], A, n, True, GV['N']]
thic_sol = solve_ivp(fun=FVM_RHS, y0=h_initial, t_span=t_span, args=(args,), method='BDF', rtol=1e-6, atol=1e-8, events=[unstable, steady_state_tracked])

ADV, NLT, TOT, DPT, T = thic_collector
ADV = np.array(ADV)
NLT = np.array(NLT)
TOT = np.array(TOT)
DPT = np.array(DPT)
T = np.array(T)

np.save("paper_figures/term_tracker/thickening_collector/ADV.npy", ADV)
np.save("paper_figures/term_tracker/thickening_collector/NLT.npy", NLT)
np.save("paper_figures/term_tracker/thickening_collector/TOT.npy", TOT)
np.save("paper_figures/term_tracker/thickening_collector/DPT.npy", DPT)
np.save("paper_figures/term_tracker/thickening_collector/T.npy", T)

del ADV, NLT, TOT, DPT, T
gc.collect()

A = 0.15

print(f"Newtonian | A = {A} ")
newt_collector_dp = [[], [], [], [], []]
FVM_RHS = make_ode(newt_collector_dp)
n = 1.0
args = [PL_DP_make_step, GV['dx'], None, GV['Q'], A, n, False, GV['N']]
newt_sol = solve_ivp(fun=FVM_RHS, y0=h_initial, t_span=t_span, args=(args,), method='BDF', rtol=1e-6, atol=1e-8, events=[unstable, steady_state_tracked])

ADV, NLT, TOT, DPT, T = newt_collector_dp
ADV = np.array(ADV)
NLT = np.array(NLT)
TOT = np.array(TOT)
DPT = np.array(DPT)
T = np.array(T)

np.save("paper_figures/term_tracker/newtonian_collector/ADV_dp.npy", ADV)
np.save("paper_figures/term_tracker/newtonian_collector/NLT_dp.npy", NLT)
np.save("paper_figures/term_tracker/newtonian_collector/TOT_dp.npy", TOT)
np.save("paper_figures/term_tracker/newtonian_collector/DPT_dp.npy", DPT)
np.save("paper_figures/term_tracker/newtonian_collector/T_dp.npy", T)

del ADV, NLT, TOT, DPT, T
gc.collect()

print(f"Thinning | A = {A} ")
thin_collector_dp = [[], [], [], [], []]
FVM_RHS = make_ode(thin_collector_dp)
n = 0.8
args = [PL_DP_make_step, GV['dx'], None, GV['Q'], A, n, False, GV['N']]
thin_sol = solve_ivp(fun=FVM_RHS, y0=h_initial, t_span=t_span, args=(args,), method='BDF', rtol=1e-6, atol=1e-8, events=[unstable, steady_state_tracked])

ADV, NLT, TOT, DPT, T = thin_collector_dp
ADV = np.array(ADV)
NLT = np.array(NLT)
TOT = np.array(TOT)
DPT = np.array(DPT)
T = np.array(T)

np.save("paper_figures/term_tracker/thinning_collector/ADV_dp.npy", ADV)
np.save("paper_figures/term_tracker/thinning_collector/NLT_dp.npy", NLT)
np.save("paper_figures/term_tracker/thinning_collector/TOT_dp.npy", TOT)
np.save("paper_figures/term_tracker/thinning_collector/DPT_dp.npy", DPT)
np.save("paper_figures/term_tracker/thinning_collector/T_dp.npy", T)

del ADV, NLT, TOT, DPT, T
gc.collect()

print(f"Thickening | A = {A} ")
thic_collector_dp = [[], [], [], [], []]
FVM_RHS = make_ode(thic_collector_dp)
n = 1.2
args = [PL_DP_make_step, GV['dx'], None, GV['Q'], A, n, False, GV['N']]
thic_sol = solve_ivp(fun=FVM_RHS, y0=h_initial, t_span=t_span, args=(args,), method='BDF', rtol=1e-6, atol=1e-8, events=[unstable, steady_state_tracked])

ADV, NLT, TOT, DPT, T = thic_collector_dp
ADV = np.array(ADV)
NLT = np.array(NLT)
TOT = np.array(TOT)
DPT = np.array(DPT)
T = np.array(T)

np.save("paper_figures/term_tracker/thickening_collector/ADV_dp.npy", ADV)
np.save("paper_figures/term_tracker/thickening_collector/NLT_dp.npy", NLT)
np.save("paper_figures/term_tracker/thickening_collector/TOT_dp.npy", TOT)
np.save("paper_figures/term_tracker/thickening_collector/DPT_dp.npy", DPT)
np.save("paper_figures/term_tracker/thickening_collector/T_dp.npy", T)

del ADV, NLT, TOT, DPT, T
gc.collect()