import numpy as np
import json

from scipy.integrate import solve_ivp
from glob_var.FVM.FVM_RHS import FVM_RHS
from glob_var.FVM.stop_events import unstable, steady_state
from non_newtonian_thin_film_solve.individual_files.power_law_dp import make_step as PL_DP_make_step
from non_newtonian_thin_film_solve.individual_files.power_law_startup import make_step as PL_make_step

try:
    with open('../../glob_var/global_variables.json') as f:
        GV = json.load(f)
except FileNotFoundError:
    print("Global variables json not found!")

unstable.terminal = True
steady_state.terminal = True

h_initial = np.ones(GV['N']) * GV['h0']
t_span = GV['t-span']

A = 0.0

print(f"Newtonian | A = {A} ")
n = 1.0
args = [PL_make_step, GV['dx'], None, GV['Q'], A, n, False, GV['N']]
newt_sol = solve_ivp(fun=FVM_RHS, y0=h_initial, t_span=t_span, args=(args,), method='BDF', rtol=1e-6, atol=1e-8, events=[unstable, steady_state])

print(f"Thinning | A = {A} ")
n = 0.8
args = [PL_make_step, GV['dx'], None, GV['Q'], A, n, False, GV['N']]
thin_sol = solve_ivp(fun=FVM_RHS, y0=h_initial, t_span=t_span, args=(args,), method='BDF', rtol=1e-6, atol=1e-8, events=[unstable, steady_state])

print(f"Thickening | A = {A} ")
n = 1.2
args = [PL_make_step, GV['dx'], None, GV['Q'], A, n, False, GV['N']]
thic_sol = solve_ivp(fun=FVM_RHS, y0=h_initial, t_span=t_span, args=(args,), method='BDF', rtol=1e-6, atol=1e-8, events=[unstable, steady_state])

np.save("data/thic_sol_t", thic_sol.t)
np.save("data/thin_sol_t", thin_sol.t)
np.save("data/newt_sol_t", newt_sol.t)

np.save("data/thic_sol", thic_sol.y)
np.save("data/thin_sol", thin_sol.y)
np.save("data/newt_sol", newt_sol.y)

A = 0.15

print(f"Newtonian | A = {A} ")
n = 1.0
args = [PL_DP_make_step, GV['dx'], None, GV['Q'], A, n, False, GV['N']]
newt_sol = solve_ivp(fun=FVM_RHS, y0=h_initial, t_span=t_span, args=(args,), method='BDF', rtol=1e-6, atol=1e-8, events=[unstable, steady_state])

print(f"Thinning | A = {A} ")
n = 0.8
args = [PL_DP_make_step, GV['dx'], None, GV['Q'], A, n, False, GV['N']]
thin_sol = solve_ivp(fun=FVM_RHS, y0=h_initial, t_span=t_span, args=(args,), method='BDF', rtol=1e-6, atol=1e-8, events=[unstable, steady_state])

print(f"Thickening | A = {A} ")
n = 1.2
args = [PL_DP_make_step, GV['dx'], None, GV['Q'], A, n, False, GV['N']]
thic_sol = solve_ivp(fun=FVM_RHS, y0=h_initial, t_span=t_span, args=(args,), method='BDF', rtol=1e-6, atol=1e-8, events=[unstable, steady_state])


np.save("data/thic_sol_t_dp", thic_sol.t)
np.save("data/thin_sol_t_dp", thin_sol.t)
np.save("data/newt_sol_t_dp", newt_sol.t)

np.save("data/thic_sol_dp", thic_sol.y)
np.save("data/thin_sol_dp", thin_sol.y)
np.save("data/newt_sol_dp", newt_sol.y)

