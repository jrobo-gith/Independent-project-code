from glob_var.animation import Animation
from glob_var.FVM.FVM_RHS import FVM_RHS
from newtonian_thin_film_solve.individual_files.newtonian_DP import make_step as newt_make_step

from scipy.integrate import solve_ivp
import matplotlib.pyplot as plt
import numpy as np
import json

try:
    with open('../glob_var/global_variables.json') as f:
        GV = json.load(f)
except FileNotFoundError:
    print("Global variables json not found!")

As = [0, 0.1, 0.01, 0.001, 0.0001]

solutions = []

for A in As:
    args = [newt_make_step, GV['dx'], 3, GV['Q'], A, None]
    h_initial = np.ones(GV['N']) * GV['h0']
    min_t = GV['t-span'][f"{GV['L']}"]
    t_span = (0, min_t)

    sol = solve_ivp(fun=FVM_RHS, args=(args,), y0=h_initial, t_span=t_span, method='BDF', rtol=1e-6, atol=1e-8).y
    sol = sol[None, :, :]
    solutions.append(sol)

anim = Animation(
    num_rows=1, num_cols=5,
    fig_size=(15, 12), x=GV['x'],
    num_frames=166, data=solutions,
    fig_details={'x-lim': (0, GV['L']),
                 'y-lim': (0, GV['h0']),
                 'legend': [False, False, False, False, False],}
)
anim.instantiate_animation()
anim.save_animation("test_anim")
