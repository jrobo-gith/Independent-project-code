# Import packages
import matplotlib.pyplot as plt
import numpy as np
from scipy.optimize import root

from non_newtonian_thin_film_solve.individual_files.power_law_startup import make_step

import json
import os

current_dir = os.path.dirname(__file__)
project_root = os.path.abspath(os.path.join(current_dir, '..', '..'))

try:
    with open(project_root + '/glob_var/global_variables.json') as f:
        GV = json.load(f)
except FileNotFoundError:
    print("Global variables json not found!")

def FVM_RHS(h: np.ndarray, args: tuple) -> np.ndarray:
    """
    RHS of equation, made for scipy's solve_ivp function that takes care of this stiff fourth order PDE.
    """

    make_step, dx, pwr, Q, _, n, _, N = args

    h = h.copy()
    dhdt = np.zeros_like(h)

    # i = 0
    h[0] = GV['h0']

    # i = 1
    q_plus, q_minus = make_step(h=h, i=1, args=args)
    dhdt[1] = - (q_plus - Q) / dx

    # i = N - 2
    q_plus, q_minus = make_step(h=h, i=N - 2, args=args)
    dhdt[N - 2] = - (h[N - 2] - q_minus) / dx

    # i = N - 1
    h[N - 1] = h[N - 2]
    dhdt[N - 1] = dhdt[N - 2]

    for i in range(2, N - 2):
        q_plus, q_minus = make_step(h=h, i=i, args=args)
        dhdt[i] = -(q_plus - q_minus) / dx

    return dhdt


args = [make_step, GV['dx'], 3, GV['Q'], None, 1.2, False, GV['N']]
h_initial = np.ones(GV['N']) * 0.8
h_initial[0] = GV['h0']

sol = root(fun=FVM_RHS, args=(args,), x0=h_initial)

plt.plot(GV['x'], sol.x)
plt.show()

