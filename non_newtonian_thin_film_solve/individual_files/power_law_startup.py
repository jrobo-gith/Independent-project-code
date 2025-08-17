import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp

import os
import json
import time

current_dir = os.path.dirname(__file__)
project_root = os.path.abspath(os.path.join(current_dir, '..', '..'))

try:
    with open(project_root + '/glob_var/global_variables.json') as f:
        GV = json.load(f)
except FileNotFoundError:
    print("Global variables json not found!")

from glob_var.FVM.FVM_RHS import FVM_RHS
from glob_var.FVM.stop_events import unstable, steady_state

def make_step(h, i, args):
    """
    (Try, except)'s are for the BCs, as q_minus won't be computable for BCs at the start and q_plus won't be computable
    for BCs at the end.
    The bool 'print_except' is for making sure anything inside the for loop is not displaying a 0.
    """

    _, dx, pwr, Q, _, n, linear, _ = args

    DX = 1 / dx ** 3
    epsilon = 0

    try:
        if linear:
            non_linear_h = 1.0
        else:
            non_linear_h = (0.5 * (h[i] + h[i + 1])) ** ((2 * n + 1) / n) + epsilon
        third_order = abs(-h[i - 1] + 3 * h[i] - 3 * h[i + 1] + h[i + 2] + epsilon) ** (1 / n)
        third_order_sign = np.sign(-h[i - 1] + 3 * h[i] - 3 * h[i + 1] + h[i + 2])
        advection_term = h[i]

        q_plus = DX * non_linear_h * third_order_sign * third_order + advection_term

    except IndexError:
        q_plus = 0

    try:
        if linear:
            non_linear_h = 1.0
        else:
            non_linear_h = (0.5 * (h[i] + h[i - 1])) ** ((2 * n + 1) / n) + epsilon
        third_order = abs(-h[i - 2] + 3 * h[i - 1] - 3 * h[i] + h[i + 1] + epsilon) ** (1 / n)
        third_order_sign = np.sign(-h[i - 2] + 3 * h[i - 1] - 3 * h[i] + h[i + 1])
        advection_term = h[i - 1]

        q_minus = DX * non_linear_h * third_order_sign * third_order + advection_term

    except IndexError:
        q_minus = 0

    return q_plus, q_minus

if __name__ == '__main__':
    n = 1.2
    h_initial = np.ones(GV['N']) * GV['h0']
    t_span = GV['t-span']
    L = GV['L']
    dx = L/GV['N']
    args = [make_step, GV['dx'], 3, GV['Q'], None, n, False, GV['N']]

    steady_state.terminal = True
    unstable.terminal = True

    start = time.time()
    sol = solve_ivp(fun=FVM_RHS, y0=h_initial, t_span=t_span, args=(args,), method='BDF', rtol=1e-6, atol=1e-8, events=[steady_state, unstable])
    end = time.time()

    print(f"Time taken = {end - start:.2f} seconds")

    print(sol.status)
    print(sol.success)
    print(sol.message)

    fig, ax = plt.subplots(nrows=1, ncols=1, figsize=(8, 10))
    ax.plot(GV['x'], sol.y[:, -1], label=f'End, t={sol.t[-1]}')
    ax.set_ylim(0, 1)
    ax.grid(True)
    ax.legend()
    ax.set_title(f"Startup flow with $h_0$={GV['h0']}, Q={GV['Q']} and n={n}")
    ax.set_ylabel('height$(h)$')
    ax.set_xlabel('Length $(x)$')
    fig.show()
