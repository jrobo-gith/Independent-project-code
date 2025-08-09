import numpy as np
from scipy.integrate import solve_ivp
import matplotlib.pyplot as plt
import os
import json

from glob_var.FVM.FVM_RHS import FVM_RHS
from glob_var.FVM.stop_events import unstable, steady_state

current_dir = os.path.dirname(__file__)
project_root = os.path.abspath(os.path.join(current_dir, '..', '..'))

try:
    with open(project_root + '/glob_var/global_variables.json') as f:
        GV = json.load(f)
except FileNotFoundError:
    print("Global variables json not found!")


def make_step(h, i, args):
    """
    (Try, except)'s are for the BCs, as q_minus won't be computable for BCs at the start and q_plus won't be computable
    for BCs at the end.
    The bool 'print_except' is for making sure anything inside the for loop is not displaying a 0.
    """

    _, dx, pwr, Q, A, n, linear, _ = args

    DX = 1/dx**3

    try:
        disjoining_pressure_term = 3 * A * (((h[i+1]+h[i])/2)**-4) * ((h[i+1]-h[i])/dx)
        if linear:
            non_linear_h = 1
        else:
            non_linear_h = (0.5 * (h[i] + h[i+1]))**((2*n + 1)/n)
        third_order = abs(DX * (-h[i-1] + 3*h[i] - 3*h[i+1] + h[i+2]) + disjoining_pressure_term) ** (1 / n)
        third_order_sign = np.sign(DX * (-h[i-1] + 3*h[i] - 3*h[i+1] + h[i+2]) + disjoining_pressure_term)
        advection_term = h[i]

        q_plus = non_linear_h * third_order_sign * third_order + advection_term

    except IndexError:
        q_plus = 0

    try:
        disjoining_pressure_term = 3 * A * (((h[i] + h[i-1])/2)**-4) * ((h[i] - h[i-1]) / dx)
        if linear:
            non_linear_h = 1
        else:
            non_linear_h = (0.5 * (h[i] + h[i-1])) ** ((2*n+1)/n)
        third_order = abs(DX * (-h[i-2] + 3*h[i-1] - 3*h[i] + h[i+1]) + disjoining_pressure_term) ** (1 / n)
        third_order_sign = np.sign(DX * (-h[i-2] + 3*h[i-1] - 3*h[i] + h[i+1]) + disjoining_pressure_term)
        advection_term = h[i-1]

        q_minus = non_linear_h * third_order_sign * third_order + advection_term

    except IndexError:
        q_minus = 0

    return q_plus, q_minus

if __name__ == '__main__':
    n = 1.0
    A = 0.0
    Q = 0.95
    h_initial = np.ones(GV['N']) * GV['h0']
    h_initial[0] = GV['h0']
    t_span = GV['t-span']
    args = [make_step, GV['dx'], None, Q, A, n, False, GV['N']]

    unstable.terminal = True
    steady_state.terminal = True
    sol = solve_ivp(fun=FVM_RHS, y0=h_initial, t_span=t_span, args=(args,), method='BDF', rtol=1e-6, atol=1e-8,
                    events=[unstable, steady_state])

    print(sol.status)
    print(sol.success)
    print(sol.message)
    print(sol.t[-1])

    plt.plot(GV['x'], sol.y[:, -1])
    plt.title(f"Graph showing startup-flow solve of Newtonian fluid\nwith $Q={GV['Q']}$, $t={t_span[1]}$, $n={n}$")
    plt.grid(True)
    plt.xlabel('Surface Length $(x)$')
    plt.ylabel('Film Height $(y)$')
    plt.show()