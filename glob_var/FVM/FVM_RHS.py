import numpy as np
import os
import json

current_dir = os.path.dirname(__file__)
project_root = os.path.abspath(os.path.join(current_dir, '..', '..'))

try:
    with open(project_root + '/glob_var/global_variables.json') as f:
        GV = json.load(f)
except FileNotFoundError:
    print("Global variables json not found!")

def FVM_RHS(t: float, h: np.ndarray, args: tuple) -> np.ndarray:
    """
    RHS of equation, made for scipy's solve_ivp function that takes care of this stiff fourth order PDE.
    """

    N = GV['N']

    make_step, dx, pwr, Q, _, n = args

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