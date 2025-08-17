import json

from diffrax import steady_state_event

from glob_var.FVM.FVM_RHS import FVM_RHS
from glob_var.FVM.stop_events import unstable, steady_state
from scipy.integrate import solve_ivp
import numpy as np

try:
    with open('../../glob_var/global_variables.json') as f:
        GV = json.load(f)
except FileNotFoundError:
    print("Global variables json not found!")

t_span = GV['t-span']
h_initial = np.ones(GV['N']) * GV['h0']

def make_step(h, i, args):
    """
    Try, excepts are for the BCs, as q_minus won't be computable for BCs at the start and q_plus won't be computable
    for BCs at the end.
    """

    _, dx, pwr, Q, _, n, _, _ = args

    Dx = 1 / dx ** 3
    try:
        non_linear_term = ((h[i] + h[i + 1]) / 2) ** pwr
        q_plus = Dx * non_linear_term * (-h[i - 1] + 3 * h[i] - 3 * h[i + 1] + h[i + 2]) + h[i]
    except IndexError:
        q_plus = 0

    try:
        non_linear_term = ((h[i] + h[i - 1]) / 2) ** pwr
        q_minus = Dx * non_linear_term * (-h[i - 2] + 3 * h[i - 1] - 3 * h[i] + h[i + 1]) + h[i - 1]
    except IndexError:
        q_minus = 0

    return q_plus, q_minus

if __name__ == '__main__':
    unstable.terminal = True
    steady_state.terminal = True
    args = [make_step, GV['dx'], 3, GV['Q'], None, 1.0, None, GV['N']]
    nl_sol = solve_ivp(fun=FVM_RHS, args=(args,), y0=h_initial, t_span=t_span, method='BDF', rtol=1e-6, atol=1e-8,
                  events=[unstable, steady_state])

    print(nl_sol.t)
    print(len(nl_sol.t))