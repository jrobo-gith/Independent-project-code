import numpy as np
import matplotlib.pyplot as plt
import os
import json
from scipy.integrate import solve_ivp

from glob_var.FVM.FVM_RHS import FVM_RHS

current_dir = os.path.dirname(__file__)
project_root = os.path.abspath(os.path.join(current_dir, '..', '..'))

try:
    with open(project_root + '/glob_var/global_variables.json') as f:
        GV = json.load(f)
except FileNotFoundError:
    print("Global variables json not found!")

def make_step(h, i, args):
    """
    Try, excepts are for the BCs, as q_minus won't be computable for BCs at the start and q_plus won't be computable
    for BCs at the end.
    """

    _, dx, pwr, Q, A, n, _ = args

    Dx = 1 / dx ** 3
    try:
        disjoint_pressure_term = 3 * A * (2/(h[i+1]+h[i])) * ((h[i+1]-h[i])/dx)
        non_linear_term = ((h[i] + h[i + 1]) / 2) ** pwr
        q_plus = Dx * non_linear_term * (-h[i - 1] + 3 * h[i] - 3 * h[i + 1] + h[i + 2]) + disjoint_pressure_term + h[i]
    except IndexError:
        q_plus = 0

    try:
        disjoint_pressure_term = 3 * A * (2 / (h[i] + h[i-1])) * ((h[i] - h[i-1]) / dx)
        non_linear_term = ((h[i] + h[i - 1]) / 2) ** pwr
        q_minus = Dx * non_linear_term * (-h[i - 2] + 3 * h[i - 1] - 3 * h[i] + h[i + 1]) + disjoint_pressure_term + h[i - 1]
    except IndexError:
        q_minus = 0

    return q_plus, q_minus

if __name__ == '__main__':
    A = 0.13
    args = [make_step, GV['dx'], 3, GV['Q'], A, None, False]
    h_initial = np.ones(GV['N']) * GV['h0']
    min_t = GV['t-span'][f"{GV['L']}"]
    t_span = (0, min_t)

    sol = solve_ivp(fun=FVM_RHS, args=(args,), y0=h_initial, t_span=t_span, method='BDF', rtol=1e-6, atol=1e-8)

    print(sol.status)
    print(sol.success)
    print(sol.message)

    plt.plot(GV['x'], sol.y[:, -1])
    plt.title(f"Graph showing startup-flow solve of Newtonian fluid\nwith $Q={GV['Q']}$, $t={t_span[1]}$, $A={A}$")
    plt.grid(True)
    plt.xlabel('Surface Length $(x)$')
    plt.ylabel('Film Height $(y)$')
    plt.show()





# def make_step_alt_DP_FERRAN(h, i, args):
#     """
#     Try, excepts are for the BCs, as q_minus won't be computable for BCs at the start and q_plus won't be computable
#     for BCs at the end.
#     """
#
#     _, dx, pwr, Q, A, n = args
#
#     Dx = 1 / dx ** 3
#     DP = lambda h: A / (6 * np.pi * h ** 3)
#     try:
#         disjoint_pressure_term = (-DP(h[i] + DP(h[i+2])))/dx
#         non_linear_term = ((h[i] + h[i + 1]) / 2) ** pwr
#         q_plus = Dx * non_linear_term * (-h[i - 1] + 3 * h[i] - 3 * h[i + 1] + h[i + 2] - disjoint_pressure_term) + h[i]
#     except IndexError:
#         q_plus = 0
#
#     try:
#         disjoint_pressure_term = (-DP(h[i-2] + DP(h[i])))/dx
#         non_linear_term = ((h[i] + h[i - 1]) / 2) ** pwr
#         q_minus = Dx * non_linear_term * (-h[i - 2] + 3 * h[i - 1] - 3 * h[i] + h[i + 1] - disjoint_pressure_term) + h[
#             i - 1]
#     except IndexError:
#         q_minus = 0
#
#     return q_plus, q_minus

