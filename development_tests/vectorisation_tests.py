## This file is used for testing to see if i can vectorise the for loop to optimise the computation.

import matplotlib.pyplot as plt
import numpy as np
from scipy.integrate import solve_bvp
from scipy.integrate import solve_ivp

GV = {
    'N': 200,  # Number of points in space,
    'L': 15,  # Length of surface, used when length isn't being varied,
    'h0': 1.0,  # Height of film at h=0,
    'Q': 0.75,  # Flux at h=0, used when flux isn't being varied,
    't-span': (0, 100),  # Min and max time for which startup flow regimes are run,

    'Q-list': [0.1, 0.25,
               0.5, 0.75,
               0.9, 0.95],  # List of fluxes to be tested when varying flux,
    'L-list': [8, 10,
               16, 32,
               64, 80],  # List of lengths to be tested when varying length scales for validation
    'n-list': [0.25, 0.5,
               0.75, 1.0,
               1.25, 1.4],  # List of n's to be testing when varying rheology using the power-law,
    'colors': ['red', 'green',
               'blue', 'black',
               'purple', 'cyan']  # List of colors for plotting
}
GV['dx'] = GV['L'] / GV['N']
GV['x'] = np.linspace(0, GV['L'], GV['N'])


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


def newt_make_step(h, i, args):
    """
    Try, excepts are for the BCs, as q_minus won't be computable for BCs at the start and q_plus won't be computable
    for BCs at the end.
    """

    _, dx, pwr, Q, _, n = args

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


try:
    args = [newt_make_step, GV['dx'], 3, GV['Q'], None, None]
    h_initial = np.ones(GV['N']) * GV['h0']
    t_span = GV['t-span']

    sol = solve_ivp(fun=FVM_RHS, args=(args,), y0=h_initial, t_span=t_span, method='BDF', rtol=1e-6, atol=1e-8)

    print(sol.status)
    print(sol.success)
    print(sol.message)

except ValueError:
    print("Value Error")

plt.plot(GV['x'], sol.y[:, -1])
plt.title(f"Graph showing startup-flow solve of Newtonian fluid\nwith $Q={GV['Q']}$, $t={GV['t-span'][1]}$")
plt.grid(True)
plt.xlabel('Surface Length $(x)$')
plt.ylabel('Film Height $(y)$')
plt.show()