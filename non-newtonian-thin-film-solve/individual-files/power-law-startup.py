import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp

N = 100
L = 10
dx = L/N
n = 0.1

h_0 = 1.0
Q = 0.8

x = np.linspace(0, L, N)
h_initial = np.ones(N) * h_0
h_initial[0] = h_0

t_span = (0, 100)

non_linear_hs = []
third_orders = []
third_order_signs = []
q_pluses = []

def make_step(h, i, n, print_except=False):
    """
    (Try, except)'s are for the BCs, as q_minus won't be computable for BCs at the start and q_plus won't be computable
    for BCs at the end.
    The bool 'print_except' is for making sure anything inside the for loop is not displaying a 0.
    """

    DX = 1/dx**3
    epsilon = 0

    try:
        non_linear_h = (0.5 * (h[i] + h[i+1]))**((2*n + 1)/n) + epsilon
        third_order = abs(-h[i-1] + 3*h[i] - 3*h[i+1] + h[i+2] + epsilon) ** (1/ n)
        third_order_sign = np.sign(-h[i-1] + 3*h[i] - 3*h[i+1] + h[i+2])
        advection_term = h[i]

        q_plus = DX  * non_linear_h * third_order_sign * third_order + advection_term

        non_linear_hs.append(non_linear_h)
        third_orders.append(third_order)
        third_order_signs.append(third_order_sign)
        q_pluses.append(q_plus)

    except IndexError:
        q_plus = 0
        if print_except: print("Q plus is 0!")

    try:
        non_linear_h = (0.5 * (h[i] + h[i-1])) ** ((2*n+1)/n) + epsilon
        third_order = abs(-h[i-2] + 3*h[i-1] - 3*h[i] + h[i+1] + epsilon) ** (1 / n)
        third_order_sign = np.sign(-h[i-2] + 3*h[i-1] - 3*h[i] + h[i+1])
        advection_term = h[i-1]

        q_minus = DX * non_linear_h * third_order_sign * third_order + advection_term

    except IndexError:
        q_minus = 0
        if print_except: print("Q minus is 0!")

    return q_plus, q_minus


def FVM(t, h):
    h = h.copy()
    dhdt = np.zeros_like(h)

    # i = 0
    h[0] = h_0

    # i = 1
    q_plus, q_minus = make_step(h=h, i=1, n=n)
    dhdt[1] = - (q_plus - Q) / dx

    # i = N - 2
    q_plus, q_minus = make_step(h=h, i=N-2, n=n)
    dhdt[N-2] = - (h[N-2] - q_minus) / dx

    # i = N - 1
    h[N-1] = h[N-2]
    dhdt[N-1] = dhdt[N-2]

    for i in range(2, N-2):
        q_plus, q_minus = make_step(h, i, n, print_except=True)
        dhdt[i] = -(q_plus - q_minus) / dx

    return dhdt

try:
    sol = solve_ivp(fun=FVM, y0=h_initial, t_span=t_span, method='BDF', rtol=1e-6, atol=1e-8)

    print(sol.status)
    print(sol.success)
    print(sol.message)

    fig, ax = plt.subplots(nrows=1, ncols=1, figsize=(8, 10))

    # ax.plot(x, sol.y[:, 0], label=f'Initial, t={t_span[0]}')
    ax.plot(x, sol.y[:, -1], label=f'End, t={t_span[1]}')
    ax.set_ylim(0, 1)
    ax.grid(True)
    ax.legend()
    ax.set_title(f"Startup flow with $h_0$={h_0}, Q={Q} and n={n}")
    ax.set_ylabel('height$(h)$')
    ax.set_xlabel('Length $(x)$')
    fig.show()

except ValueError:
    print("Value Error")

    non_linear_hs = np.array(non_linear_hs)
    third_orders = np.array(third_orders)
    third_order_signs = np.array(third_order_signs)
    q_pluses = np.array(q_pluses)

    fig, ax = plt.subplots(nrows=2, ncols=2, figsize=(8, 10))
    fig.suptitle("Term values for $q_{i+1/2}$ over time steps with smoothing\ n=0.99")

    ax[0, 0].scatter(non_linear_hs)
    ax[0, 0].set_title(f'Non-linear term')
    ax[0, 0].set_xlabel('Time steps $(t)$')

    ax[0, 1].scatter(third_orders)
    ax[0, 1].set_title(f'Third order term')
    ax[0, 1].set_xlabel('Time steps $(t)$')

    ax[1, 0].hist(third_order_signs)
    ax[1, 0].set_title(f'Third order sign histogram')

    ax[1, 1].scatter(q_pluses)
    ax[1, 1].set_title(f'Q pluses')
    ax[1, 1].set_xlabel('Time steps $(t)$')

    fig.tight_layout()
    fig.show()
