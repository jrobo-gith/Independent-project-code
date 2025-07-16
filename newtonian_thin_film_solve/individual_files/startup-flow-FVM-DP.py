import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp

N = 100
L = 10
dx = L/N

h_0 = 1.0
Q = 0.75

x = np.linspace(0, L, N)
h_initial = np.zeros(N)
h_initial[0] = h_0

t_span = (0, 100)

# Disjoint pressure variables
A = 0
h_star = 0.005
n = 4
m = 3
sigma = 0.004

def make_step(h, i, pwr=3):
    """
    Try, excepts are for the BCs, as q_minus won't be computable for BCs at the start and q_plus won't be computable
    for BCs at the end.
    """
    Dx = 1/dx**3
    try:
        non_linear_term =  ((h[i] + h[i+1])/2)**pwr
        DP = A * ((h_star/h[i])**n - (h_star/h[i])**m)
        q_plus = Dx * non_linear_term * (sigma * -h[i-1] + 3*h[i] - 3*h[i+1] + h[i+2] + DP) + h[i]
    except IndexError: q_plus = 0

    try:
        non_linear_term =  ((h[i] + h[i-1])/2)**pwr
        DP = A * ((h_star / h[i-1]) ** n - (h_star / h[i-1]) ** m)
        q_minus = Dx * non_linear_term * (sigma * -h[i-2] + 3*h[i-1] - 3*h[i] + h[i+1] + DP) + h[i-1]
    except IndexError: q_minus = 0

    return q_plus, q_minus

def f(t, h):
    h = h.copy()
    dhdt = np.zeros_like(h)

    # i = 0
    h[0] = h_0

    # i = 1
    q_plus, q_minus = make_step(h=h, i=1)
    dhdt[1] = - (q_plus - Q) / dx

    # i = N - 2
    q_plus, q_minus = make_step(h=h, i=N-2)
    dhdt[N-2] = - (h[N-2] - q_minus) / dx

    # i = N - 1
    h[N-1] = h[N-2]
    dhdt[N-1] = dhdt[N-2]

    for i in range(2, N-2):
        q_plus, q_minus = make_step(h, i)
        dhdt[i] = -(q_plus - q_minus) / dx

    return dhdt

try:
    sol = solve_ivp(fun=f, y0=h_initial, t_span=t_span, method='BDF', rtol=1e-6, atol=1e-8)

    print(sol.status)
    print(sol.success)
    print(sol.message)

    fig, ax = plt.subplots(nrows=1, ncols=1, figsize=(8, 10))

    ax.plot(x, sol.y[:, 0], label=f'Initial, t={t_span[0]}')
    ax.plot(x, sol.y[:, -1], label=f'End, t={t_span[1]}')
    print(sol.y[:, -1]==0)
    ax.set_ylim(0, 1)
    ax.grid(True)
    ax.legend()
    ax.set_title(f"Startup flow with $h_0$={h_0} and Q={Q}")
    ax.set_ylabel('height$(h)$')
    ax.set_xlabel('Length $(x)$')
    fig.show()

except ValueError:
    print("Value Error")
