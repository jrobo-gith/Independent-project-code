import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp

N = 1_000
L = 400
dx = L/N
t_span = 10

Q = 0.8
h_0 = 1.0

h_initial = np.ones(N+1) * h_0
x = np.linspace(0, L, len(h_initial))

def f(t, h):
    h = h.copy()
    dhdt = np.zeros_like(h)

    # Include BCs
    h[0] = h_0
    h[1] = (h[0] + 3*h[2] - h[3] - (((Q-h[0])*dx**3)/h[0]**3)) / 3
    h[N-1] = h[N]
    h[N] = (3*h[N-2] - h[N-3])/2

    for i in range(2, N):
        fwd_term_1 = h[i+1]**3 * (h[i+1]-3*h[i]+3*h[i-1]-3*h[i-2])
        fwd_term_2 = h[i]**3 * (h[i] - 3*h[i-1]+3*h[i-2] - 3*h[i-3])
        advection_term = (h[i+1]-h[i])/dx
        dhdt[i] = (1/dx**4) * (fwd_term_1 - fwd_term_2) - advection_term

    return dhdt

def main(y0, t_span):
    sol = solve_ivp(f, t_span, y0, method='RK45', rtol=1e-6, atol=1e-8)
    print(sol.success)
    print(sol.status)
    print(sol.message)
    plt.plot(x, sol.y[:, -1])
    plt.show()

if __name__ == '__main__':
    main(y0=h_initial, t_span=[0, t_span])