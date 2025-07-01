# This code utilises central differences to solve most of the equation, but we don't have enough BC's to solve them all,
# so we use forward differences to solve for a couple of boundary conditions.

import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp

N = 1_00
L = 4
dx = 1.0

t_span = 10
x = np.linspace(0, L, N+1)

Q = 0.7
h_0 = 1.0

# Define initial conditions
y0 = h_0 * np.ones(N+1)

def f(t, h):
    h = h.copy()
    dhdt = np.zeros_like(h)

    # Include BCs
    h[0] = h_0
    h[1] = (h[0] + 3*h[2] - h[3] - (((Q-h[0])*dx**3)/h[0]**3)) / 3
    h[N-1] = h[N]
    h[N] = (3*h[N-2] - h[N-3])/2

    # Implement central differences
    for i in range(3, N-3):
        fi_P1 = ((h[i+1]**3)/(2*dx**3)) * (h[i+3] - 2*h[i+2] + 2*h[i] - h[i-1])
        fi_M1 = ((h[i-1]**3)/(2*dx**3)) * (h[i+1] - 2*h[i] + 2*h[i-2] - h[i-3])
        advection_term = h[i] - h[i-1]
        dhdt[i] = -(fi_P1 - fi_M1)/(dx) - advection_term/dx

    # Where i = 2
    i = 2
    fi_pos = ((h[i+1]**3)/(2*dx**3)) * (h[i+3] - 2*h[i+2] + 2*h[i] - h[i-1])    # CTD
    fi_neg = ((h[i-1]**3)/(dx**3)) * (h[i+2] - 3*h[i+1] + 3*h[i] - h[i-1])      # FWD
    advection_term = h[i] - h[i-1]
    dhdt[i] = -(fi_pos-fi_neg)/(dx) - advection_term/dx

    # Where i = N-2
    i = N-2
    fi_pos = ((h[i+1]**3)/(dx**3)) * (h[i-2] - 3*h[i-1] + 3*h[i] - h[i+1])      # FWD
    fi_neg = ((h[i-1]**3)/(2*dx**3)) * (h[i+1] - 2*h[i] + 2*h[i-2] - h[i-3])    # CTD
    advection_term = h[i] - h[i-1]
    dhdt[i] = -(fi_pos-fi_neg)/(dx) - advection_term/dx

    # plt.plot(dhdt)
    # plt.title(f"T={t}")
    # plt.show()
    return dhdt

def main(y0, t_span):
    sol = solve_ivp(f, t_span, y0, method='RK45', rtol=1e-6, atol=1e-8)
    print(sol.success)
    print(sol.status)
    print(sol.message)
    plt.plot(x, sol.y[:, -1])
    plt.show()

if __name__ == '__main__':
    main(y0=y0, t_span=[0, t_span])
