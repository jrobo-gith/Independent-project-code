import numpy as np
from scipy.integrate import solve_ivp
import matplotlib.pyplot as plt

h_0 = 1.0
N = 1000
L = 4
dx = 1
Q = 0.8
x = np.linspace(0, L, N)

def f(t, h):
    h = h.copy()
    dhdt = np.zeros_like(h)

    h[0] = h_0

    for i in range(3, N-3):
        non_linear_term = (1/(16*dx**3)) * ((h[i]+h[i+1])**3 - (h[i]+h[i-1])**3)
        q_halves = -h[i-3] + 3*h[i-1] - 3*h[i+1] + h[i+3]
        advection_term = 2*h[i-1]

        dhdt[i] = -non_linear_term * q_halves - advection_term

    return dhdt

h_init = np.ones(N)*h_0
t_span = (0, 1)

solution = solve_ivp(f, t_span, h_init)

print(solution.status)
print(solution.success)
print(solution.message)


fig = plt.figure()
plt.plot(x, solution.y[:, -1], label=f't={t_span[1]}')
plt.plot(x, solution.y[:, 0], label='t=0')
plt.title(f"Initial guess (little BC work) 02/07/25 \nh_0={h_0}, N={N} L={L}, dx={dx}  Q={Q}, tend={t_span[1]}")
plt.xlabel("Length (L)")
plt.ylabel("Height (h)")
plt.legend()
plt.show()

fig.savefig("finite-vol-images/failed/initial.png")