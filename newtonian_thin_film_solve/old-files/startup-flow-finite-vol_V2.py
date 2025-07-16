import numpy as np
from scipy.integrate import solve_ivp
import matplotlib.pyplot as plt

N = 1000
L = 4
dx = 0.1

h_0 = 1.0
Q = 0.9

def f(t, h):
    h = h.copy()
    dhdt = np.zeros_like(h)

    h[0] = h_0
    h[1] = Q
    h[2] = Q

    for i in range(3, N-3):
        outside_dx = 1/(16 * dx**4)
        q_plus_half = (h[i] + h[i+1])**3 * (-h[i-3] + 2*h[i-2] - 2*h[i] + h[i+1])
        q_minus_half = (h[i] + h[i-1])**3 * (-h[i-1] + 2*h[i] - 2*h[i+2] + h[i+3])
        advection_term = (2*h[i-1])/dx
        dhdt[i] = -outside_dx * (q_plus_half - q_minus_half) - advection_term

    return dhdt
x = np.linspace(0, L, N)
h_initial = np.zeros(N)
h_initial[0] = h_0
h_initial[1] = Q
h_initial[2] = Q

t_span = (0, 10)

sol = solve_ivp(fun=f,
                y0=h_initial,
                t_span=t_span,
                method='RK45')

print(sol.status)
print(sol.success)
print(sol.message)


fig, ax = plt.subplots()

ax.plot(x, sol.y[:, 0], label='initial t=0')
ax.plot(x, sol.y[:, -1], label=f'end t={t_span[1]}')
ax.legend()
ax.grid(True)
plt.show()
