import numpy as np
from scipy.integrate import solve_ivp
import matplotlib.pyplot as plt

N = 10
L = 4
dx = L/N

h_0 = 1.0
Q = 0.8

common_multiplier = 1/(8*dx**4)

def f(t, h):
    h = h.copy()
    dhdt = np.zeros_like(h) * h_0

    h[0] = h_0
    h[1] = (h[0] + 3*h[2] - h[3] - (((Q-h[0])*dx**3)/h[0]**3)) / 3
    h[-1] = (3 * h[N - 2] - h[N - 3]) / 2

    for i in range(3, N-3):

    return dhdt



x = np.linspace(0, L, N)
h_initial = np.ones(N)
h_initial[0] = h_0
h_initial[1] = Q
h_initial[2] = Q

t_span = (0, 1)

sol = solve_ivp(fun=f,
                y0=h_initial,
                t_span=t_span,
                method='BDF',
                rtol=1e-6,
                atol=1e-8)

print(sol.status)
print(sol.success)
print(sol.message)

fig, ax = plt.subplots(nrows=3, ncols=1, figsize=(8, 10))

ax[0].plot(x, sol.y[:, 0], label='initial t=0')
ax[0].legend()
ax[0].grid(True)

ax[1].plot(x, sol.y[:, -1], label=f'end t={t_span[1]}', color='orange')
ax[1].legend()
ax[1].grid(True)
ax[1].set_ylim([-h_0, h_0])

ax[2].plot(x, sol.y[:, -1], label=f'end t={t_span[1]}', color='orange')
ax[2].legend()
ax[2].grid(True)

fig.tight_layout()
fig.show()

