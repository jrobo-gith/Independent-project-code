import numpy as np
from scipy.integrate import solve_ivp
import matplotlib.pyplot as plt

N = 10
L = 4
dx = L / N

h_0 = 1.0
Q = 0.8

common_multiplier = 1 / (8 * dx ** 4)

def f(t, h):
    h = h.copy()
    dhdt = np.zeros_like(h) * h_0

    h[0] = h_0
    h[1] = ((dx ** 3) / 3) * (Q - h_0) + (1 / 3) * h_0 + h[2] - (1 / 3) * h[3]
    h[-1] = (3 * h[N - 3] - h[N - 4]) / 2
    h[-2] = h[-1]

    for i in range(3, N - 3):
        q_i_plus_half = common_multiplier * (h[i] + 3 * h[i + 1] - 3 * h[i + 2] + h[i + 3])
        q_i_minus_half = common_multiplier * (-h[i - 3] + 3 * h[i - 2] - h[i - 1] + h[i])
        advection_term = (2 * h[i - 1]) / dx

        dhdt[i] = q_i_plus_half - q_i_minus_half + advection_term

    # Where i = 2
    i = 2
    fi_pos = (1 / (2 * dx ** 3)) * (h[i + 3] - 2 * h[i + 2] + 2 * h[i] - h[i - 1])  # CTD
    fi_neg = (1 / (dx ** 3)) * (h[i + 2] - 3 * h[i + 1] + 3 * h[i] - h[i - 1])  # FWD
    advection_term = h[i] - h[i - 1]
    dhdt[i] = -(fi_pos - fi_neg) / (dx) - advection_term / dx

    # Where i = N-3
    i = N - 3
    fi_pos = (1 / (dx ** 3)) * (h[i - 2] - 3 * h[i - 1] + 3 * h[i] - h[i + 1])  # FWD
    fi_neg = (1 / (2 * dx ** 3)) * (h[i + 1] - 2 * h[i] + 2 * h[i - 2] - h[i - 3])  # CTD
    advection_term = h[i] - h[i - 1]
    dhdt[i] = -(fi_pos - fi_neg) / (dx) - advection_term / dx


    plt.plot(dhdt)
    plt.show()

    return dhdt


x = np.linspace(0, L, N)
h_initial = np.linspace(1, 0.5, N)

# h_initial[0] = h_0
# h_initial[1] = ((dx ** 3) / (3 * h_0 ** 3)) * (Q - h_0) + (1 / 3) * h_0 + h_initial[2] - (1 / 3) * h_initial[3]
# h_initial[-1] = (3 * h_initial[N - 3] - h_initial[N - 4]) / 2
# h_initial[-2] = h_initial[-1]

t_span = (0, 1)

sol = solve_ivp(fun=f,
                y0=h_initial,
                t_span=t_span,
                method='Radau',
                rtol=1e-8,
                atol=1e-10)

print(sol.status)
print(sol.success)
print(sol.message)

fig, ax = plt.subplots(nrows=4, ncols=1, figsize=(8, 10))

ax[0].plot(x, sol.y[:, 0], label='initial t=0')
ax[0].legend()
ax[0].grid(True)

ax[1].plot(x, sol.y[:, 10], label='1st timestep')
ax[1].legend()
ax[1].grid(True)

ax[2].plot(x, sol.y[:, 100], label='2nd timestep')
ax[2].legend()
ax[2].grid(True)

ax[3].plot(x, sol.y[:, 250], label='3rd timestep')
ax[3].legend()
ax[3].grid(True)

# ax[1].plot(x, sol.y[:, -1], label=f'end t={t_span[1]}', color='orange')
# ax[1].legend()
# ax[1].grid(True)
# ax[1].set_ylim([-h_0, h_0])
#
# ax[2].plot(x, sol.y[:, -1], label=f'end t={t_span[1]}', color='orange')
# ax[2].legend()
# ax[2].grid(True)

fig.tight_layout()
fig.show()

