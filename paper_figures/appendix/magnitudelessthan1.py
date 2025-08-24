import matplotlib.pyplot as plt
import numpy as np

h = np.linspace(0.1, 1, 1000)

calc_mag = lambda h, n: abs(h**(-4))**(1/n)
calc_mag_simple = lambda h, n: abs(h)**(1/n)

reverse_ns = [0.2, 0.5, 0.95]
ls = ['-', '--', '-.']

fig, ax = plt.subplots(figsize=(10, 10))

ax.set_title("Effect when |x| < 1 at different ns")
ax.set_xlabel("Term Magnitude")
ax.set_ylabel("Film height")
ax.grid(True)
ax.plot(calc_mag_simple(h, 1.0), h, color='green', label=f"Newtonian: n=1.0")

for i, reverse_n in enumerate(reverse_ns):
    ax.plot(calc_mag_simple(h, 1.0 - reverse_n), h, color='k', linestyle=ls[i], label=f"Thinning: n = {np.round(1-reverse_n, 2)}")

for i, reverse_n in enumerate(reverse_ns):
    ax.plot(calc_mag_simple(h, 1.0 + reverse_n), h, color='b', linestyle=ls[i],
            label=f"Thickening: n = {1.0 + reverse_n}")
ax.legend()
plt.tight_layout()
plt.show()