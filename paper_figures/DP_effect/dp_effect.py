import numpy as np
import matplotlib.pyplot as plt
import json

try:
    with open('../../glob_var/global_variables.json') as f:
        GV = json.load(f)
except FileNotFoundError:
    print("Global variables json not found!")

N = 1000
h = np.linspace(0.8, 1, N)

calc_DP = lambda h, A, n: ((3 * A)/ h**4) ** (1/n)
calc_DP_NL = lambda h, A, n: (((3 * A)/ h**4) ** (1/n)) * (h**((2*n+1)/n))
calc_NL = lambda h, n: (h**((2*n+1)/n))
A = 0.16

thinning = calc_DP(h=h, A=A, n=0.8)
thickening = calc_DP(h=h, A=A, n=1.2)

thinning_NL_DP = calc_DP_NL(h=h, A=A, n=0.8)
thickening_NL_DP = calc_DP_NL(h=h, A=A, n=1.2)

thinning_NL = calc_NL(h=h, n=0.8)
thickening_NL = calc_NL(h=h, n=1.2)

fig, ax = plt.subplots(ncols=2, figsize=(10, 8))
ax[0].plot(thinning, h, label="Thinning", color='k')
ax[0].plot(thickening, h, label="Thickening", color='k', linestyle='--')
ax[0].legend()
ax[0].grid(True)
ax[0].set_xlabel("DP Strength")
ax[0].set_ylabel("Film height $(h)$")


ax[1].plot(thinning_NL_DP, h, label="Thinning Non-linear", color='coral')
ax[1].plot(thickening_NL_DP, h, label="Thickening Non-linear", color='coral', linestyle='--')
ax[1].legend()
ax[1].grid(True)
ax[1].set_xlabel("DP Strength")
ax[1].set_ylabel("Film height $(h)$")
fig.savefig("dp_effect_non_lin.png")