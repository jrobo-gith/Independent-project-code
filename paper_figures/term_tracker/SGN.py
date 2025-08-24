import numpy as np
import matplotlib.pyplot as plt

newt = np.load("../timeseries_graphs/data/newt_sol_dp.npy")
newt_index = np.where(newt[:, -1] == min(newt[:, -1]))
thic = np.load("../timeseries_graphs/data/thic_sol_dp.npy")
thic_index = np.where(thic[:, -1] == min(thic[:, -1]))

newt = np.load("newtonian_collector/SGN_DP.npy")
thin = np.load("thinning_collector/SGN_DP.npy")
thic = np.load("thickening_collector/SGN_DP.npy")
x = np.linspace(0, 16, 300)

time = -1

fig, ax = plt.subplots(nrows=3, figsize=(10,10))

ax[0].plot(x, newt[time, 0, :], label="Q_minus", color='k')
ax[0].plot(x, newt[time, 1, :], label="Q_plus", color='b')
# ax[0].plot(x, (newt[time, 1, :]-newt[time, 0, :]), label='Q_plus-Q_minus', color='g', linestyle='--')
ax[0].axvline(x=x[newt_index], linestyle='-.', color='r', label='Pinch Point')
ax[0].set_title("Newtonian Fluid $(n=1.0)$", fontsize=16)
ax[0].set_xlabel("Surface Length $(x)$", fontsize=14)

ax[1].plot(x, thin[time, 0, :], label="Q_minus", color='k')
ax[1].plot(x, thin[time, 1, :], label="Q_plus", color='b')
# ax[1].plot(x, thin[time, 1, :]-thin[time, 0, :], label='Q_plus-Q_minus', color='g', linestyle='--')
ax[1].set_xlabel("Surface Length $(x)$", fontsize=14)
ax[1].set_title("Shear-Thinning $(n=0.8)$", fontsize=16)

ax[2].plot(x, thic[time, 0, :], label="Q_minus", color='k')
ax[2].plot(x, thic[time, 1, :], label="Q_plus", color='b')
# ax[2].plot(x, thic[time, 1, :]-thic[time, 0, :], label='Q_plus-Q_minus', color='g', linestyle='--')
ax[2].axvline(x=x[thic_index], linestyle='-.', color='r', label='Pinch Point')
ax[2].set_title("Shear-Thickening $(n=1.2)$", fontsize=16)
ax[2].set_xlabel("Surface Length $(x)$", fontsize=14)

ax[0].legend(loc='upper left')
ax[1].legend(loc='upper left')
ax[2].legend(loc='upper left')

ax[0].grid(True)
ax[1].grid(True)
ax[2].grid(True)

# ax[0].set_xlim(11.5, 12.5)
# ax[1].set_xlim(8, 16)
# ax[2].set_xlim(8.5, 8.51)

# ax[0].set_ylim(-20, 35)
# ax[1].set_ylim(-0.0000002, 0.0000002)
# ax[2].set_ylim(-0.1, 0.15)

ax[0].set_ylabel("Sign", fontsize=14)
ax[1].set_ylabel("Sign", fontsize=14)
ax[2].set_ylabel("Sign", fontsize=14)


plt.tight_layout()
fig.suptitle("Sign of $(h_{xxx} + \Pi_x)$ at $Q=0.75$, $A=0.15$", fontsize=16, y=1.05)
fig.savefig("graphs/SGN.png", bbox_inches='tight')
