import numpy as np
import matplotlib.pyplot as plt

newt = np.load("newtonian_collector/MAG_DP.npy")
thin = np.load("thinning_collector/MAG_DP.npy")
thic = np.load("thickening_collector/MAG_DP.npy")
x = np.linspace(0, 16, 300)

time = -1


fig, ax = plt.subplots(nrows=3, figsize=(10,10))

ax[0].plot(x, newt[time, 0, :], label="Q_minus", color='k')
ax[0].plot(x, newt[time, 1, :], label="Q_plus", color='b')
ax[0].plot(x, (newt[time, 1, :]-newt[time, 0, :]), label='difference', color='r', linestyle='--')

ax[1].plot(x, thin[time, 0, :], label="Q_minus", color='k')
ax[1].plot(x, thin[time, 1, :], label="Q_plus", color='b')
ax[1].plot(x, thin[time, 1, :]-thin[time, 0, :], label='difference', color='r', linestyle='--')

ax[2].plot(x, thic[time, 0, :], label="Q_minus", color='k')
ax[2].plot(x, thic[time, 1, :], label="Q_plus", color='b')
ax[2].plot(x, thic[time, 1, :]-thic[time, 0, :], label='difference', color='r', linestyle='--')
ax[2].set_title("Shear-Thickening")

ax[0].legend(loc='upper left')
ax[1].legend(loc='upper left')
ax[2].legend(loc='upper left')

ax[0].grid(True)
ax[1].grid(True)
ax[2].grid(True)

ax[0].set_xlim(11.5, 12.5)
ax[1].set_xlim(8.25, 8.75)
ax[2].set_xlim(8, 9)

ax[0].set_ylim(-20, 30)
ax[1].set_ylim(-0.0000001, 0.0000001)
ax[2].set_ylim(-10, 10)

fig.suptitle("Tend to 1 => TOT\nTend to 0 => DPT")

fig.savefig("Test_MAG.png")
