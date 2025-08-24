import numpy as np
import matplotlib.pyplot as plt

newt = np.load("../timeseries_graphs/data/newt_sol_dp.npy")
newt_index = np.where(newt[:, -1] == min(newt[:, -1]))
thic = np.load("../timeseries_graphs/data/thic_sol_dp.npy")
thic_index = np.where(thic[:, -1] == min(thic[:, -1]))

newt = np.load("newtonian_collector/RAT_DP.npy")
thin = np.load("thinning_collector/RAT_DP.npy")
thic = np.load("thickening_collector/RAT_DP.npy")
x = np.linspace(0, 16, 300)

times = [-500, -400, -300, -250, -200, -150, -100, -75, -50, -25, -10, -1]

for time in times:

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
    ax[2].set_xlim(8.0, 9.0)

    # ax[0].set_ylim(-20, 35)
    # ax[1].set_ylim(-0.0000002, 0.0000002)
    # ax[2].set_ylim(-0.1, 0.15)

    ax[0].set_ylabel("Dominant Term", fontsize=14)
    fig.text(-0.015, 0.73, "$(\Pi_x)$", rotation='horizontal', fontsize=14)
    fig.text(-0.015, 0.94, "$(h_{xxx})$", rotation='horizontal', fontsize=14)

    ax[1].set_ylabel("Dominant Term", fontsize=14)
    fig.text(-0.015, 0.4, "$(\Pi_x)$", rotation='horizontal', fontsize=14)
    fig.text(-0.015, 0.6, "$(h_{xxx})$", rotation='horizontal', fontsize=14)

    ax[2].set_ylabel("Dominant Term", fontsize=14)
    fig.text(-0.015, 0.07, "$(\Pi_x)$", rotation='horizontal', fontsize=14)
    fig.text(-0.015, 0.28, "$(h_{xxx})$", rotation='horizontal', fontsize=14)

    plt.tight_layout()
    fig.suptitle("Ratio of of terms $h_{xxx}$ to $\Pi_x$ at $Q=0.75$, $A=0.15$", fontsize=16, y=1.05)

    ax[0].axhline(y=0.15, color='orange', linestyle='-.')
    ax[1].axhline(y=0.15, color='orange', linestyle='-.')
    ax[2].axhline(y=0.15, color='orange', linestyle='-.')

    fig.show()
    # fig.savefig("graphs/RAT.png", bbox_inches='tight')
