import numpy as np
import matplotlib.pyplot as plt

newtonian = np.load("newtonian.npy")
thinning = np.load("thinning.npy")
# thickening = np.load("thickening.npy")

As = np.linspace(0, 0.5, 12)
Qs = np.linspace(0.1, 0.95, 12)

step = 2

Q_samples = Qs[1::step]
A_samples = As[1::step]

newt_samples = newtonian[1::step, 1::step]
thin_samples = thinning[1::step, 1::step]
# thic_samples = thickening[1::step, 1::step]

min_t_newt = min(newt_samples.flatten())
max_t_newt = max(newt_samples.flatten())
min_t_thin = min(thin_samples.flatten())
max_t_thin = max(thin_samples.flatten())

min_t = min(min_t_newt, min_t_thin)
max_t = max(max_t_newt, max_t_thin)

fig, ax = plt.subplots(nrows=2, ncols=3, figsize=(15, 10))
plt.subplots_adjust(hspace=0.0, wspace=0.0)
axes = np.array(ax).flatten()

for i, PLOT in enumerate(axes):
    PLOT.plot(A_samples, newt_samples[i, :], label="Newtonian", color='red', linewidth=2)
    PLOT.plot(A_samples, thin_samples[i, :], label="Thinning", color='blue', linewidth=2)
    # PLOT.plot(A_samples, thic_samples[i, :], label="Thickening")
    PLOT.legend(loc='upper right')
    PLOT.text(0.25, 85, f"$Q={np.round(Q_samples[i], 2)}$", ha="center", va="top", fontsize=16, bbox=dict(facecolor="white", alpha=1.0, edgecolor="none"))
    PLOT.set_ylim(0, max_t+10)
    PLOT.grid(True)
    PLOT.set_yticklabels([])
    if i <= 2: PLOT.set_xticklabels([]), PLOT.set_xlabel("")

axes[0].set_yticklabels(np.arange(0, max_t+10, 10))
axes[3].set_yticklabels(np.arange(0, max_t+10, 10))

fig.text(0.05, 0.5, "Time Taken $(s)$", ha="left", va="center", fontsize=14, rotation='vertical')
fig.text(0.5, 0.05, "Disjoint Pressure Strength $(A)$", ha="center", va="center", fontsize=14, rotation='horizontal')
fig.text(0.5, 0.93, "Graph showing time taken until a\nsteady state has been reached", ha="center", va="center", fontsize=16, rotation='horizontal')
fig.savefig("ttss.png", bbox_inches='tight')
