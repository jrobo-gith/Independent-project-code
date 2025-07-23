import matplotlib.pyplot as plt
import numpy as np

r = 2
c = 1

fig, ax = plt.subplots(nrows=r, ncols=c)

for i in range(r):
    for j in range(c):
        ax[i][j].set_xlabel("$x$")

fig.tight_layout()
fig.show()