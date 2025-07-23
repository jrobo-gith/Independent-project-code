import numpy as np
import matplotlib.pyplot as plt
import os
import json
current_dir = os.path.dirname(__file__)
project_root = os.path.abspath(os.path.join(current_dir, '..', '..'))

try:
    with open(project_root + '/glob_var/global_variables.json') as f:
        GV = json.load(f)
except FileNotFoundError:
    print("Global variables json not found!")

data = np.load('../data/varying_Q_A/varying_Q_A_data.npy')
table = np.load('../data/varying_Q_A/varying_Q_A_success_table.npy')


fig, ax = plt.subplots(nrows=1, ncols=3, figsize=(10, 5))

[ax[0].plot(GV['x'], data[0, 0, i, :]) for i in range(data.shape[2])]
[ax[1].plot(GV['x'], data[1, 0, i, :]) for i in range(data.shape[2])]
[ax[2].plot(GV['x'], data[2, 0, i, :]) for i in range(data.shape[2])]

ax[0].set_title("THINNING")
ax[1].set_title("NEWTONIAN")
ax[2].set_title("THICKENING")

fig.show()

print(table)