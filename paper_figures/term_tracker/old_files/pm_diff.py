import numpy as np
import matplotlib.pyplot as plt
import json
import matplotlib as mpl

try:
    with open('../../../glob_var/global_variables.json') as f:
        GV = json.load(f)
except FileNotFoundError:
    print("Global variables json not found!")

ADV = np.load("thinning_collector/ADV.npy")
ADV_dp = np.load("thinning_collector/ADV_dp.npy")
T = np.load("thinning_collector/T.npy")
T_dp = np.load("thinning_collector/T_dp.npy")

print(ADV.shape)