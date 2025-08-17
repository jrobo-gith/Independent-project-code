import numpy as np
import matplotlib.pyplot as plt
import json
from scipy.interpolate import interp1d
import gc

try:
    with open('../../glob_var/global_variables.json') as f:
        GV = json.load(f)
except FileNotFoundError:
    print("Global variables json not found!")

terms = ["NLT", "TOT", "ADV", "DPT"]
newtonian_directory = "../term_tracker/newtonian_collector/"
thinning_directory = "../term_tracker/thinning_collector/"
thickening_directory = "../term_tracker/thickening_collector/"

total_array = np.load(newtonian_directory + "sol_y_dp.npy")[1:298, :]

def calculate_percentage(total_array:np.ndarray, term_str:str, directory:str) -> np.ndarray:
    """Takes any term and calculates the percentage contribution to the overall array over time"""
    term_arr = np.load(directory+term_str+"_dp.npy")
    term_arr_av = ((term_arr[:, 1, 1:298] - term_arr[:, 0, 1:298]) / GV['dx']).transpose(1, 0)

    f = interp1d(np.linspace(0, 1, term_arr_av.shape[1]), term_arr_av)
    term_arr_av_interp = f(np.linspace(0, 1, total_array.shape[1]))

    assert total_array.shape == term_arr_av_interp.shape, print(total_array.shape, term_arr_av.shape)

    del f, term_arr_av, term_arr
    gc.collect()

    return (term_arr_av_interp/total_array) * 100

newtonian_terms = []
thinning_terms = []
thickening_terms = []

for term in terms:
    perc = calculate_percentage(total_array, term, newtonian_directory)
    newtonian_terms.append(perc)

    perc = calculate_percentage(total_array, term, thinning_directory)
    thinning_terms.append(perc)

    perc = calculate_percentage(total_array, term, thickening_directory)
    thickening_terms.append(perc)

fig, ax = plt.subplots(nrows=3, figsize=(10, 10))

[ax[0].plot(GV['x'][1:298], newtonian_terms[i][:, -1], label=f"{terms[i]}") for i in range(len(newtonian_terms))]
[ax[1].plot(GV['x'][1:298], thinning_terms[i][:, -1], label=f"{terms[i]}") for i in range(len(newtonian_terms))]
[ax[2].plot(GV['x'][1:298], thickening_terms[i][:, -1], label=f"{terms[i]}") for i in range(len(newtonian_terms))]

ax[0].set_ylim(-200, 200)
ax[1].set_ylim(-200, 200)
ax[2].set_ylim(-200, 200)

ax[0].legend()
ax[1].legend()
ax[2].legend()

fig.show()
