import json
import numpy as np

t_min = int(np.load('../cluster_running/data/min_t/min_t.npy'))

GV = {
    'N': 350,                               # Number of points in space,
    'L': 15,                                # Length of surface, used when length isn't being varied,
    'h0': 1.0,                              # Height of film at h=0,
    'Q': 0.75,                              # Flux at h=0, used when flux isn't being varied,
    't-span': (0, t_min),                   # Min and max time for which startup flow regimes are run,

    'Q-list': [0.1, 0.25,
               0.5, 0.75,
               0.9, 0.95],                  # List of fluxes to be tested when varying flux,
    'L-list': [8, 10,
               16, 32,
               64, 80],                     # List of lengths to be tested when varying length scales for validation
    'n-list': [0.25, 0.5,
               0.75, 1.0,
               1.25, 1.4],                  # List of n's to be testing when varying rheology using the power-law,
    'colors': ['red', 'green',
               'blue', 'black',
               'purple', 'cyan']            # List of colors for plotting
}
GV['dx'] = GV['L'] / GV['N']
x = np.linspace(0, GV['L'], GV['N'])
GV['x'] = list(x)


file = 'global_variables.json'
with open(file, 'w') as f:
    json.dump(GV, f)

print("Global variables updated successfully!")