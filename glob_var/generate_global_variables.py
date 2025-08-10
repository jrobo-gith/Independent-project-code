import json
import numpy as np

GV = {
    'N': 300,                               # Number of points in space,
    'large-N': 1875,                        # Large N to satisfy large-L,
    'L': 16,                                # Length of surface, used when length isn't being varied,
    'large-L': 100,                         # Large L that all pinch should fall within
    'h0': 1.0,                              # Height of film at h=0,
    'Q': 0.75,                              # Flux at h=0, used when flux isn't being varied,
    't-span': (0, 1_000_000),               # Arbitrarily large such that the ODE reaches a steady state

    'Q-list': [0.1, 0.25,
               0.5, 0.75,
               0.9, 0.95],                  # List of fluxes to be tested when varying flux,
    'L-list': [8, 10,
               16, 32,
               64, 80,
               180, 360],                    # List of lengths to be tested when varying length scales for validation
    'n-list': [0.25, 0.5,
               0.75, 1.0,
               1.25, 1.4],                  # List of n's to be testing when varying rheology using the power-law,
    'colors': ['red', 'green',
               'blue', 'black',
               'purple', 'cyan',
               'orange', 'pink'],            # List of colors for plotting
    'cmap': 'cool'
}
GV['dx'] = GV['L'] / GV['N']
GV['large-dx'] = GV['large-L'] / GV['large-N']
x = np.linspace(0, GV['L'], GV['N'])
GV['x'] = list(x)
large_x = np.linspace(0, GV['large-L'], GV['large-N'])
GV['large-x'] = list(large_x)


file = 'global_variables.json'
with open(file, 'w') as f:
    json.dump(GV, f)

print("Global variables updated successfully!")