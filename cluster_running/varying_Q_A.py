## This file will generate data for three difference power law fluids, where n < 1.0, n = 1.0 and n > 1.0. This will
# give us a gradient of stable, unstable and deformed scenarios at a high resolution. The variables looked at are the
# strength of the disjoint pressure (A) and the volume flux (Q).

## Import function from individual files
from non_newtonian_thin_film_solve.individual_files.power_law_dp import PL_DP_make_step
from glob_var.FVM.FVM_RHS import FVM_RHS

import numpy as np
import os
import json
from scipy.integrate import solve_ivp
import matplotlib.pyplot as plt

current_dir = os.path.dirname(__file__)
project_root = os.path.abspath(os.path.join(current_dir, '..'))

try:
    with open(project_root + '/glob_var/global_variables.json') as f:
        GV = json.load(f)
except FileNotFoundError:
    print("Global variables json not found!")

resolution = 500

ns = [0.75, 1.0, 1.2]
Qs = np.linspace(0.001, 0.95, resolution)
As = np.logspace(-5, 0, resolution)

# Take max time from data file
max_t = np.load('data/min_t.npy')



# if __name__ == '__main__':
#     n = 1.0
#     h_initial = np.ones(GV['N']) * GV['h0']
#     t_span = GV['t-span']
#     args = [PL_DP_make_step, GV['dx'], None, GV['Q'], 0.001, n]
#
#     sol = solve_ivp(fun=FVM_RHS, y0=h_initial, t_span=t_span, args=(args,), method='BDF', rtol=1e-6, atol=1e-8)
#
#     print(sol.status)
#     print(sol.success)
#     print(sol.message)
#
#     plt.plot(GV['x'], sol.y[:, -1])
#     plt.title(f"Graph showing startup-flow solve of Newtonian fluid\nwith $Q={GV['Q']}$, $t={GV['t-span'][1]}$, $n={n}$")
#     plt.grid(True)
#     plt.xlabel('Surface Length $(x)$')
#     plt.ylabel('Film Height $(y)$')
#     plt.show()
