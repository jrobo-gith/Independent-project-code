import numpy as np
import json
from scipy.integrate import solve_ivp
from glob_var.FVM.FVM_RHS import FVM_RHS
from glob_var.FVM.stop_events import unstable, steady_state
from

with open("global_variables.json") as f:
    GV = json.load(f)

resolution = 100
Qs = np.linspace(0.1, 0.95, resolution)
As = np.linspace(0, 0.5, resolution)
data = np.zeros((len(Qs), len(As), GV['N']))

n = 1.0
linear = False
h_initial = np.ones(GV['N'])
t_span = GV['t-span']
unstable.terminal = True
steady_state.terminal = True

for i, Q in enumerate(Qs):
    for j, A in enumerate(As):
        args = [make_step, GV['dx'], None, Q, A, n, linear, GV['N']]
        data[i, j, :] = solve_ivp(fun=FVM_RHS, y0=h_initial, args=(args,), t_span=t_span, method='BDF',
                                  rtol=1e-6, atol=1e-8, events=[unstable, steady_state]).y[:, -1]


np.save("data/newtonian.npy", data)

