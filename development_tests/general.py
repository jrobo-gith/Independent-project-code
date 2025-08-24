import numpy as np
import json
import matplotlib.pyplot as plt
from scipy.interpolate import interp1d
from scipy.integrate import solve_ivp
from non_newtonian_thin_film_solve.individual_files.power_law_steady import solver as bdf_solver_NN
from non_newtonian_thin_film_solve.individual_files.power_law_startup import make_step
from glob_var.FVM.stop_events import unstable, steady_state
from glob_var.FVM.FVM_RHS import FVM_RHS
from paper_figures.validation_graphs.graph_generator import generate_validation_graph as gvg
from paper_figures.validation_graphs.error import temporal_error, spatial_error, interpolate

try:
    with open('../glob_var/global_variables.json') as f:
        GV = json.load(f)
except FileNotFoundError:
    print("Global variables json not found!")

t_span = GV['t-span']
n = 0.8
Q = 0.75

# startup = np.load("test_data/startup.npy")

args = [make_step, GV['dx'], 0, Q, None, n, True, GV['N']]
startup = solve_ivp(fun=FVM_RHS, args=(args,), y0=np.ones(GV['N']), t_span=t_span, method='BDF', rtol=1e-6, atol=1e-8,
                  events=[unstable, steady_state])

steady = bdf_solver_NN(q=Q, L=GV['L'], n=n, linear=True).y[0]

f = interp1d(np.linspace(0, 1, len(startup)), startup)
startup = f(np.linspace(0, 1, len(steady)))

x = np.linspace(0, GV['L'], steady.shape[0])

fig, ax = plt.subplots(ncols=2, figsize=(12, 10))

ax[0].plot(x, startup, label='Startup flow')
ax[0].plot(x, steady, label='Steady state')
ax[0].set_ylim(0.7, 1.0)
ax[0].set_xlim(0, 5)
ax[0].legend()
ax[0].grid(True)

ax[1].plot(np.abs(startup - steady), label='Error')

plt.show()

