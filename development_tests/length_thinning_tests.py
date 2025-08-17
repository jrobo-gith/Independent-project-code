from glob_var.FVM.FVM_RHS import FVM_RHS
from non_newtonian_thin_film_solve.individual_files.power_law_startup import make_step
import numpy as np
import json
from scipy.integrate import solve_ivp
from scipy.interpolate import interp1d
import matplotlib.pyplot as plt
from non_newtonian_thin_film_solve.individual_files.power_law_steady import solver as bdf_solver

try:
    with open('../glob_var/global_variables.json') as f:
        GV = json.load(f)
except FileNotFoundError:
    print("Global variables json not found!")

def unstable(t, h, args):
    """Returns minimum of the array h, if the value is 0, the solver terminates."""
    return min(h)

L_list = [5, 20, 40, 80]

linear_solution = []
non_linear_solution = []

t_span = GV['t-span']
n = 1.0

fig, ax = plt.subplots()
ax.set_xlim(0, 5)
ax.set_ylim(0.7, 1.0)

for L in L_list:
    print(f"L = {L}")
    SS = bdf_solver(q=0.75, L=L, n=n, linear=False).y[0]

    def steady_state(t, h, args):
        """Triggers an event when the time derivative is nearly 0, meaning the system has reached a near-steady state"""
        dhdt = FVM_RHS(t, h, args)
        f = interp1d(np.linspace(0, 1, len(SS)), SS)
        steady = f(np.linspace(0, 1, len(dhdt)))
        return np.mean(dhdt) - 1e-3

    h_initial = np.ones(GV['N']) * GV['h0']
    dx = L / GV['N']

    args = [make_step, dx, 0, GV['Q'], 0.0, n, False, GV['N']]
    sol = solve_ivp(fun=FVM_RHS, args=(args,), y0=h_initial, t_span=t_span, method='BDF', rtol=1e-6, atol=1e-8,
                  events=[unstable, steady_state])
    linear_solution.append(sol.y[:, -1])
    print("Solved linear case")

    ax.plot(sol.y[:, -1])
    fig.show()

    args = [make_step, dx, 3, GV['Q'], 0.0, n, False, GV['N']]
    sol = solve_ivp(fun=FVM_RHS, args=(args,), y0=h_initial, t_span=t_span, method='BDF', rtol=1e-6, atol=1e-8,
                  events=[unstable, steady_state])
    non_linear_solution.append(sol.y[:, -1])
    print("Solved non-linear case\n")
    ax.plot(sol.y[:, -1])
    fig.show()
