import numpy as np
import json
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp
from glob_var.animation import Animation

try:
    with open('../glob_var/global_variables.json') as f:
        GV = json.load(f)
except FileNotFoundError:
    print("Global variables json not found!")

def unstable(t, h, args):
    """Returns minimum of the array h, if the value is 0, the solver terminates."""
    return min(h)

def steady_state(t, h, args):
    """Triggers an event when the time derivative is nearly 0, meaning the system has reached a near-steady state"""
    dhdt = FVM_RHS(t, h, args)
    return np.linalg.norm(dhdt) - 1e-4

advection_terms = []
times = []

def FVM_RHS(t: float, h: np.ndarray, args: tuple) -> np.ndarray:
    """
    RHS of equation, made for scipy's solve_ivp function that takes care of this stiff fourth order PDE.
    """

    times.append(t)

    make_step, dx, pwr, Q, _, n, _, N = args

    local_advection_term = np.zeros((2, N-2))

    h = h.copy()
    dhdt = np.zeros_like(h)

    # i = 0
    h[0] = GV['h0']

    # i = 1
    q_plus, q_minus, adv = make_step(h=h, i=1, args=args)
    dhdt[1] = - (q_plus - Q) / dx
    local_advection_term[:, 0] = adv

    # i = N - 2
    q_plus, q_minus, adv = make_step(h=h, i=N - 2, args=args)
    dhdt[N - 2] = - (h[N - 2] - q_minus) / dx
    local_advection_term[:, 1] = adv

    # i = N - 1
    h[N - 1] = h[N - 2]
    dhdt[N - 1] = dhdt[N - 2]

    for i in range(2, N - 2):
        q_plus, q_minus, adv = make_step(h=h, i=i, args=args)
        dhdt[i] = -(q_plus - q_minus) / dx
        local_advection_term[:, i] = adv

    advection_terms.append(local_advection_term)

    return dhdt

def make_step(h, i, args):
    """
    (Try, except)'s are for the BCs, as q_minus won't be computable for BCs at the start and q_plus won't be computable
    for BCs at the end.
    The bool 'print_except' is for making sure anything inside the for loop is not displaying a 0.
    """

    _, dx, pwr, Q, A, n, linear, _ = args

    DX = 1/dx**3

    try:
        disjoining_pressure_term = 3 * A * (((h[i+1]+h[i])/2)**-4) * ((h[i+1]-h[i])/dx)
        if linear:
            non_linear_h = 1
        else:
            non_linear_h = (0.5 * (h[i] + h[i+1]))**((2*n + 1)/n)
        third_order = abs(DX * (-h[i-1] + 3*h[i] - 3*h[i+1] + h[i+2]) + disjoining_pressure_term) ** (1 / n)
        third_order_sign = np.sign(DX * (-h[i-1] + 3*h[i] - 3*h[i+1] + h[i+2]) + disjoining_pressure_term)
        advection_term_plus = h[i]

        q_plus = non_linear_h * third_order_sign * third_order + advection_term_plus

    except IndexError:
        q_plus = 0
        advection_term_plus = 0

    try:
        disjoining_pressure_term = 3 * A * (((h[i] + h[i-1])/2)**-4) * ((h[i] - h[i-1]) / dx)
        if linear:
            non_linear_h = 1
        else:
            non_linear_h = (0.5 * (h[i] + h[i-1])) ** ((2*n+1)/n)
        third_order = abs(DX * (-h[i-2] + 3*h[i-1] - 3*h[i] + h[i+1]) + disjoining_pressure_term) ** (1 / n)
        third_order_sign = np.sign(DX * (-h[i-2] + 3*h[i-1] - 3*h[i] + h[i+1]) + disjoining_pressure_term)
        advection_term_minus = h[i-1]

        q_minus = non_linear_h * third_order_sign * third_order + advection_term_minus

    except IndexError:
        q_minus = 0
        advection_term_minus = 0

    return q_plus, q_minus, [advection_term_minus, advection_term_plus]


unstable.terminal = True
steady_state.terminal = True

h_initial = np.ones(GV['N']) * GV['h0']
t_span = GV['t-span']

A = 0.0

print(f"Newtonian | A = {A} ")
n = 1.0
args = [make_step, GV['dx'], None, GV['Q'], A, n, False, GV['N']]
newt_sol = solve_ivp(fun=FVM_RHS, y0=h_initial, t_span=t_span, args=(args,), method='BDF', rtol=1e-6, atol=1e-8, events=[unstable, steady_state])

adv_term = np.array(advection_terms)
adv_term = np.transpose(adv_term, (1, 2, 0))

times = np.array(times)

# Now outline some figure details to add to the plot
fig_details = {
    'x-lim': (0, times[-1]),
    'y-lim': (-1, 1),
    'legend': [False,],
    'grid': [True],
    'title': [f"Startup Flow of adv term", True],
    'x-label': ['Surface Length $(x)$'],
    'y-label': ['Advection Strength'],
}
# Instantiate the Animation class
newt_animation = Animation(num_rows=1, num_cols=1,
                           fig_size=(10, 8), x=np.linspace(times[0], times[-1], adv_term.shape[1]),
                           data=[adv_term,], min_timestep=1700,
                           fig_details=fig_details, interval=20, title_updates=times)
newt_animation.instantiate_animation()
newt_animation.save_animation('test')
