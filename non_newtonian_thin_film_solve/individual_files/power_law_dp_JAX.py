import matplotlib.pyplot as plt
import os
import json
import jax.numpy as jnp
import jax
import diffrax

from glob_var.FVM.FVM_RHS_JAX import ThinFilm
from glob_var.FVM.stop_events import unstable, steady_state

current_dir = os.path.dirname(__file__)
project_root = os.path.abspath(os.path.join(current_dir, '..', '..'))

try:
    with open(project_root + '/glob_var/global_variables.json') as f:
        GV = json.load(f)
except FileNotFoundError:
    print("Global variables json not found!")


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
        third_order_sign = jnp.sign(DX * (-h[i-1] + 3*h[i] - 3*h[i+1] + h[i+2]) + disjoining_pressure_term)
        advection_term = h[i]

        q_plus = non_linear_h * third_order_sign * third_order + advection_term

    except IndexError:
        q_plus = 0

    try:
        disjoining_pressure_term = 3 * A * (((h[i] + h[i-1])/2)**-4) * ((h[i] - h[i-1]) / dx)
        if linear:
            non_linear_h = 1
        else:
            non_linear_h = (0.5 * (h[i] + h[i-1])) ** ((2*n+1)/n)
        third_order = abs(DX * (-h[i-2] + 3*h[i-1] - 3*h[i] + h[i+1]) + disjoining_pressure_term) ** (1 / n)
        third_order_sign = jnp.sign(DX * (-h[i-2] + 3*h[i-1] - 3*h[i] + h[i+1]) + disjoining_pressure_term)
        advection_term = h[i-1]

        q_minus = non_linear_h * third_order_sign * third_order + advection_term

    except IndexError:
        q_minus = 0

    return q_plus, q_minus

@jax.jit
def main():
    A = 0.0
    n = 1.0
    args = (make_step, GV['dx'], None, GV['Q'], A, n, False, GV['N'])
    thin_film = ThinFilm()
    term = diffrax.ODETerm(thin_film)
    t0 = 0.0
    t1 = 1_00
    y0 = jnp.ones(GV['N']) * GV['h0']
    dt0 = 0.0002
    solver = diffrax.Kvaerno5()
    # saveat = diffrax.SaveAt(ts=jnp.array([0.0, 1e-4, 1e-3, 1e-2, 1e-1, 1e0, 1e1, 1e2]))
    stepsize_controller = diffrax.PIDController(rtol=1e-8, atol=1e-8)
    event = diffrax.Event(unstable, steady_state)
    sol = diffrax.diffeqsolve(term, solver, t0, t1, dt0, y0, stepsize_controller=stepsize_controller,
                              args=args)
    return sol

s = main()
print(s)