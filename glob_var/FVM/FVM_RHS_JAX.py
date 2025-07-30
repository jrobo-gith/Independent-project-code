import os
import json
import jax.numpy as jnp
import jax
import equinox as eqx
import diffrax


current_dir = os.path.dirname(__file__)
project_root = os.path.abspath(os.path.join(current_dir, '..', '..'))

try:
    with open(project_root + '/glob_var/global_variables.json') as f:
        GV = json.load(f)
except FileNotFoundError:
    print("Global variables json not found!")


class ThinFilm(eqx.Module):
    def __call__(self, t: float, h: jnp.ndarray, args: tuple) -> jnp.ndarray:
        """
        RHS of equation, made for scipy's solve_ivp function that takes care of this stiff fourth order PDE.
        """

        make_step, dx, pwr, Q, _, n, _, N = args

        h = h.copy()
        dhdt = jnp.zeros_like(h)

        # i = 0
        h = h.at[0].set(GV['h0'])

        # i = 1
        q_plus, q_minus = make_step(h=h, i=1, args=args)
        dhdt = dhdt.at[1].set(- (q_plus - Q) / dx)

        # i = N - 2
        q_plus, q_minus = make_step(h=h, i=N - 2, args=args)
        dhdt = dhdt.at[N-2].set(- (h[N - 2] - q_minus) / dx)

        # i = N - 1
        h = h.at[N-1].set(h[N-2])
        dhdt = dhdt.at[N-1].set(dhdt[N - 2])

        for i in range(2, N - 2):
            q_plus, q_minus = make_step(h=h, i=i, args=args)
            dhdt = dhdt.at[i].set(-(q_plus - q_minus) / dx)
        return dhdt