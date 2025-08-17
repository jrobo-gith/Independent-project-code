from scipy.integrate import solve_ivp
import numpy as np
import json
from glob_var.animation import Animation
from glob_var.FVM.FVM_RHS import FVM_RHS
from glob_var.FVM.stop_events import unstable, steady_state
from newtonian_thin_film_solve.individual_files.startup_flow_FVM import make_step as newt_make_step

try:
    with open('../../glob_var/global_variables.json') as f:
        GV = json.load(f)
except FileNotFoundError:
    print("Global variables json not found!")


args = [newt_make_step, GV['dx'], 3, GV['Q'], None, None, False, GV['N']]
h_initial = np.ones(GV['N']) * 0.01
h_initial[0] = GV['h0']
t_span = GV['t-span']
sol = solve_ivp(fun=FVM_RHS, args=(args,), y0=h_initial, t_span=t_span, method='BDF', rtol=1e-6, atol=1e-8, events=[unstable, steady_state])

# Because the data we just generated is only a 2d array and our animation class takes 3D matrices, we need to add an extra dimension to sol.
data = sol.y
data = [data[None, :, :],]
num_frames = sol.y.shape[1]

# Now outline some figure details to add to the plot
fig_details = {
    'x-lim': (0, GV['L']),
    'y-lim': (0, GV['h0']),
    'legend': [False,],
    'grid': [True],
    'title': [f"Startup Flow of Newtonian Fluid, Q={GV['Q']}",],
    'x-label': ['Surface Length $(x)$'],
    'y-label': ['Film Height $(y)$'],
}

# Instantiate the Animation class
newt_animation = Animation(num_rows=1, num_cols=1,
                           fig_size=(10, 8), x=GV['x'],
                           data=data, min_timestep=num_frames,
                           fig_details=fig_details, interval=20)

newt_animation.instantiate_animation()
newt_animation.save_animation('../gifs/newt_lower_initial_h')