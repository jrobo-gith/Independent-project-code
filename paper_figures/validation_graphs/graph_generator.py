##### Used [gridspec](https://matplotlib.org/stable/api/_as_gen/matplotlib.gridspec.GridSpec.html) when making these types of plots

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import json

try:
    with open('../../glob_var/global_variables.json') as f:
        GV = json.load(f)
except FileNotFoundError:
    print("Global variables json not found!")


def generate_validation_graph(L_arrays:list, Q_arrays:list, space_error_y_arrays:list,
                              time_error_x_arrays:list, time_error_y_arrays:list, steady_states_array:list,
                              directory:str, rheology:str):

    L_list = [5, 20, 40, 80]
    linear_SS, non_linear_SS = steady_states_array
    linear_solution, non_linear_solution = L_arrays    
    linear_solution_Q, non_linear_solution_Q = Q_arrays
    time_linear_error_x, time_non_linear_error_x = time_error_x_arrays
    time_linear_error_y, time_non_linear_error_y = time_error_y_arrays
    space_linear_error_y, space_non_linear_error_y = space_error_y_arrays
    
    fig = plt.figure(figsize=(10, 10))

    gs = gridspec.GridSpec(3, 2, figure=fig, height_ratios=[1.1, 1.1, 0.8])
    large_top = fig.add_subplot(gs[0, :])
    time_error = fig.add_subplot(gs[1, 0])
    space_error = fig.add_subplot(gs[1, 1])

    length_graph = fig.add_subplot(gs[2, :])

    [large_top.plot(GV['x'], linear_solution_Q[i], color=GV['colors'][i], linestyle='--') for i in
     range(len(GV['Q-list']))]
    [large_top.plot(GV['x'], non_linear_solution_Q[i], color=GV['colors'][i], label=f"$Q={GV['Q-list'][i]}$",
                     linestyle='-') for i in range(len(GV['Q-list']))]
    [large_top.scatter(GV['x'], linear_SS[i], color=GV['colors'][i], marker='x', s=2) for i in range(len(GV['Q-list']))]
    [large_top.scatter(GV['x'], non_linear_SS[i], color=GV['colors'][i], marker='x', s=2) for i in
     range(len(GV['Q-list']))]
    large_top.legend(loc='lower right')
    large_top.grid(True)
    large_top.set_title(r"Complete BVP varying flux $\tilde{h}$" + f"at surface length $L={GV['L']}$", fontsize=16)
    large_top.set_xlabel(r"Surface Length $(\tilde{x})$", fontsize=14)
    large_top.set_ylabel(r"Film Height $(\tilde{y})$", fontsize=14)
    large_top.set_ylim(0.7, 1)

    [length_graph.plot(np.linspace(0, L_list[i], GV['N']), linear_solution[i], linestyle='--', color=GV['colors'][i]) for i in range(len(L_list))]
    [length_graph.plot(np.linspace(0, L_list[i], GV['N']), non_linear_solution[i], linestyle='-', label=f"L={L}", color=GV['colors'][i]) for i, L in
     enumerate(L_list)]
    length_graph.legend(loc='upper right')
    length_graph.grid(True)
    length_graph.set_xlim(0, 5)
    length_graph.set_ylim(0.7, 1.0)
    length_graph.set_title(r"Varying $\tilde{L}$, visualising downward meniscus", fontsize=16)
    
    [time_error.plot(time_linear_error_x[i], time_linear_error_y[i], linestyle='--', color=GV['colors'][i]) for i in range(len(GV['Q-list']))]
    [time_error.plot(time_non_linear_error_x[i], time_non_linear_error_y[i], linestyle='-', label=f"$Q={GV['Q-list'][i]}$", color=GV['colors'][i]) for i in
     range(len(GV['Q-list']))]
    time_error.grid(True)
    time_error.set_title("% Deviation from steady state\nover time", fontsize=16)
    time_error.set_ylabel("Error (%)", fontsize=14)
    time_error.set_xlabel(r"Time $(\tilde{t})$", fontsize=14)

    space_x = np.linspace(0, 16, len(space_linear_error_y[0]))
    [space_error.plot(space_x, space_linear_error_y[i], linestyle='--', color=GV['colors'][i]) for i in
     range(len(GV['Q-list']))]
    [space_error.plot(space_x, space_non_linear_error_y[i], linestyle='-',
                     label=f"$Q={GV['Q-list'][i]}$", color=GV['colors'][i]) for i in
     range(len(GV['Q-list']))]
    space_error.grid(True)
    space_error.set_title("% Deviation from steady state at\nfinal timestep", fontsize=16)
    space_error.set_xlabel(r"Surface Length $(\tilde{x})$", fontsize=14)
    space_error.set_ylabel("Error (%)", fontsize=14)

    fig.suptitle(
        r"Validation graph varying flux $\tilde{Q}$ with fixed $\tilde{L}$ (Top) the error"+"\nover time (middle left) and against the steady state model (middle right)\nand length scale"+r" $\tilde{L}$ with fixed $\tilde{Q}$ (Bottom)",
        fontsize=16, y=1.)
    plt.tight_layout()
    fig.savefig(directory, bbox_inches='tight')