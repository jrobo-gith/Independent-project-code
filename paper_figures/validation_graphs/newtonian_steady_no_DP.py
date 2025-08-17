import numpy as np
import matplotlib.pyplot as plt
import json
from newtonian_thin_film_solve.individual_files.steady_state_central_differences import solver as bdf_solver_N

try:
    with open('../../glob_var/global_variables.json') as f:
        GV = json.load(f)
except FileNotFoundError:
    print("Global variables json not found!")

linear_solution_Q_steady = []      # Store linear solutions varying Q
linear_solution_steady = []      # Store linear solutions varying L
non_linear_solution_Q_steady = []  # Store non-linear solutions varying Q
non_linear_solution_steady = []  # Store non-linear solutions varying L

L_list = [5, 20, 40, 80]

for i in range(len(GV['Q-list'])):
    linear_solution_Q_steady.append(bdf_solver_N(q=GV['Q-list'][i], L=GV['L'], linear=True).y[0])
    non_linear_solution_Q_steady.append(bdf_solver_N(q=GV['Q-list'][i], L=GV['L'], linear=False).y[0])
for L in L_list:
    linear_solution_steady.append(bdf_solver_N(q=GV['Q'], L=L, linear=True).y[0])
    non_linear_solution_steady.append(bdf_solver_N(q=GV['Q'], L=L, linear=False).y[0])


fig, ax = plt.subplots(nrows=2, figsize=(10, 15))

ax[1].set_xlim(0, 5)

large_left = ax[0]
top_left = top_right = bottom_left = bottom_right = ax[1]

[large_left.plot(np.linspace(0, GV['L'], len(linear_solution_Q_steady[i]))
, linear_solution_Q_steady[i], color=GV['colors'][i], linestyle='--', linewidth=2, label=f"$Q={GV['Q-list'][i]}$") for i in range(len(GV['Q-list']))]
[large_left.plot(np.linspace(0, GV['L'], len(non_linear_solution_Q_steady[i])), non_linear_solution_Q_steady[i], color=GV['colors'][i], linestyle='-', linewidth=2) for i in range(len(GV['Q-list']))]
large_left.legend(loc='lower right')
large_left.grid(True)
large_left.set_title(f"Complete BVP varying flux Q at surface length $L={GV['L']}$", fontsize=16)
large_left.set_xlabel("Surface Length $(x)$", fontsize=14)
large_left.set_ylabel("Film Height $(y)$", fontsize=14)
large_left.set_ylim(0.7, 1)

top_left.set_title(f"Varying length L, confirming no boundary influence on meniscus", fontsize=16)
top_left.plot(np.linspace(0, L_list[0], len(linear_solution_steady[0])), linear_solution_steady[0], linestyle='--', color='k')
top_left.plot(np.linspace(0, L_list[0], len(non_linear_solution_steady[0])), non_linear_solution_steady[0], linestyle='-', color='k', label=f"L = {L_list[0]}")
top_left.grid(True)
top_left.set_ylim(0, 1)
top_left.set_ylabel("Film Height $(y)$", fontsize=14)

top_right.plot(np.linspace(0, L_list[1], len(linear_solution_steady[1])), linear_solution_steady[1], linestyle='--', color='red')
top_right.plot(np.linspace(0, L_list[1], len(non_linear_solution_steady[1])), non_linear_solution_steady[1], linestyle='-', color='red', label=f"L = {L_list[1]}")
top_right.grid(True)
top_right.set_ylim(0, 1)

bottom_left.plot(np.linspace(0, L_list[2], len(linear_solution_steady[2])), linear_solution_steady[2], linestyle='--', color='coral')
bottom_left.plot(np.linspace(0, L_list[2], len(non_linear_solution_steady[2])), non_linear_solution_steady[2], linestyle='-', color='coral', label=f"L = {L_list[2]}")
bottom_left.grid(True)
bottom_left.set_ylim(0, 1)
bottom_left.set_ylabel("Film Height $(y)$", fontsize=14)
bottom_left.set_xlabel("Surface Length $(x)$", fontsize=14)

bottom_right.plot(np.linspace(0, L_list[3], len(linear_solution_steady[3])), linear_solution_steady[3], linestyle='--', color='blue')
bottom_right.plot(np.linspace(0, L_list[3], len(non_linear_solution_steady[3])), non_linear_solution_steady[3], linestyle='-', color='blue', label=f"L = {L_list[3]}")
bottom_right.grid(True)
bottom_right.set_ylim(0, 1)
bottom_right.set_xlabel("Surface Length $(x)$", fontsize=14)

ax[1].legend()

fig.suptitle("Validation graph varying flux Q with fixed L (left) and length scale L with fixed Q (right)", fontsize=20, y=1.055)
plt.tight_layout()

fig.savefig("graphs/newtonian_steady_no_DP.png")