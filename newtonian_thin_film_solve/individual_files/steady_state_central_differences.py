# This is the numerical solution to the standard dimensional thin film equation for a STEADY STATE FLOW (dh/dt=0):
# dh/dt + sigma/3*mu * d/dx(h^3 d^3h/dx^3) -Ux = 0
# This solution uses central differences to solve the fourth order PDE

# Import packages
import matplotlib.pyplot as plt
import numpy as np
from scipy.integrate import solve_bvp
import json
import os

current_dir = os.path.dirname(__file__)
project_root = os.path.abspath(os.path.join(current_dir, '..', '..'))

try:
    with open(project_root + '/glob_var/global_variables.json') as f:
        GV = json.load(f)
except FileNotFoundError:
    print("Global variables json not found!")

# Define Functions
def ODE(x, y, pwr, Q):
    """
    Split ODE into three first order ODEs, returns the derivative of the vector y linking the ODEs. Variable 'a' represents the non-linearity in the model, when set to 0, there is no non-linearity.
    """
    dy_3 = (Q-y[0])/(y[0]**pwr)

    return np.array([y[1], y[2], dy_3])

def bc(x_zero, x_L, Q):
    """
    Boundary conditions for the BVP to show the height at each boundary and flux at x=0.
    """
    return np.array([x_zero[0]-1, x_L[1], Q-x_L[0]])

def solver(q:float, L:int, linear:bool):
    """
    Uses scipy integrate to solve the boundary value problem
    """
    if linear:
        pwr = 0
    else:
        pwr = 3
    x = np.linspace(0, L, 1_000_00)
    y = np.zeros((3, x.size))
    y[0] = GV['h0']
    solution = solve_bvp(lambda x,y: ODE(x, y, pwr=pwr, Q=q), lambda x,y: bc(x, y, Q=q), x, y, max_nodes=4000000000, tol=1e-2)
    return solution

def plot_solution(solution, q, axes=None):
    """
    Plots the solution to the numerically solved thin film equation, adds a diagram of the extruder also
    """
    if axes is None:
        fig, axes = plt.subplots(nrows=1, ncols=1, figsize=(10, 5))
    axes.plot(solution.x, solution.y[0], color='g', linestyle='-', linewidth=1, marker='.', )
    axes.set_title(f"Solution to BVP (Q={q}) \n $h(x=0)={GV['h0']}$")
    axes.grid(True)
    axes.set_xlabel('Surface Length $(x)$')
    axes.set_ylabel('Film Height $(y)$')
    if __name__ == "__main__":
        fig.show()

# Run everything in main
def main():
    """
    Main function encapsulating all running code
    """
    solution = solver(0.75, GV['L'], linear=False)
    plot_solution(solution, GV['Q'])

if __name__ == "__main__":
    # Run main
    main()

