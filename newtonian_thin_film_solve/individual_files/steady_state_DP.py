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
def ODE(x, y, Q, pwr, A):
    """
    Split ODE into three first order ODEs, returns the derivative of the vector y linking the ODEs. 'A' represents the
    strength of the disjoint pressure.
    """
    DP = 3 * A * (1/y[0]) * y[1]
    dy_3 = (Q - y[0] - DP) / (y[0] ** pwr)

    return np.array([y[1], y[2], dy_3])

def bc(x_zero, x_L, Q):
    """
    Boundary conditions for the BVP to show the height at each boundary and flux at x=0.
    """
    return np.array([x_zero[0]-1, x_L[1], x_L[0]-Q])

def solver(q:float, L:int):
    """
    Uses scipy integrate to solve the boundary value problem
    """

    x = np.linspace(0, L, GV['N'])
    y = np.zeros((3, x.size))
    y[0] = GV['h0']
    solution = solve_bvp(lambda x,y: ODE(x, y, Q=q, pwr=3, A=0.16), lambda x,y: bc(x, y, Q=q), x, y)
    return solution

def plot_solution(solution, q, axes=None):
    """
    Plots the solution to the numerically solved thin film equation, adds a diagram of the extruder also
    """
    if axes is None:
        fig, axes = plt.subplots(nrows=1, ncols=1, figsize=(10, 5))
    axes.plot(solution.x, solution.y[0], color='g', linestyle='-', linewidth=2)
    axes.set_title(f"Solution to BVP (Q={q}) \n $h(x=0)={GV['h0']}$")
    axes.grid(True)
    axes.set_xlabel('Surface Length $(x)$')
    axes.set_ylabel('Film Height $(y)$')
    fig.show()

# Run everything in main
def main():
    """
    Main function encapsulating all running code
    """
    solution = solver(GV['Q'], GV['L'])
    plot_solution(solution, GV['Q'])

if __name__ == "__main__":
    # Run main
    main()