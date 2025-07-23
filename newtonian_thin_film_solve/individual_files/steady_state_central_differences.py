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

N = GV['N']
L = GV['L']
Q = GV['Q']
h_0 = GV['h0']

# Define Functions
def ODE(x, y):
    """
    Split ODE into three first order ODEs, returns the derivative of the vector y linking the ODEs.
    """
    dy_3 = (Q-y[0])/(y[0]**3)

    return np.array([y[1], y[2], dy_3])

def bc(x_zero, x_L):
    """
    Boundary conditions for the BVP to show the height at each boundary and flux at x=0.
    """
    return np.array([x_zero[0]-1, x_L[1], x_L[0]-Q])

def solver():
    """
    Uses scipy integrate to solve the boundary value problem
    """
    x = np.linspace(0, L, N)
    y = np.zeros((3, x.size))
    y[0] = h_0
    solution = solve_bvp(ODE, bc, x, y)
    print(solution.y[0, :].shape)
    return solution

def plot_solution(solution):
    """
    Plots the solution to the numerically solved thin film equation
    """
    fig, axes = plt.subplots(nrows=1, ncols=1, figsize=(8, 8))
    axes.plot(solution.x, solution.y[0, :])
    fig.show()

# Run everything in main
def main():
    """
    Main function encapsulating all running code
    """
    solution = solver()
    plot_solution(solution)

if __name__ == "__main__":
    # Run main
    main()

