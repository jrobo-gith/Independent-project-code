# This is the numerical solution to the standard dimensional thin film equation for a STARTUP-FLOW:
# dh/dt + sigma/3*mu * d/dx(h^3 d^3h/dx^3) - Ux = 0
# This solution uses central differences to solve the fourth order PDE

# Import packages
import matplotlib.pyplot as plt
import numpy as np
import scipy as sp

# Define variables
t_end = 1_000                           # Length of time we want to run for
N = 10_000                              # Number of spatial points
L = 1000                                # Length of solid
U = 0.00001                             # Bed speed (m/s)
h_0 = 1.0                               # Initial height h(x=0) = 1.0
h_L = 0.7                               # Height at end of fluid h(x=L) = 0.2
sigma = 0.001                           # Surface tension
mu = 0.001                              # Dynamic viscosity
a = 10

dx = L/N
dt = 0.5 * dx**2 / a

G = ((sigma*dt)/(6*mu*(dx**2)))

def next_step(matrix, j, i):
    """Compute the next step of the solver (h^{j+1})"""
    matrix[j+1, i] = matrix[j, i] - G * ((matrix[j, i+1]**3) * (matrix[j, i+2]-2*matrix[j, i+1]) - ((matrix[j, i-1]**3)
    * (2*matrix[j, i-1] - matrix[j, i-2])))
    return matrix

def plot_matrix(matrix):
    fig, ax = plt.subplots()
    ax.plot(matrix[-1])
    plt.show()

def main():
    """Main function to run"""
    matrix = np.zeros((t_end, N))
    matrix[0, :] = 0
    matrix[-1, :] = h_L
    matrix[:, 0] = h_0


    for j in range(t_end):
        for i in range(2, N-2):
            matrix = next_step(matrix, j, i)
    print(matrix)

if __name__ == '__main__':
    main()