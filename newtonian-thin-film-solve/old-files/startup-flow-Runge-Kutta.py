# dh/dt = sigma/3mu d/dx(h^3 d^3h/dx) - Udh/dx

# Import packages
import numpy as np
from scipy.integrate import solve_ivp
import matplotlib.pyplot as plt

# Define variables
U = 1                   # Bed speed (m/s)
sigma = 0.001           # Surface tension
mu = 0.001              # Viscosity
h_0 = 1.0               # Initial height h(x=0) = 1.0
h_L = 0.4               # Height at end of fluid h(x=L) = 0.2
L = 1_000               # Max length of x
N = 1_000              # Nodes
t_end = 1_00

a = 10
dx = L/N
dt = 0.5 * dx**2 / a
print(f"dt = {dt}")

def next_step_position(matrix, G, i):
    """Compute the next step of the solver (h^{j+1})"""
    new_matrix = matrix[i] + G * ((matrix[i+1]**3) * (matrix[i+2]-2*matrix[i+1]) - ((matrix[i-1]**3)
    * (2*matrix[i-1] - matrix[i-2])) - U*(matrix[i] - matrix[i-1])/dx)
    return new_matrix

# Define functions
def main():
    time = np.arange(0, t_end, dt)
    master_matrix = []

    counter = 0
    matrix = np.zeros(N)
    matrix[0] = h_0
    matrix[1] = h_0
    matrix[-1] = h_L

    while counter < t_end:
        G = ((sigma * dt) / (6 * mu * (dx ** 2)))
        M = matrix.copy()
        for i in range(1, N-2):
            M[i+1] = next_step_position(M, G, i)
        counter += dt
        master_matrix.append(M)

    master_matrix = np.array(master_matrix)
    plt.plot(master_matrix[0, :])
    plt.plot(master_matrix[56, :])
    plt.show()

if __name__ == '__main__':
    main()