# Uses central differences to compute the startup flow of the thin film equation with an advection term using a forward
# in time, central in space scheme.

import numpy as np
import matplotlib.pyplot as plt

def make_step(matrix, j, i, args):
    dx, var, U = args

    main_term = (matrix[j, i]**3 * (matrix[j, i+2] - 2*matrix[j, i+1] + 2*matrix[j, i-1] - matrix[j, i-2]))/(2* dx ** 3)

    advection_term = U * ((matrix[j, i+1] - matrix[j, i-1])/(2*dx))

    matrix[j+1, i] = matrix[j, i] - var *  main_term - advection_term


    return matrix

def main():
    space = 20          # Length of solid surface (m)
    N = 1000            # Number of nodes in space
    time = 100          # Duration of flow (s)
    h_0 = 1.0           # Height of extruder from solid surface (µm)
    sigma = 0.001       # Surface tension
    mu = 0.001          # Viscosity
    dx = 10             # Change in spatial dim
    dt = 0.1            # Change in temporal dim
    U = 1.5             # Bed speed (m/s)

    var = (sigma * dt)/(3*mu)
    args = [dx, var, U]

    matrix = np.zeros((time, N))

    # Experimental
    Q = 1
    h_minus1 = (((Q-h_0)*dx)/(h_0**3)) - 0.9 + 2 * h_0
        # Establish BCs
    matrix[:, 0] = h_minus1
    matrix[:, 1] = h_0
    matrix[:, 2] = 0.9

    for j in range(time-1):
        for i in range(2, N-2):
            matrix = make_step(matrix, j, i, args)
    return matrix

if __name__ == '__main__':
    mat = main()
    plt.plot(mat[0, :], label='Initial')
    # plt.plot(mat[25, :], label='25')
    # plt.plot(mat[50, :], label='50')
    # plt.plot(mat[75, :], label='75')
    plt.plot(mat[-1, :], label='End')
    plt.xlim([0, 20])
    plt.legend()
    plt.show()


# NOTES
# Equation, when made non-dimensional, leaves no parameters so all we need to vary is Q.