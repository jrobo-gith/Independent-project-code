import numpy as np

def make_step(h, i, args):
    """
    (Try, except)'s are for the BCs, as q_minus won't be computable for BCs at the start and q_plus won't be computable
    for BCs at the end.
    The bool 'print_except' is for making sure anything inside the for loop is not displaying a 0.
    """

    _, dx, pwr, Q, A, n, linear, _ = args

    DX = 1/dx**3

    try:
        disjoining_pressure_term_plus = 3 * A * (((h[i+1]+h[i])/2)**-4) * ((h[i+1]-h[i])/dx)
        if linear:
            non_linear_h_plus = 1
        else:
            non_linear_h_plus = (0.5 * (h[i] + h[i+1]))**((2*n + 1)/n)
        third_order_plus = abs(DX * (-h[i-1] + 3*h[i] - 3*h[i+1] + h[i+2]) + disjoining_pressure_term_plus) ** (1 / n)
        third_order_sign = np.sign(DX * (-h[i-1] + 3*h[i] - 3*h[i+1] + h[i+2]) + disjoining_pressure_term_plus)
        advection_term_plus = h[i]

        q_plus = non_linear_h_plus * third_order_sign * third_order_plus + advection_term_plus

    except IndexError:
        q_plus = non_linear_h_plus =  third_order_plus = advection_term_plus = 0

    try:
        disjoining_pressure_term_minus = 3 * A * (((h[i] + h[i-1])/2)**-4) * ((h[i] - h[i-1]) / dx)
        if linear:
            non_linear_h_minus = 1
        else:
            non_linear_h_minus = (0.5 * (h[i] + h[i-1])) ** ((2*n+1)/n)
        third_order_minus = abs(DX * (-h[i-2] + 3*h[i-1] - 3*h[i] + h[i+1]) + disjoining_pressure_term_minus) ** (1 / n)
        third_order_sign = np.sign(DX * (-h[i-2] + 3*h[i-1] - 3*h[i] + h[i+1]) + disjoining_pressure_term_minus)
        advection_term_minus = h[i-1]

        q_minus = non_linear_h_minus * third_order_sign * third_order_minus + advection_term_minus

    except IndexError:
        q_minus = non_linear_h_minus = third_order_minus = advection_term_minus = 0

    return q_plus, q_minus, [advection_term_minus, advection_term_plus], [non_linear_h_minus, non_linear_h_plus], [third_order_minus, third_order_plus], [disjoining_pressure_term_minus**(1/n), disjoining_pressure_term_plus**(1/n)]
