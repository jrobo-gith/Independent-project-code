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
        disjoining_pressure_term = 3 * A * (((h[i+1]+h[i])/2)**-4) * ((h[i+1]-h[i])/dx)
        if linear:
            non_linear_h = 1
        else:
            non_linear_h = (0.5 * (h[i] + h[i+1]))**((2*n + 1)/n)
        third_order = abs(DX * (-h[i-1] + 3*h[i] - 3*h[i+1] + h[i+2]) + disjoining_pressure_term) ** (1 / n)
        third_order_sign = np.sign(DX * (-h[i-1] + 3*h[i] - 3*h[i+1] + h[i+2]) + disjoining_pressure_term)
        advection_term = h[i]

        q_plus = non_linear_h * third_order_sign * third_order + advection_term

        tracking_1_plus = third_order_sign
        tracking_2_plus = non_linear_h * third_order * third_order_sign
        tracking_3_plus = (abs(DX * (-h[i-1] + 3*h[i] - 3*h[i+1] + h[i+2])) / (abs(DX * (-h[i-1] + 3*h[i] - 3*h[i+1] + h[i+2])) + abs(disjoining_pressure_term)))

    except IndexError:
        tracking_1_plus = 0
        tracking_2_plus = 0
        tracking_3_plus = 0
        q_plus = 0

    try:
        disjoining_pressure_term = 3 * A * (((h[i] + h[i-1])/2)**-4) * ((h[i] - h[i-1]) / dx)
        if linear:
            non_linear_h = 1
        else:
            non_linear_h = (0.5 * (h[i] + h[i-1])) ** ((2*n+1)/n)
        third_order = abs(DX * (-h[i-2] + 3*h[i-1] - 3*h[i] + h[i+1]) + disjoining_pressure_term) ** (1 / n)
        third_order_sign = np.sign(DX * (-h[i-2] + 3*h[i-1] - 3*h[i] + h[i+1]) + disjoining_pressure_term)
        advection_term = h[i-1]

        q_minus = non_linear_h * third_order_sign * third_order + advection_term

        tracking_1_minus = third_order_sign
        tracking_2_minus = non_linear_h * third_order * third_order_sign
        tracking_3_minus = (abs(DX * (-h[i-2] + 3*h[i-1] - 3*h[i] + h[i+1])) / (
                    abs(DX * (-h[i-2] + 3*h[i-1] - 3*h[i] + h[i+1])) + abs(disjoining_pressure_term)))

    except IndexError:
        tracking_1_minus = 0
        tracking_2_minus = 0
        tracking_3_minus = 0
        q_minus = 0

    return q_plus, q_minus, [tracking_1_minus, tracking_1_plus], [tracking_2_minus, tracking_2_plus], [tracking_3_minus, tracking_3_plus]
