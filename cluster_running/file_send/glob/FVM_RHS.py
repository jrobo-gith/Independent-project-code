import numpy as np

def FVM_RHS(t: float, h: np.ndarray, args: tuple) -> np.ndarray:
    """
    RHS of equation, made for scipy's solve_ivp function that takes care of this stiff fourth order PDE.
    """

    make_step, dx, pwr, Q, _, n, _, N = args

    h = h.copy()
    dhdt = np.zeros_like(h)

    # i = 0
    h[0] = 1

    # i = 1
    q_plus, q_minus = make_step(h=h, i=1, args=args)
    dhdt[1] = - (q_plus - Q) / dx

    # i = N - 2
    q_plus, q_minus = make_step(h=h, i=N - 2, args=args)
    dhdt[N - 2] = - (h[N - 2] - q_minus) / dx

    # i = N - 1
    h[N - 1] = h[N - 2]
    dhdt[N - 1] = dhdt[N - 2]

    for i in range(2, N - 2):
        q_plus, q_minus = make_step(h=h, i=i, args=args)
        dhdt[i] = -(q_plus - q_minus) / dx

    return dhdt

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

    except IndexError:
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

    except IndexError:
        q_minus = 0

    return q_plus, q_minus