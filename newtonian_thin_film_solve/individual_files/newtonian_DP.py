import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp

import os
import json

current_dir = os.path.dirname(__file__)
project_root = os.path.abspath(os.path.join(current_dir, '..', '..'))

try:
    with open(project_root + '/glob_var/global_variables.json') as f:
        GV = json.load(f)
except FileNotFoundError:
    print("Global variables json not found!")


def make_step(h, i, args):
    """
    Try, excepts are for the BCs, as q_minus won't be computable for BCs at the start and q_plus won't be computable
    for BCs at the end.
    """

    _, dx, pwr, Q, A, n = args

    Dx = 1 / dx ** 3
    try:
        disjoint_pressure_term = -A / (6 * np.pi * h[i] ** 3)
        non_linear_term = ((h[i] + h[i + 1]) / 2) ** pwr
        q_plus = Dx * non_linear_term * (-h[i - 1] + 3 * h[i] - 3 * h[i + 1] + h[i + 2] - disjoint_pressure_term) + h[i]
    except IndexError:
        q_plus = 0

    try:
        disjoint_pressure_term = -A / (6 * np.pi * h[i - 1] ** 3)
        non_linear_term = ((h[i] + h[i - 1]) / 2) ** pwr
        q_minus = Dx * non_linear_term * (-h[i - 2] + 3 * h[i - 1] - 3 * h[i] + h[i + 1] - disjoint_pressure_term) + h[
            i - 1]
    except IndexError:
        q_minus = 0

    return q_plus, q_minus
