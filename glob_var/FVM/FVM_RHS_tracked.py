import numpy as np
import os
import json

current_dir = os.path.dirname(__file__)
project_root = os.path.abspath(os.path.join(current_dir, '..', '..'))

try:
    with open(project_root + '/glob_var/global_variables.json') as f:
        GV = json.load(f)
except FileNotFoundError:
    print("Global variables json not found!")

def make_ode(collector):
    def FVM_RHS(t: float, h: np.ndarray, args: tuple) -> np.ndarray:
        """
        RHS of equation, made for scipy's solve_ivp function that takes care of this stiff fourth order PDE.
        """

        make_step, dx, pwr, Q, _, n, _, N = args

        local_collector = np.zeros((2, len(collector), h.shape[0]))

        h = h.copy()
        dhdt = np.zeros_like(h)

        # i = 0
        h[0] = GV['h0']

        # i = 1
        q_plus, q_minus, ADV, NLT, TOT, DPT = make_step(h=h, i=1, args=args)
        dhdt[1] = - (q_plus - Q) / dx

        local_collector[:, 0, 1] = ADV
        local_collector[:, 1, 1] = NLT
        local_collector[:, 2, 1] = TOT
        local_collector[:, 3, 1] = DPT

        # i = N - 2
        q_plus, q_minus, ADV, NLT, TOT, DPT = make_step(h=h, i=N - 2, args=args)
        dhdt[N - 2] = - (h[N - 2] - q_minus) / dx

        local_collector[:, 0, -1] = ADV
        local_collector[:, 1, -1] = NLT
        local_collector[:, 2, -1] = TOT
        local_collector[:, 3, -1] = DPT

        # i = N - 1
        h[N - 1] = h[N - 2]
        dhdt[N - 1] = dhdt[N - 2]

        for i in range(2, N - 2):
            q_plus, q_minus, ADV, NLT, TOT, DPT = make_step(h=h, i=i, args=args)
            dhdt[i] = -(q_plus - q_minus) / dx

            local_collector[:, 0, i] = ADV
            local_collector[:, 1, i] = NLT
            local_collector[:, 2, i] = TOT
            local_collector[:, 3, i] = DPT

        collector[0].append(local_collector[:, 0, :])
        collector[1].append(local_collector[:, 1, :])
        collector[2].append(local_collector[:, 2, :])
        collector[3].append(local_collector[:, 3, :])
        collector[4].append(t)

        return dhdt

    return FVM_RHS