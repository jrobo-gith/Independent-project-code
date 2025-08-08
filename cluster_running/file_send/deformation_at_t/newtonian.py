import numpy as np
import json
from scipy.integrate import solve_ivp
from  cluster_running.file_send.glob.FVM_RHS import FVM_RHS, make_step



def unstable(t, h, args):
    """Returns minimum of the array h, if the value is 0, the solver terminates."""
    return min(h)

def steady_state(t, h, args):
    """Triggers an event when the time derivative is nearly 0, meaning the system has reached a near-steady state"""
    dhdt = FVM_RHS(t, h, args)
    return np.linalg.norm(dhdt) - 1e-4

resolution = 5
Qs = np.linspace(0.1, 0.95, resolution)
As = np.linspace(0, 0.3, resolution)

data = np.zeros((len(Qs), len(As), ))