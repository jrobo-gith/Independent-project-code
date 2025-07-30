from glob_var.FVM.FVM_RHS import FVM_RHS
import numpy as np

def unstable(t, h, args):
    """Returns minimum of the array h, if the value is 0, the solver terminates."""
    return min(h)

def steady_state(t, h, args):
    """Triggers an event when the time derivative is nearly 0, meaning the system has reached a near-steady state"""
    dhdt = FVM_RHS(t, h, args)
    return np.linalg.norm(dhdt) - 1e-5