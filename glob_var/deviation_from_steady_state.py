import numpy as np
import json
from scipy.interpolate import interp1d

## This file contains the function for computing the absolute error in deviation from the steady state of its respective
# model. This is intended for the heatmap and serves as the metric for the colorbar going from stable to unstable.

def magnitude_of_deviation(steady_state:np.ndarray, startup_flow:np.ndarray, scalar:bool=True):
    """Computes the magnitude of the absolute error between the startup-flow and the relevant steady state condition.
    Returns a floating point integer between 0 and 1, where 0 is an error of < 1e-4, and 1 is an error of > 1"""
    if steady_state.shape != startup_flow.shape:
        # print(f"Array sizes between steady state and startup flow do not match ({steady_state.shape} != "
        #       f"{startup_flow.shape}), interpolating...")
        f = interp1d(np.linspace(0, 1, len(startup_flow)), startup_flow)
        startup_flow = f(np.linspace(0, 1, len(steady_state)))
    assert startup_flow.shape == steady_state.shape, print(f"""Startup-flow shape is not the same as the steady state
                                                           shape, {startup_flow.shape} != {steady_state.shape}""")

    if scalar:
        return abs(np.linalg.norm(steady_state - startup_flow))
    else:
        return abs(steady_state-startup_flow)
