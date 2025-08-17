import numpy as np
from scipy.interpolate import interp1d

def temporal_error(steady:np.ndarray, startup:np.ndarray):
    steady = interpolate(steady, startup)
    return np.linalg.norm(steady-startup)/np.linalg.norm(steady) * 100

def spatial_error(steady:np.ndarray, startup:np.ndarray):
    steady = interpolate(steady, startup)
    return np.abs(steady-startup)

def interpolate(steady, startup):
    f = interp1d(np.linspace(0, 1, len(steady)), steady)
    steady = f(np.linspace(0, 1, len(startup)))
    return steady