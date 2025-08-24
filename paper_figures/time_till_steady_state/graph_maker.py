import numpy as np
from map_gen import make_heatmap

times_newt = np.load("newt_data/times.npy")
times_thin = np.load("thin_data/times.npy")
times_thic = np.zeros((24,24))

assert times_newt.shape == times_thin.shape == times_thic.shape

Qs = np.linspace(0.7, 0.95, times_newt.shape[0])
As = np.linspace(0.0, 0.5, times_newt.shape[0])

Z = [times_newt, times_thin, times_thic]

make_heatmap(x=As, y=Qs, Z=Z, dir="ttss")