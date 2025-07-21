## This file finds the minimum time elapsed such that the difference between the startup flow and its steady state for
# each core model are below a certain tolerance.

import numpy as np

tolerance = 1e-4
placeholder_t_min = 100


if __name__ == '__main__':
    np.save('data/min_t', placeholder_t_min)
