import numpy as np

def check_for_success(solution, Q):
    """This function decides whether to claim that the film has been successful, where the status of the ivp solver returns 0, or
    whether that has been some deformation in the fluid but not significant such that it would affect the fluid, or that
    the fluid fails altogether, where the status returned is -1"""
    # Add success / deformation / fail to stability details
    if solution.status == 0:
        # Compute threshold based on flux
        threshold = Q - 0.02
        # Ask whether the film height moves below the threshold
        if np.min(solution.y[:, -1]) < threshold:
            success = 2
        else:
            # If it doesn't (False), then it has not deformed too much
            success = 1
    else:
        # The film has become unstable
        success = 0
    return success