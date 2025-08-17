import numpy as np
import matplotlib.pyplot as plt
from glob_var.animation import Animation

ADV = np.load("newtonian_collector/ADV.npy")
ADV_dp = np.load("newtonian_collector/ADV_dp.npy")

NLT = np.load("newtonian_collector/NLT.npy")
NLT_dp = np.load("newtonian_collector/NLT_dp.npy")

TOT = np.load("newtonian_collector/TOT.npy")
TOT_dp = np.load("newtonian_collector/TOT_dp.npy")

DPT = np.load("newtonian_collector/DPT.npy")
DPT_dp = np.load("newtonian_collector/DPT_dp.npy")

T = np.load("newtonian_collector/T.npy")
T_dp = np.load("newtonian_collector/T_dp.npy")

print(ADV.shape, NLT.shape, TOT.shape, DPT.shape, T.shape)
print(ADV_dp.shape, NLT_dp.shape, TOT_dp.shape, DPT_dp.shape, T_dp.shape)

ADV = np.transpose(ADV, axes=(1, 2, 0))
ADV_dp = np.transpose(ADV_dp, axes=(1, 2, 0))

# fig_details = {
#     'x-lim': (0, 16),
#     'y-lim': (0, 1),
#     'legend': [False,],
#     'grid': [True],
#     'title': [f"Startup Flow of adv term", True],
#     'x-label': ['Surface Length $(x)$'],
#     'y-label': ['Advection Strength'],
# }
# # Instantiate the Animation class
# newt_animation = Animation(num_rows=1, num_cols=1,
#                            fig_size=(10, 8), x=np.linspace(0, 16, ADV_dp.shape[1]-3),
#                            data=[ADV_dp[:, 1:298, :],], min_timestep=T_dp.shape[0],
#                            fig_details=fig_details, interval=20, title_updates=T_dp)
# newt_animation.instantiate_animation()
# newt_animation.save_animation('animations/test')
