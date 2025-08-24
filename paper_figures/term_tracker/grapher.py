import numpy as np
import matplotlib.pyplot as plt
import json
import matplotlib as mpl

try:
    with open('../../glob_var/global_variables.json') as f:
        GV = json.load(f)
except FileNotFoundError:
    print("Global variables json not found!")


def grapher(max_t:float, data:list, data_dp:list, times:list, times_dp:list, term:str, y_lims:dict):

    c = np.arange(0, max_t)
    norm = mpl.colors.Normalize(vmin=c.min(), vmax=c.max())
    cmap = mpl.cm.ScalarMappable(norm=norm, cmap=GV['cmap'])
    cmap.set_array([])

    fig, ax = plt.subplots(nrows=3, ncols=2, figsize=(15, 15))
    fig.subplots_adjust(hspace=0.1)

    num_intervals = 5

    LW = 2.5

    def plot_lines(axis, sol_y, sol_t, ls='-'):
        sol_y = sol_y.transpose(1, 0)
        step = int(len(sol_t)/num_intervals)
        rgb_step = int(max(sol_t) / num_intervals)
        intermediate_array = sol_y[:, ::step]
        for i in range(1, num_intervals):
            axis.plot(GV['x'][1:298], intermediate_array[:, i], c=cmap.to_rgba(i * rgb_step), linewidth=LW, linestyle=ls)
    newt, thin, thic = data
    newt_dp, thin_dp, thic_dp = data_dp
    newt_t, thin_t, thic_t = times
    newt_t_dp, thin_t_dp, thic_t_dp = times_dp

    plot_lines(ax[0, 0], newt[:, 0, 1:298], newt_t, ls='-')
    plot_lines(ax[0, 0], newt[:, 1, 1:298], newt_t, ls='--')

    plot_lines(ax[0, 1], newt_dp[:, 0, 1:298], newt_t_dp, ls='-')
    plot_lines(ax[0, 1], newt_dp[:, 1, 1:298], newt_t_dp, ls='--')

    plot_lines(ax[1, 0], thin[:, 0, 1:298], thin_t, ls='-')
    plot_lines(ax[1, 0], thin[:, 1, 1:298], thin_t, ls='--')

    plot_lines(ax[1, 1], thin_dp[:, 0, 1:298], thin_t_dp, ls='-')
    plot_lines(ax[1, 1], thin_dp[:, 1, 1:298], thin_t_dp, ls='--')

    plot_lines(ax[2, 0], thic[:, 0, 1:298], thic_t, ls='-')
    plot_lines(ax[2, 0], thic[:, 1, 1:298], thic_t, ls='--')

    plot_lines(ax[2, 1], thic_dp[:, 0, 1:298], thic_t_dp, ls='-')
    plot_lines(ax[2, 1], thic_dp[:, 1, 1:298], thic_t_dp, ls='--')

    ax[0, 0].grid(True)
    ax[0, 0].set_ylabel("Newtonian Fluid no DP", fontsize=14)
    ax[0, 0].set_xticks(np.arange(0, GV['L'] + 1, 2))
    ax[0, 0].set_xticklabels([])
    ax[0, 0].tick_params(axis='y', labelsize=14)
    ax[0, 0].set_title("(A)", fontsize=16)

    ax[0, 1].grid(True)
    ax[0, 1].set_ylabel("Newtonian Fluid, $DP=0.15$", fontsize=14)
    ax[0, 1].set_xticks(np.arange(0, GV['L'] + 1, 2))
    ax[0, 1].set_xticklabels([])
    ax[0, 1].tick_params(axis='y', labelsize=14)
    ax[0, 1].set_title("(B)", fontsize=16)

    ax[1, 0].grid(True)
    ax[1, 0].set_ylabel("Shear-thinning Fluid no DP", fontsize=14)
    ax[1, 0].set_xticks(np.arange(0, GV['L'] + 1, 2))
    ax[1, 0].set_xticklabels([])
    ax[1, 0].tick_params(axis='y', labelsize=14)

    ax[1, 1].grid(True)
    ax[1, 1].set_ylabel("Shear-thinning Fluid $DP=0.15$", fontsize=14)
    ax[1, 1].set_xticks(np.arange(0, GV['L'] + 1, 2))
    ax[1, 1].set_xticklabels([])
    ax[1, 1].tick_params(axis='y', labelsize=14)

    ax[2, 0].grid(True)
    ax[2, 0].set_ylabel("Shear-thickening Fluid no DP", fontsize=14)
    ax[2, 0].set_xticks(np.arange(0, GV['L'] + 1, 2))
    ax[2, 0].set_xlabel("Surface Length $(x)$", fontsize=14)
    ax[2, 0].tick_params(axis='y', labelsize=14)
    ax[2, 0].tick_params(axis='x', labelsize=14)

    ax[2, 1].grid(True)
    ax[2, 1].set_ylabel("Shear-thickening Fluid $DP=0.15$", fontsize=14)
    ax[2, 1].set_xticks(np.arange(0, GV['L'] + 1, 2))
    ax[2, 1].set_xlabel("Surface Length $(x)$", fontsize=14)
    ax[2, 1].tick_params(axis='y', labelsize=14)
    ax[2, 1].tick_params(axis='x', labelsize=14)

    fig.text(0.02, 0.5, "Term Strength", rotation='vertical', fontsize=16, va='center', ha='center')

    ax[0, 0].set_ylim(y_lims['newt'])
    ax[1, 0].set_ylim(y_lims['thin'])
    ax[2, 0].set_ylim(y_lims['thic'])

    ax[0, 1].set_ylim(y_lims['newt_dp'])
    ax[1, 1].set_ylim(y_lims['thin_dp'])
    ax[2, 1].set_ylim(y_lims['thic_dp'])

    cbar_ax = fig.add_axes([0.92, 0.11, 0.03, 0.77])
    cb = fig.colorbar(cmap, cax=cbar_ax, ticks=np.arange(0, max_t + 1, 5))
    cb.set_label("Time (s)", fontsize=16)
    cb.ax.tick_params(labelsize=14)

    fig.suptitle(f"Timeseries graph showing evolution of the {term} in time", fontsize=18, y=0.92)

    fig.savefig(f"graphs/{term}.png", bbox_inches='tight')

def grapher_dp(data, times, max_t, y_lims, term):

    c = np.arange(0, max_t)
    norm = mpl.colors.Normalize(vmin=c.min(), vmax=c.max())
    cmap = mpl.cm.ScalarMappable(norm=norm, cmap=GV['cmap'])
    cmap.set_array([])

    fig, ax = plt.subplots(nrows=3, ncols=1, figsize=(15, 15))
    fig.subplots_adjust(hspace=0.1)

    num_intervals = 5

    LW = 2.5

    def plot_lines(axis, sol_y, sol_t, ls='-'):
        sol_y = sol_y.transpose(1, 0)
        step = int(len(sol_t)/num_intervals)
        rgb_step = int(max(sol_t) / num_intervals)
        intermediate_array = sol_y[:, ::step]
        for i in range(1, num_intervals):
            axis.plot(GV['x'][1:298], intermediate_array[:, i], c=cmap.to_rgba(i * rgb_step), linewidth=LW, linestyle=ls)

    newt, thin, thic = data
    newt_t, thin_t, thic_t = times

    plot_lines(ax[0], newt[:, 0, 1:298], newt_t, ls='-')
    plot_lines(ax[0], newt[:, 1, 1:298], newt_t, ls='--')

    plot_lines(ax[1], thin[:, 0, 1:298], thin_t, ls='-')
    plot_lines(ax[1], thin[:, 1, 1:298], thin_t, ls='--')

    plot_lines(ax[2], thic[:, 0, 1:298], thic_t, ls='-')
    plot_lines(ax[2], thic[:, 1, 1:298], thic_t, ls='--')

    ax[0].grid(True)
    ax[0].set_ylabel("Newtonian Fluid, $DP=0.15$", fontsize=14)
    ax[0].set_xticks(np.arange(0, GV['L'] + 1, 2))
    ax[0].set_xticklabels([])
    ax[0].tick_params(axis='y', labelsize=14)
    ax[0].set_title("(B)", fontsize=16)

    ax[1].grid(True)
    ax[1].set_ylabel("Shear-thinning Fluid $DP=0.15$", fontsize=14)
    ax[1].set_xticks(np.arange(0, GV['L'] + 1, 2))
    ax[1].set_xticklabels([])
    ax[1].tick_params(axis='y', labelsize=14)

    ax[2].grid(True)
    ax[2].set_ylabel("Shear-thickening Fluid $DP=0.15$", fontsize=14)
    ax[2].set_xticks(np.arange(0, GV['L'] + 1, 2))
    ax[2].set_xlabel("Surface Length $(x)$", fontsize=14)
    ax[2].tick_params(axis='y', labelsize=14)
    ax[2].tick_params(axis='x', labelsize=14)

    fig.text(0.02, 0.5, "Term Strength", rotation='vertical', fontsize=16, va='center', ha='center')

    ax[0].set_ylim(y_lims['newt_dp'])
    ax[1].set_ylim(y_lims['thin_dp'])
    ax[2].set_ylim(y_lims['thic_dp'])

    cbar_ax = fig.add_axes([0.92, 0.11, 0.03, 0.77])
    cb = fig.colorbar(cmap, cax=cbar_ax, ticks=np.arange(0, max_t + 1, 5))
    cb.set_label("Time (s)", fontsize=16)
    cb.ax.tick_params(labelsize=14)

    fig.suptitle(f"Timeseries graph showing evolution of the {term} in time", fontsize=18, y=0.92)

    fig.savefig(f"graphs/{term}.png", bbox_inches='tight')