## File for making animations for startup-flow models.
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
import numpy as np

class Animation:

    """
    Animation class for creating animations that move across time and is able to plot multiple rows and columns.

    The data array has a VERY specific layout.
    If I want to plot an animation with 4 plots, 2 rows and 2 columns, the data array will be structured like
        data = [[plot1, plot2],
                [plot3, plot4]]

    Inside each plot (plotx), we have a three-dimensional array of lines, time, and space. So if I wanted plot 1 to
    have 3 lines and plot 2 needed only 1 line, the arrays would look like:
        plot1 = [(line, space, time),
                 (line, space, time),
                 (line, space, time)]

        plot2 = [(line, space, time),]
    """

    def __init__(self, num_rows:int, num_cols:int, fig_size:tuple, x:np.ndarray, data:list, num_frames:int, fig_details:dict):
        self.animation = None
        self.num_rows = num_rows
        self.num_cols = num_cols
        self.fig_size = fig_size
        self.x = x
        self.data = data
        self.num_frames = num_frames
        self.fig_details = fig_details

        # Initialise figure
        self.fig, self.ax = plt.subplots(nrows=self.num_rows, ncols=self.num_cols, figsize=self.fig_size)

        self.anim_plot = []

        if self.num_rows == 1 and self.num_cols == 1:
            self.anim_plot.append(self.ax.plot([], [])[0])

            self.ax.set_xlim(fig_details['x-lim'])
            self.ax.set_ylim(fig_details['y-lim'])
            self.ax.grid(True)
            if self.fig_details['legend'][0]:
                self.ax.legend()

        else:
            if self.num_rows == 1:
                for i in range(self.num_cols):
                    self.anim_plot.append(self.ax[i].plot([], [])[0])

                    self.ax[i].set_xlim(fig_details['x-lim'])
                    self.ax[i].set_ylim(fig_details['y-lim'])
                    self.ax[i].grid(True)
                    if self.fig_details['legend'][i]:
                        self.ax[i].legend()

            elif self.num_cols == 1:
                for i in range(self.num_rows):
                    self.anim_plot.append(self.ax[i].plot([], [])[0])

                    self.ax[i].set_xlim(fig_details['x-lim'])
                    self.ax[i].set_ylim(fig_details['y-lim'])
                    self.ax[i].grid(True)
                    if self.fig_details['legend'][i]:
                        self.ax[i].legend()

            else:
                p_num = 0
                for i in range(self.num_rows):
                    for j in range(self.num_cols):
                        self.anim_plot.append(self.ax[i, j].plot([], [])[0])

                        self.ax[i, j].set_xlim(fig_details['x-lim'])
                        self.ax[i, j].set_ylim(fig_details['y-lim'])
                        self.ax[i, j].grid(True)
                        if self.fig_details['legend'][p_num]:
                            self.ax[i].legend()
                        p_num += 1


    def update_data(self, frame):
        plot_num = 0

        for i in range(self.num_rows):
            for j in range(self.num_cols):
                active_plot = self.anim_plot[plot_num]
                active_data = self.data[plot_num]
                [active_plot.set_data(self.x, active_data[k, :, frame]) for k in range(active_data.shape[0])]
                plot_num += 1


    def instantiate_animation(self):
        self.animation = FuncAnimation(
                    fig = self.fig,
                    func = self.update_data,
                    frames = self.num_frames,
                    interval = 25,
                )

    def show_animation(self):
        self.fig.show()

    def save_animation(self, directory):
        self.animation.save(directory + '.gif', writer='pillow', fps=24)
