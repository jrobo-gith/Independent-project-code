## File for making animations for startup-flow models.
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
import numpy as np
from scipy.interpolate import interp1d

class Animation:

    """
    Animation class for creating animations that move across time and is able to plot multiple rows and columns.

    The data array has a VERY specific layout.
    If I want to plot an animation with 4 plots, 2 rows and 2 columns, the data array will be structured like
        data = [plot1, plot2, plot3, plot4]

    Inside each plot (plotx), we have a three-dimensional array of lines, time, and space. So if I wanted plot 1 to
    have 3 lines and plot 2 needed only 1 line, the arrays would look like:
        plot1 = [(space, time),
                 (space, time),
                 (space, time)]

        plot2 = [(space, time),]
    """

    def __init__(self, fig_size:tuple, x:np.ndarray, data:list, fig_details:dict,
                 min_timestep:int, num_rows:int=1, num_cols:int=1):
        self.animation = None
        self.num_rows = num_rows
        self.num_cols = num_cols
        self.fig_size = fig_size
        self.x = x
        self.data = data
        self.fig_details = fig_details
        self.min_timestep = min_timestep

        # Initialise figure
        self.fig, self.ax = plt.subplots(nrows=self.num_rows, ncols=self.num_cols, figsize=self.fig_size)

        # Make axes into a list so they're easily looped through
        self.ax = np.array(self.ax).flatten()
        self.anim_plot = []


        for i, PLOT in enumerate(self.ax):
            self.anim_plot.append(PLOT.plot([], [])[0])

            # Add plot details
            PLOT.set_xlim(fig_details['x-lim'])
            PLOT.set_ylim(fig_details['y-lim'])
            PLOT.set_title(fig_details['title'][i])
            PLOT.set_xlabel(fig_details['x-label'][i])
            PLOT.set_ylabel(fig_details['y-label'][i])

            if fig_details['legend'][i]:
                PLOT.legend()
            if fig_details['grid'][i]:
                PLOT.grid(True)

        self.fig.tight_layout()

    def update_data(self, frame):
        plot_num = 0

        # Use interpolation to make all the time steps equal so we can loop through them equally
        time_of_min = np.linspace(0, 1, self.min_timestep)

        for i in range(self.num_rows):
            for j in range(self.num_cols):
                active_plot = self.anim_plot[plot_num]
                active_data = self.data[plot_num]
                old_time = np.linspace(0, 1, active_data.shape[2])
                f = interp1d(old_time, active_data, axis=2)
                interpolated_data = f(time_of_min)
                [active_plot.set_data(self.x, interpolated_data[k, :, frame]) for k in range(interpolated_data.shape[0])]
                plot_num += 1

    def instantiate_animation(self):
        self.animation = FuncAnimation(
                    fig = self.fig,
                    func = self.update_data,
                    frames = self.min_timestep,
                    interval = 10,
                )

    def save_animation(self, directory):
        self.animation.save(directory + '.gif', writer='pillow', fps=24)