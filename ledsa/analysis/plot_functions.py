import matplotlib.pyplot as plt
import numpy as np
from ..core import _led_helper as led
import os

# os path separator
sep = os.path.sep


def plot_z_fitpar(fig, fit_par, image_id, channel, led_array):
    """plots the height of a LED array against one fit parameter"""

    # dictionary of the fit parameter positions
    par_dic = {
        "x": 2,
        "y": 3,
        "dx": 4,
        "dy": 5,
        "A": 6,
        "alpha": 7,
        "wx": 8,
        "wy": 9,
        "fit_fun": 11,
        "fit_nfev": 12
    }

    # TODO: check if it works when leds are ignored
    # load the experiment data
    coord = led.load_file(".{}analysis{}led_search_areas_with_coordinates.csv".format(sep, sep), delim=',')
    parameters = led.load_file(".{}analysis{}channel{}{}{}_led_positions.csv".format(sep, sep, channel, sep, image_id), delim=',')

    # only keep the parameter of interest
    parameters = np.concatenate([parameters[:, :2],  parameters[:, par_dic[fit_par]:par_dic[fit_par]+1]], axis=1)

    # sort over the LED id column
    parameters = parameters[parameters[:, 0].argsort()]

    # add the z coordinate
    parameters = np.concatenate([parameters, coord[:, -1:]], axis=1)

    # only take the elements of the specified LED array
    mask = parameters[:, 1] == led_array
    parameters = parameters[mask]

    # make the plot
    ax = fig.gca(xlabel=fit_par, ylabel='h[m]')
    plot, = ax.plot(parameters[:, 2], parameters[:, 3])
    plot.set_label('LED_array {}'.format(led_array))
    ax.legend()
    plt.title('Image ID: {} - Channel: {}'.format(image_id, channel))
