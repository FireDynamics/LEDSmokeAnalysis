import matplotlib.pyplot as plt
import numpy as np
from ..core import _led_helper as led
import os
from typing import Union, Tuple

# os path separator
sep = os.path.sep

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


def plot_z_fitpar(fig: plt.figure, fit_par: str, image_id: int, channel: int,
                  led_arrays: Union[Tuple[int, ...], int]) -> None:
    """plots the height of a LED array against one fit parameter"""
    # make led_arrays a tuple
    if type(led_arrays) == int:
        led_arrays = (led_arrays,)

    # TODO: check if it works when leds are ignored
    # load the experiment data
    coord = led.load_file(".{}analysis{}led_search_areas_with_coordinates.csv".format(sep, sep), delim=',')
    parameters = led.load_file(".{}analysis{}channel{}{}{}_led_positions.csv".format(sep, sep, channel, sep, image_id),
                               delim=',')

    # only keep the parameter of interest
    parameters = np.concatenate([parameters[:, :2], parameters[:, par_dic[fit_par]:par_dic[fit_par] + 1]], axis=1)

    # sort over the LED id column
    parameters = parameters[parameters[:, 0].argsort()]

    # add the z coordinate
    parameters = np.concatenate([parameters, coord[:, -1:]], axis=1)

    for array in led_arrays:
        # only take the elements of one LED array
        mask = parameters[:, 1] == array
        array_parameters = parameters[mask]

        # make the plot
        ax = fig.gca(xlabel=fit_par, ylabel='h[m]')
        plot, = ax.plot(array_parameters[:, 2], array_parameters[:, 3])
        plot.set_label('LED_array {}'.format(array))

    # add legend and title
    ax.legend()
    plt.title('Image ID: {} - Channel: {}'.format(image_id, channel))


def plot_t_fitpar(fig, led_id, fit_par, channel, image_id_start, image_id_finish, skip_images=0):
    """Plots the time development of a fit parameter"""
    times = led.load_file(".{}analysis{}image_infos_analysis.csv".format(sep, sep), delim=',', dtype=str)
    plot_info = np.array([[0, 0]])

    # find time and fit parameter for every image
    for image_id in range(image_id_start, image_id_finish+1, skip_images + 1):
        try:
            parameters = led.load_file(".{}analysis{}channel{}{}{}_led_positions.csv".format(
                sep, sep, channel, sep, image_id), delim=',', silent=True)
        except Exception as err:
            print('Warning:', err)
            print('Will only use the files loaded before.')
            break

        # get the row of parameters corresponding to the led_id
        led_info = parameters[parameters[:, 0] == led_id].flatten()

        # get the time corresponding to the image id
        time = times[times[:, 0] == str(image_id)]
        plot_info = np.append(plot_info, [[time[0, 3], led_info[par_dic[fit_par]]]], axis=0)

    # delete the row used for initialization
    plot_info = np.delete(plot_info, 0, 0)

    # make the plot
    ax = fig.gca(xlabel='time[s]', ylabel=fit_par)
    plot, = ax.plot(plot_info[:, 0], plot_info[:, 1])
    plot.set_label('LED {}'.format(led_id))
    ax.legend()
    plt.title('Time plot of Fit Parameter {} for Channel {}'.format(fit_par, channel))
