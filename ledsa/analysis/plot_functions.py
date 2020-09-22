import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from ..core import _led_helper as led
import ledsa.analysis.calculations as calc
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
    "fit_nfev": 12,
    "normalized_A": 13
}


def plot_z_fitpar(fig: plt.figure, fit_par: str, img_id: int, channel: int,
                  led_arrays: Union[Tuple[int, ...], int]) -> None:
    """plots the height of a LED array against one fit parameter"""
    # make led_arrays a tuple
    if type(led_arrays) == int:
        led_arrays = (led_arrays,)

    fit_parameters = calc.read_hdf(channel)
    fit_parameters = calc.include_column_if_nonexistent(fit_parameters, fit_par, channel)
    print(fit_parameters)
    fit_parameters = fit_parameters.loc[img_id, :]

    ax = fig.gca(xlabel=fit_par, ylabel='height/m')
    for line in led_arrays:
        plot, = ax.plot(fit_parameters[fit_parameters['line'] == line][fit_par],
                        fit_parameters[fit_parameters['line'] == line]['height'])
        plot.set_label(f'LED_Array{line}, C{channel}')
    ax.legend()
    plt.title(f'Plot of fit parameter {fit_par} against the height.\n'
              f'Image: {img_id}')


def plot_z_fitpar_from_average(fig, fit_par, img_id, channel, led_arrays, window_size=51):
    # make led_arrays a tuple
    if type(led_arrays) == int:
        led_arrays = (led_arrays,)

    fit_parameters = calc.read_hdf(channel)
    fit_parameters = calc.include_column_if_nonexistent(fit_parameters, fit_par, channel)
    cropped_parameters = fit_parameters.loc[(float(img_id-(window_size-1)//2), ):
                                            (float(img_id+(window_size-1)//2), )][['line', fit_par, 'height']]
    mean = cropped_parameters.mean(axis=0, level='led_id')      # .to_numpy()

    print(cropped_parameters)
    print(mean)

    ax = fig.gca(xlabel=fit_par, ylabel='height/m')
    for line in led_arrays:
        plot, = ax.plot(mean[mean['line'] == line][fit_par], mean[mean['line'] == line]['height'])
        plot.set_label(f'LED_Array{line}, C{channel}')
    ax.legend()
    plt.title(f'Plot of averaged fit parameter {fit_par} over time against the height.\n'
              f'Image: {img_id}, window_size: {window_size}')


def plot_t_fitpar(fig, led_id, fit_par, channel, image_id_start, image_id_finish):
    """Plots the time development of a fit parameter"""
    plot_info = _calc_t_fitpar_plot_info(led_id, fit_par, channel, image_id_start, image_id_finish)

    ax = fig.gca(xlabel='time[s]', ylabel=fit_par)
    plot, = ax.plot(plot_info['experiment_time'], plot_info[fit_par])
    plot.set_label(f'LED{led_id}, C{channel}')
    ax.legend()
    plt.title(f'Time plot of Fit Parameter {fit_par}')


def plot_t_fitpar_with_moving_average(fig, led_id, fit_par, channel, image_id_start, image_id_finish, box_size=61):
    """Plots the time development of a fit parameter and its moving average"""
    plot_info = _calc_t_fitpar_plot_info(led_id, fit_par, channel, image_id_start, image_id_finish)
    average = plot_info[fit_par].rolling(box_size, center=True, win_type='gaussian').sum(std=10) / 10 / np.sqrt(2 * np.pi)

    ax = fig.gca(xlabel='time[s]', ylabel=fit_par)
    plot, = ax.plot(plot_info['experiment_time'], plot_info[fit_par], alpha=0.2)
    plot.set_label(f'LED{led_id}, C{channel}')
    plot, = ax.plot(plot_info['experiment_time'], average, c=plot.get_color())
    plot.set_label(f'average')
    ax.legend()
    plt.title(f'Time plot of Fit Parameter {fit_par}')


def _calc_t_fitpar_plot_info(led_id, fit_par, channel, image_id_start, image_id_finish):
    times = led.load_file(".{}analysis{}image_infos_analysis.csv".format(sep, sep), delim=',', dtype=str)
    times = pd.DataFrame(times[:, [0, 3]], columns=['img_id', 'experiment_time'], dtype=np.float64)
    times.set_index('img_id')
    fit_parameters = calc.read_hdf(channel)
    fit_parameters = calc.include_column_if_nonexistent(fit_parameters, fit_par, channel)
    idx = pd.IndexSlice
    fit_parameters = fit_parameters.loc[idx[:, led_id], fit_par]
    fit_parameters = fit_parameters.reset_index()[fit_par]
    plot_info = pd.concat([times, fit_parameters], axis=1, sort=False)
    plot_info = plot_info[plot_info['img_id'] >= image_id_start]
    return plot_info[plot_info['img_id'] <= image_id_finish]



