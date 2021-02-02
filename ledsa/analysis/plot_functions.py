import ledsa.core.model
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from PIL import Image

from ledsa import LEDSA
from ..core import led_helper as led
import ledsa.analysis.calculations as calc
from ..core.ConfigData import ConfigData as CD
import os
from typing import Union, Tuple

# os path separator
sep = os.path.sep

"""Some functions simplifying the visualization of the data."""


def plot_z_fitpar(fig: plt.figure, parameter: str, img_id: int, channel: int,
                  led_arrays: Union[Tuple[int, ...], int]) -> None:
    """Plot the height of a LED array against one parameter."""
    # make led_arrays a tuple
    if type(led_arrays) == int:
        led_arrays = (led_arrays,)

    parameterDF = calc.read_hdf(channel)
    parameterDF = calc.include_column_if_nonexistent(parameterDF, parameter, channel)
    parameterDF = parameterDF.loc[img_id, :]

    ax = fig.gca(xlabel=parameter, ylabel='height/m')
    for line in led_arrays:
        plot, = ax.plot(np.array(parameterDF[parameterDF['line'] == line][parameter]),
                        np.array(parameterDF[parameterDF['line'] == line]['height']))
        plot.set_label(f'LED_Array{line}, C{channel}')
    ax.legend()
    plt.title(f'Plot of {parameter} against the height.\n'
              f'Image: {img_id}')


def plot_z_fitpar_from_average(fig: plt.figure, parameter: str, img_id: int, channel: int, 
                               led_arrays: Union[Tuple[int, ...], int], window_size=51):
    """Plot the hight of a led array against an parameter averaged over multiple images."""
    if type(led_arrays) == int:
        led_arrays = (led_arrays,)

    parameterDF = calc.read_hdf(channel)
    parameterDF = calc.include_column_if_nonexistent(parameterDF, parameter, channel)
    cropped_parameters = parameterDF.loc[(float(img_id-(window_size-1)//2), ):
                                            (float(img_id+(window_size-1)//2), )][['line', parameter, 'height']]
    mean = cropped_parameters.mean(axis=0, level='led_id')

    ax = fig.gca(xlabel=parameter, ylabel='height/m')
    for line in led_arrays:
        plot, = ax.plot(np.array(mean[mean['line'] == line][parameter]), np.array(mean[mean['line'] == line]['height']))
        plot.set_label(f'LED_Array{line}, C{channel}')
    ax.legend()
    plt.title(f'Plot of averaged fit parameter {parameter} over time against the height.\n'
              f'Image: {img_id}, window_size: {window_size}')


def plot_t_fitpar(fig: plt.figure, led_id: int, parameter: str, channel: int, image_id_start: int,
                  image_id_finish: int):
    """Plot the time development of parameter."""
    plot_info = _calc_t_fitpar_plot_info(led_id, parameter, channel, image_id_start, image_id_finish)

    ax = fig.gca(xlabel='time[s]', ylabel=parameter)
    plot, = ax.plot(np.array(plot_info['experiment_time']), np.array(plot_info[parameter]))
    plot.set_label(f'LED{led_id}, C{channel}')
    ax.legend()
    plt.title(f'Time plot of Fit Parameter {parameter}')


def plot_t_fitpar_with_moving_average(fig: plt.figure, led_id: int, parameter: str, channel: int, image_id_start: int,
                                      image_id_finish: int, box_size=61):
    """Plot the time development of parameter and its moving average."""
    plot_info = _calc_t_fitpar_plot_info(led_id, parameter, channel, image_id_start, image_id_finish)
    average = plot_info[parameter].rolling(box_size, center=True, win_type='gaussian').sum(std=10) / (10 *
                                                                                                    np.sqrt(2 * np.pi))

    ax = fig.gca(xlabel='time[s]', ylabel=parameter)
    plot, = ax.plot(plot_info['experiment_time'], plot_info[parameter], alpha=0.2)
    plot.set_label(f'LED{led_id}, C{channel}')
    plot, = ax.plot(np.array(plot_info['experiment_time']), average, c=plot.get_color())
    plot.set_label(f'average')
    ax.legend()
    plt.title(f'Time plot of Fit Parameter {parameter}')


def _calc_t_fitpar_plot_info(led_id, parameter, channel, image_id_start, image_id_finish):
    times = led.load_file(".{}analysis{}image_infos_analysis.csv".format(sep, sep), delim=',', dtype=str)
    times = pd.DataFrame(times[:, [0, 3]], columns=['img_id', 'experiment_time'], dtype=np.float64)
    times.set_index('img_id', inplace=True)
    parameterDF = calc.read_hdf(channel)
    parameterDF = calc.include_column_if_nonexistent(parameterDF, parameter, channel)
    idx = pd.IndexSlice
    parameterDF = parameterDF.loc[idx[:, led_id], parameter]
    parameterDF.reset_index(drop=True, level=1, inplace=True)
    plot_info = pd.concat([times, parameterDF], axis=1, sort=False)
    plot_info = plot_info.loc[idx[image_id_start:image_id_finish+1]]
    return plot_info


def plot_led_with_fit(channel: int, time: int, led_id: int, window_radius=10):
    """Plot a single led and the corresponding fit function."""
    fig = plt.figure()

    img_id = led.get_img_id_from_time(time)
    _plot_led(fig, img_id, led_id, channel, window_radius)
    _plot_model(fig, channel, img_id, led_id, window_radius)

    plt.title(f'Channel {channel}, Image {img_id}, LED {led_id}')

    plt.show()


def _plot_led(fig, img_id, led_id, channel, window_radius):
    img = _get_led_img(led.get_time_from_img_id(img_id), led_id, window_radius)

    current_fig = plt.gcf()

    plt.figure(fig.number)
    ax = plt.gca()
    ax.imshow(img.split()[channel])

    plt.figure(current_fig.number)


def _plot_model(fig, channel, img_id, led_id, window_radius):
    mesh = np.meshgrid(np.linspace(0.5, 2 * window_radius - 0.5, 2 * window_radius),
                       np.linspace(0.5, 2 * window_radius - 0.5, 2 * window_radius))
    # load model
    model_params = _load_model(img_id, led_id, channel, window_radius)

    led_model = ledsa.core.model.led_model(mesh[0], mesh[1], model_params[0], model_params[1], model_params[2], model_params[3],
                                           model_params[4], model_params[5], model_params[6], model_params[7])

    current_fig = plt.gcf()

    plt.figure(fig.number)
    ax = plt.gca()
    con = ax.contour(mesh[0], mesh[1], led_model, levels=10, alpha=0.9)
    fig.colorbar(mappable=con, ax=ax)
    ax.scatter(model_params[0], model_params[1], color='Red')
    plt.figure(current_fig.number)


def show_img(img_id=-1, time=-1):
    """Add the image corresponding to time or with id img_id to the current plot."""
    if img_id == -1 and time == -1:
        print('Need Image ID or time to show an image')
        return
    if img_id != -1 and time != -1:
        print('Set either img_id or time to show the image')
        return
    path = _get_img_path()
    if img_id != -1:
        filename = led.get_img_name(img_id)
    else:
        filename = led.get_img_name(led.get_img_id_from_time(time))

    img = Image.open(path + filename)
    current_fig = plt.gcf()

    plt.figure()
    ax = plt.gca()
    ax.imshow(img)

    plt.figure(current_fig.number)


def show_led_diff(channel: int, led_id: int, time1: int, time2: int, window_radius=10):
    """Plot the difference of the led color value for one led at two different times."""
    led1 = _get_led_img(time1, led_id, window_radius)
    led2 = _get_led_img(time2, led_id, window_radius)

    current_fig = plt.gcf()

    fig = plt.figure()
    ax = plt.gca()
    im = ax.imshow(np.array(led1, dtype='int16')[:, :, channel] - np.array(led2, dtype='int16')[:, :, channel])
    fig.colorbar(mappable=im, ax=ax)
    plt.title(f'LED diff between LED {led_id} at {time1} and {time2} seconds')

    plt.figure(current_fig.number)


def _get_img_path():
    conf = CD()
    return conf['DEFAULT']['img_directory']


def _get_led_pos(led_id):
    positions = led.load_file(".{}analysis{}led_search_areas.csv".format(sep, sep), delim=',', dtype=int)
    for i in range(positions.shape[0]):
        if positions[i, 0] == led_id:
            return float(positions[i, 1]), float(positions[i, 2])
    raise NameError("Could not find the positions of led {}.".format(led_id))


def _fit_led(img_id, led_id, channel):
    ledsa = LEDSA(channels=channel, build_experiment_infos=False)
    ledsa.load_line_indices()
    ledsa.load_search_areas()
    filename = led.get_img_name(img_id)
    fit_res = led.generate_analysis_data(filename, ledsa.search_areas, ledsa.line_indices, ledsa.config['analyse_photo'], True,
                                         led_id)
    return fit_res


def _load_model(img_id, led_id, channel, window_radius=10):
    parameterDF = calc.read_hdf(channel)
    model_params = np.array(parameterDF.loc[img_id, led_id])[1:9]
    pix_pos = _get_led_pos(led_id)
    model_params[0:2] = model_params[0:2] - pix_pos + window_radius
    return model_params


def _get_led_img(time, led_id, window_radius=10):
    filename = led.get_img_name(led.get_img_id_from_time(time))
    path = _get_img_path()
    led_im = Image.open(path + filename)

    x, y = _get_led_pos(led_id)
    led_im = led_im.crop((y - window_radius,
                          x - window_radius,
                          y + window_radius,
                          x + window_radius))
    return led_im
