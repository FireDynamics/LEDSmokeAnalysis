import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from PIL import Image

from ledsa import LEDSA
from ..core import _led_helper as led
import ledsa.analysis.calculations as calc
from ..core.ledsa_conf import ConfigData as CD
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
    mean = cropped_parameters.mean(axis=0, level='led_id')

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
    times.set_index('img_id', inplace=True)
    fit_parameters = calc.read_hdf(channel)
    fit_parameters = calc.include_column_if_nonexistent(fit_parameters, fit_par, channel)
    idx = pd.IndexSlice
    fit_parameters = fit_parameters.loc[idx[:, led_id], fit_par]
    fit_parameters.reset_index(drop=True, level=1, inplace=True)
    plot_info = pd.concat([times, fit_parameters], axis=1, sort=False)
    plot_info = plot_info.loc[idx[image_id_start:image_id_finish+1]]
    return plot_info


def plot_led_with_fit(channel, time, led_id, window_radius=10):
    fig = plt.figure()

    img_id = led.get_img_id_from_time(time)
    plot_led(fig, img_id, led_id, channel, window_radius)
    plot_model(fig, channel, img_id, led_id, window_radius)

    plt.title(f'Channel {channel}, Image {img_id}, LED {led_id}')

    # plt.savefig(model.png)
    plt.show()


def plot_led(fig, img_id, led_id, channel, window_radius):
    img = get_led_img(led.get_time_from_img_id(img_id), led_id, window_radius)

    current_fig = plt.gcf()

    plt.figure(fig.number)
    ax = plt.gca()
    ax.imshow(img.split()[channel])

    plt.figure(current_fig.number)


def plot_model(fig, channel, img_id, led_id, window_radius):
    mesh = np.meshgrid(np.linspace(0.5, 2 * window_radius - 0.5, 2 * window_radius),
                       np.linspace(0.5, 2 * window_radius - 0.5, 2 * window_radius))

    fit_results = fit_led(img_id, led_id, channel)
    model_params = fit_results.x
    print(model_params)
    print(fit_results.keys())
    print(fit_results)
    led_model = led.led_fit(mesh[0], mesh[1], model_params[0], model_params[1], model_params[2], model_params[3],
                            model_params[4], model_params[5], model_params[6], model_params[7])

    current_fig = plt.gcf()

    plt.figure(fig.number)
    ax = plt.gca()
    con = ax.contour(mesh[0], mesh[1], led_model, levels=10, alpha=0.9)
    fig.colorbar(mappable=con, ax=ax)
    ax.scatter(model_params[0], model_params[1], color='Red')
    plt.text(0, window_radius * 2.2, f'Num. of Iterations: {fit_results.nit} -/- l2 + penalty: {fit_results.fun:.4}',
             ha='left')

    plt.figure(current_fig.number)


def show_img(img_id=-1, time=-1):
    if img_id == -1 and time == -1:
        print('Need Image ID or time to show an image')
        return
    if img_id != -1 and time != -1:
        print('Set either img_id or time to show the image')
        return
    path = get_img_path()
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


def show_led_diff(channel, led_id, time1, time2, window_radius=10):
    led1 = get_led_img(time1, led_id, window_radius)
    led2 = get_led_img(time2, led_id, window_radius)

    current_fig = plt.gcf()

    fig = plt.figure()
    ax = plt.gca()
    im = ax.imshow(np.array(led1, dtype='int16')[:, :, channel] - np.array(led2, dtype='int16')[:, :, channel])
    fig.colorbar(mappable=im, ax=ax)
    plt.title(f'LED diff between LED {led_id} at {time1} and {time2} seconds')

    plt.figure(current_fig.number)


def get_img_path():
    conf = CD()
    return conf['DEFAULT']['img_directory']


def get_led_pos(led_id):
    positions = led.load_file(".{}analysis{}led_search_areas.csv".format(sep, sep), delim=',', dtype=int)
    for i in range(positions.shape[0]):
        if positions[i, 0] == led_id:
            return float(positions[i, 1]), float(positions[i, 2])
    raise NameError("Could not find the positions of led {}.".format(led_id))


def fit_led(img_id, led_id, channel):
    ledsa = LEDSA(build_experiment_infos=False)
    ledsa.load_line_indices()
    ledsa.load_search_areas()
    ledsa.config['analyse_photo']['channel'] = str(channel)
    filename = led.get_img_name(img_id)
    fit_res = led.process_file(filename, ledsa.search_areas, ledsa.line_indices, ledsa.config['analyse_photo'], True,
                               led_id)
    return fit_res


def get_led_img(time, led_id, window_radius=10):
    filename = led.get_img_name(led.get_img_id_from_time(time))
    print(filename)
    path = get_img_path()
    led_im = Image.open(path + filename)

    x, y = get_led_pos(led_id)
    print(x, y)
    led_im = led_im.crop((y - window_radius,
                          x - window_radius,
                          y + window_radius,
                          x + window_radius))
    return led_im

