import os
from typing import List, Union

import numpy as np
import pandas as pd

import ledsa.core
from ledsa.core.ConfigData import ConfigData

sep = os.path.sep


def read_table(filename: str, delim=' ', dtype='float', atleast_2d=False, silent=False) -> np.ndarray:
    try:
        data = np.loadtxt(filename, delimiter=delim, dtype=dtype)
    except OSError as e:
        if not silent:
            print('An operation system error occurred while loading {}.'.format(filename),
                  'Maybe the file does not exist or there is no reading ',
                  'permission.\n Error Message: ', e)
        raise
    except Exception as e:
        if not silent:
            print('Some error has occurred while loading {}'.format(filename),
                  '. \n Error Message: ', e)
        raise
    else:
        if not silent:
            print('{} successfully loaded.'.format(filename))
    if atleast_2d:
        return np.atleast_2d(data)
    return np.atleast_1d(data)


def read_hdf(channel: int, path='.') -> pd.DataFrame:
    """
    Read the pandas dataframe binary at path. If binary does not exist, create it.
    :return: DataFrame with multi index 'img_id' and 'led_id'
    """
    try:
        fit_parameters = pd.read_hdf(f"{path}{sep}analysis{sep}channel{channel}{sep}all_parameters.h5", 'table')
    except FileNotFoundError:
        create_binary_data(channel)
        fit_parameters = pd.read_hdf(f"{path}{sep}analysis{sep}channel{channel}{sep}all_parameters.h5", 'table')
    fit_parameters.set_index(['img_id', 'led_id'], inplace=True)
    return fit_parameters


def extend_hdf(channel: int, quantity: str, values: np.ndarray, path='.') -> None:
    """
    Extends the binary
    """
    file = f"{path}{sep}analysis{sep}channel{channel}{sep}all_parameters.h5"
    fit_parameters = pd.read_hdf(file, 'table')
    fit_parameters[quantity] = values
    fit_parameters.to_hdf(file, 'table')


def create_binary_data(channel: int) -> None:
    """
    Creates binary file from all the #_led_positions.csv files generated in step 3
    """
    config = ConfigData()
    columns = _get_column_names(channel)

    fit_params = pd.DataFrame(columns=columns)

    # find time and fit parameter for every image

    if config['DEFAULT']['img_name_string'] == 'None':
        config.in_img_name_string()
        config.save()

    if config['analyse_photo']['first_img'] == 'None':
        config.in_first_img_analysis()
        config.save()
    first_img = int(config['analyse_photo']['first_img'])

    if config['analyse_photo']['last_img'] == 'None':
        config.in_last_img_analysis()
        config.save()
    last_img = int(config['analyse_photo']['last_img'])



    if config['DEFAULT']['img_number_overflow']:
        max_id = int(config['DEFAULT']['img_number_overflow'])
    else:
        max_id = 10**7
    number_of_images = (max_id + last_img - first_img) % max_id
    number_of_images //= int(config['analyse_photo']['skip_imgs']) + 1
    print('Loading fit parameters...')
    exception_counter = 0
    for image_id in range(1, number_of_images + 1):
        try:
            parameters = ledsa.core.file_handling.read_table(".{}analysis{}channel{}{}{}_led_positions.csv".format(
                sep, sep, channel, sep, image_id), delim=',', silent=True)
        except (FileNotFoundError, IOError):
            fit_params = fit_params.append(_param_array_to_dataframe([[np.nan] * (fit_params.shape[1] - 1)], image_id,
                                                                     columns),
                                           ignore_index=True, sort=False)
            exception_counter += 1
            continue

        parameters = parameters[parameters[:, 0].argsort()]     # sort for led_id
        parameters = _append_coordinates(parameters)
        fit_params = fit_params.append(_param_array_to_dataframe(parameters, image_id, columns),
                                       ignore_index=True, sort=False)

    print(f'{number_of_images - exception_counter} of {number_of_images} loaded.')
    # fit_params.set_index(['img_id', 'led_id'], inplace=True)
    fit_params.to_hdf(f".{sep}analysis{sep}channel{channel}{sep}all_parameters.h5", 'table', append=True)


def _get_column_names(channel: int) -> List[str]:
    parameters = ledsa.core.file_handling.read_table(f".{sep}analysis{sep}channel{channel}{sep}1_led_positions.csv",
                                                     delim=',', silent=True)
    columns = ["img_id", "led_id", "line",
               "sum_col_val", "mean_col_val", "max_col_val"]
    if parameters.shape[1] > len(columns):
        columns.extend(["led_center_x", "led_center_y"])
        columns.extend(["x", "y", "dx", "dy", "A", "alpha", "wx", "wy", "fit_success", "fit_fun", "fit_nfev"])
    if parameters.shape[1] != len(columns)-1:
        columns = _get_old_columns(parameters)
    columns.extend(["width", "height"])
    return columns


def _get_old_columns(params: np.ndarray) -> List[str]:
    """
    Includes file structures from older updates for legacy reasons
    """
    columns = []
    if params.shape[1] == 15:
        columns = ["img_id", "led_id", "line",
                   "x", "y", "dx", "dy", "A", "alpha", "wx", "wy", "fit_success", "fit_fun", "fit_nfev",
                   "sum_col_val", "mean_col_val"]
    if params.shape[1] == 4:
        columns = ["img_id", "led_id", "line",
                   "sum_col_val", "mean_col_val"]
    return columns


def _param_array_to_dataframe(array: Union[np.ndarray, List[List]], img_id: int, column_names: List[str]) -> pd.DataFrame:
    appended_array = np.empty((np.shape(array)[0], np.shape(array)[1] + 1))
    appended_array[:, 0] = img_id
    appended_array[:, 1:] = array
    fit_params = pd.DataFrame(appended_array, columns=column_names)
    return fit_params


def _append_coordinates(params: np.ndarray) -> np.ndarray:
    ac = _append_coordinates
    if "coord" not in ac.__dict__:
        try:
            ac.coord = ledsa.core.file_handling.read_table(".{}analysis{}led_search_areas_with_coordinates.csv".format(sep, sep),
                                                           delim=',', silent=True)[:, [0, -2, -1]]
        except (FileNotFoundError, IOError):
            ac.coord = False

    if type(ac.coord) == bool:
        return _append_nans(params)
    else:
        return _append_coordinates_to_params(params, ac.coord)


def _append_nans(params: np.ndarray) -> np.ndarray:
    p_with_nans = np.empty((np.shape(params)[0], np.shape(params)[1] + 2))
    p_with_nans[:] = np.NaN
    p_with_nans[:, :-2] = params
    return p_with_nans


def _append_coordinates_to_params(params: np.ndarray, coord: np.ndarray) -> np.ndarray:
    p_with_c = np.empty((np.shape(params)[0], np.shape(params)[1] + 2))
    p_with_c[:, :-2] = params

    if p_with_c.shape[0] != coord.shape[0]:
        mask = np.zeros(coord.shape)
        for led_id in p_with_c[:, 0]:
            mask = np.logical_or(mask, np.repeat((coord[:, 0] == led_id), coord.shape[1]).reshape(coord.shape))
        coord = np.reshape(coord[mask], (params.shape[0], coord.shape[1]))

    p_with_c[:, -2:] = coord[:, -2:]
    return p_with_c
