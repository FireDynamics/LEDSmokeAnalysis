from ..core.ConfigData import ConfigData
import numpy as np
import pandas as pd
from ..core import led_helper as led
import os

# os path separator
sep = os.path.sep


def normalize_parameter(parameter: str, channel: int) -> None:
    """Add a normalized version of parameter to the parameter binary file."""
    fit_parameters = read_hdf(channel)
    average = calculate_average_fitpar_without_smoke(parameter, channel)
    fit_parameters[f'normalized_{parameter}'] = fit_parameters[parameter].div(average)
    os.remove(f".{sep}analysis{sep}channel{channel}{sep}all_parameters.h5")
    fit_parameters.to_hdf(f".{sep}analysis{sep}channel{channel}{sep}all_parameters.h5", 'table')


def calculate_average_fitpar_without_smoke(parameter: str, channel: int, num_of_imgs=20) -> pd.Series:
    """Calculate the average of parameter over the first num_of_imgs images for every led."""
    fit_parameters = read_hdf(channel)
    idx = pd.IndexSlice
    fit_parameters = fit_parameters.loc[idx[1:num_of_imgs, :]]
    return fit_parameters[parameter].mean(0, level='led_id')


def create_binary_data(channel: int) -> None:
    """Compress all generated parameter csv files into a binary data object."""
    conf = ConfigData()
    columns = _get_column_names(channel)

    fit_params = pd.DataFrame(columns=columns)

    # find time and fit parameter for every image
    first_img = int(conf['analyse_photo']['first_img'])
    last_img = int(conf['analyse_photo']['last_img'])
    # TODO: add max img range to config
    number_of_images = (9999 + last_img - first_img) % 9999
    number_of_images //= int(conf['analyse_photo']['skip_imgs']) + 1
    print('Loading fit parameters...')
    exception_counter = 0
    for image_id in range(1, number_of_images + 1):
        try:
            parameters = led.load_file(".{}analysis{}channel{}{}{}_led_positions.csv".format(
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
    fit_params.to_hdf(f".{sep}analysis{sep}channel{channel}{sep}all_parameters.h5", 'table', append=True)


def clean_bin_data(channel):
    """Remove the binary parameter file for channel."""
    os.remove(f".{sep}analysis{sep}channel{channel}{sep}all_parameters.h5")


def _get_column_names(channel):
    parameters = led.load_file(f".{sep}analysis{sep}channel{channel}{sep}1_led_positions.csv", delim=',', silent=True)
    columns = ["img_id", "led_id", "line",
               "sum_col_val", "mean_col_val", "max_col_val"]
    if parameters.shape[1] > len(columns):
        columns.extend(["led_center_x", "led_center_y"])
        columns.extend(["x", "y", "dx", "dy", "A", "alpha", "wx", "wy", "fit_success", "fit_fun", "fit_nfev"])
    if parameters.shape[1] != len(columns)-1:
        columns = _get_old_columns(parameters)
    columns.extend(["width", "height"])
    return columns


def _get_old_columns(params):
    if params.shape[1] == 15:
        columns = ["img_id", "led_id", "line",
                   "x", "y", "dx", "dy", "A", "alpha", "wx", "wy", "fit_success", "fit_fun", "fit_nfev",
                   "sum_col_val", "mean_col_val"]
    if params.shape[1] == 4:
        columns = ["img_id", "led_id", "line",
                   "sum_col_val", "mean_col_val"]
    return columns

def _param_array_to_dataframe(array, img_id, column_names):
    appended_array = np.empty((np.shape(array)[0], np.shape(array)[1] + 1))
    appended_array[:, 0] = img_id
    appended_array[:, 1:] = array
    fit_params = pd.DataFrame(appended_array, columns=column_names)
    return fit_params


def _append_coordinates(params):
    ac = _append_coordinates
    if "coord" not in ac.__dict__:
        try:
            ac.coord = led.load_file(".{}analysis{}led_search_areas_with_coordinates.csv".format(sep, sep),
                                     delim=',', silent=True)[:, [0, -2, -1]]
        except (FileNotFoundError, IOError):
            ac.coord = False

    if type(ac.coord) == bool:
        return _append_nans(params)
    else:
        return _append_coordinates_to_params(params, ac.coord)


def _append_nans(params):
    p_with_nans = np.empty((np.shape(params)[0], np.shape(params)[1] + 2))
    p_with_nans[:] = np.NaN
    p_with_nans[:, :-2] = params
    return p_with_nans


def _append_coordinates_to_params(params, coord):
    p_with_c = np.empty((np.shape(params)[0], np.shape(params)[1] + 2))
    p_with_c[:, :-2] = params

    if p_with_c.shape[0] != coord.shape[0]:
        mask = np.zeros(coord.shape)
        for led_id in p_with_c[:, 0]:
            mask = np.logical_or(mask, np.repeat((coord[:, 0] == led_id), coord.shape[1]).reshape(coord.shape))
        coord = np.reshape(coord[mask], (params.shape[0], coord.shape[1]))

    p_with_c[:, -2:] = coord[:, -2:]
    return p_with_c


def read_hdf(channel: int, path='.') -> pd.DataFrame:
    """
    Read the parameter binary file or create it, if the file is not found.
    :returns DataFrame with all Parameters and multiindex ['img_id', 'led_id']
    """
    try:
        parameters = pd.read_hdf(f"{path}{sep}analysis{sep}channel{channel}{sep}all_parameters.h5", 'table')
    except FileNotFoundError:
        create_binary_data(channel)
        parameters = pd.read_hdf(f"{path}{sep}analysis{sep}channel{channel}{sep}all_parameters.h5", 'table')
    parameters.set_index(['img_id', 'led_id'], inplace=True)
    return parameters


def include_column_if_nonexistent(parameters: pd.DataFrame, parameter: str, channel: int) -> pd.DataFrame:
    """
    Check if paramerter is in the parameter DF and add it if it is not.
    Only parameters in the form of normalized_$PARAMETER can be added.
    :returns old or updated parameter DataFrame.
    """
    if parameter not in parameters.columns:
        if parameter.split('_')[0] == 'normalized':
            normalize_parameter(parameter.split('normalized_')[1], channel)
        else:
            raise Exception(f'Can not handle fit parameter: {parameter}')
        return read_hdf(channel)
    return parameters


def multiindex_series_to_nparray(multi_series: pd.Series) -> np.ndarray:
    """Convert pd multi-index series to numpy array."""
    index = multi_series.index
    print(index.levshape)
    print(index.shape)
    num_leds = pd.Series(multi_series.groupby(level=0).size()).iloc[0]
    num_imgs = pd.Series(multi_series.groupby(level=1).size()).iloc[0]
    array = np.zeros((num_imgs, num_leds))
    for i in range(num_imgs):
        array[i] = multi_series.loc[i+1]
    return array