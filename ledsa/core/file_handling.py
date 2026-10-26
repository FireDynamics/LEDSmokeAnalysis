import os
from typing import List, Union

import numpy as np
import pandas as pd

import ledsa.core
from ledsa.core.ConfigData import ConfigData
from ledsa.core.image_integrity import CORRUPTED_IMAGES_FILE_PATH


def set_flag(flag: str) -> None:
    """
    Creates a flag file in the current directory to indicate if a certain task has been performed.

    :param flag: The name of the flag to set.
    :type flag: str
    """
    with open(f'.{flag}.flag', 'w', encoding='utf-8') as flag_file:
        pass


def check_flag(flag: str) -> bool:
    """
    Checks if a flag file exists in the current directory.

    :param flag: The name of the flag to check.
    :type flag: str
    :return: True if the flag file exists, False otherwise.
    :rtype: bool
    """
    return os.path.exists(f'.{flag}.flag')


def remove_flag(flag: str) -> None:
    """
    Removes a flag file from the current directory if it exists.

    :param flag: The name of the flag to remove.
    :type flag: str
    :raises OSError: If an error occurs during file removal. This exception is caught and suppressed within the function.
    """
    try:
        os.remove(f'.{flag}.flag')
    except OSError:
        pass

def create_analysis_infos_avg():  # TODO: Move funtion somewhere else
    """
    Generate CSV files with image information for experiment and/or analysis averaged over n images from the existing
    image_infos_analysis.csv file. Skips the first n images. Contains image name, exif time and experiment time
    """
    n_summarize = 2 # TODO: remove hardcoding
    n_skip_images = 10 # TODO: Remove hardcoding
    in_file_path = os.path.join('analysis', 'image_infos_analysis.csv')
    image_infos = pd.read_csv(in_file_path)
    img_names = image_infos['Name'].tolist()
    exp_time = image_infos['Experiment_Time[s]']
    head_img_names = img_names[:n_skip_images]
    head_exp_times = exp_time[:n_skip_images]
    tail_img_names = img_names[n_skip_images:]
    tail_exp_times = exp_time[n_skip_images:]
    img_names_avg = ["/".join(tail_img_names[n:n + n_summarize]) for n in range(0, len(tail_img_names), n_summarize)]
    exp_times_avg = tail_exp_times.groupby(np.arange(len(tail_exp_times)) // n_summarize).mean()
    combined_exp_times = pd.concat([head_exp_times, exp_times_avg])
    combined_exp_times.index = range(len(combined_exp_times))
    combined_img_names = head_img_names + img_names_avg
    ids = range(1, len(combined_img_names) + 1)
    img_infos_analysis_avg = pd.DataFrame(
        {"#ID": ids, "Name": combined_img_names, "Experiment_Time[s]": combined_exp_times})
    img_infos_analysis_avg.reset_index()
    out_file_path = os.path.join('analysis', 'image_infos_analysis_avg.csv')
    img_infos_analysis_avg.to_csv(out_file_path)

def read_table(filename: str, delim=' ', dtype='float', atleast_2d=False, silent=False) -> np.ndarray:
    """
    Reads data from a file into a numpy array.

    :param filename: Name of the file to read.
    :type filename: str
    :param delim: Delimiter for the file, defaults to space.
    :type delim: str
    :param dtype: Desired data-type for the array.
    :type dtype: Union[str, type]
    :param atleast_2d: If True, always returns data as at least a 2D array.
    :type atleast_2d: bool
    :param silent: If False, prints error messages to console.
    :type silent: bool
    :return: Data read from the file as a numpy array.
    :rtype: numpy.ndarray

    :raises OSError: If an operating system error occurs, such as a missing file.
    :raises Exception: For any other error that occurs.
    """

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
    Reads data from an HDF file for a given channel. If the binary does not exist, it is created.

    :param channel: Channel number for which data is to be read.
    :type channel: int
    :param path: Directory path where the HDF is stored, defaults to the current directory.
    :type path: str
    :return: DataFrame with multi-index 'img_id' and 'led_id'.
    :rtype: pd.DataFrame

    :raises FileNotFoundError: If the HDF file is not found.
    """
    file_path = os.path.join(path, 'analysis', f'channel{channel}', 'all_parameters.h5')
    try:
        # Try reading with 'channel_values' key first
        fit_parameters = pd.read_hdf(file_path, key='channel_values')
    except (FileNotFoundError, KeyError):
        try:
            # Try reading with '/table' key as fallback
            fit_parameters = pd.read_hdf(file_path, key='/table')
        except (FileNotFoundError, KeyError):
            # If file doesn't exist or neither key works, create new binary data
            create_binary_data(channel, path=path)
            fit_parameters = pd.read_hdf(file_path, key='channel_values')

    fit_parameters.set_index(['img_id', 'led_id'], inplace=True)
    return fit_parameters


def read_hdf_avg(channel: int, path='.') -> pd.DataFrame:
    """
    Reads averaged data from an HDF file for a given channel. If binary does not exist, it's created.

    :param channel: Channel number for which data is to be read.
    :type channel: int
    :param path: Directory path where the HDF is stored, defaults to the current directory.
    :type path: str
    :return: DataFrame with multi-index 'img_id' and 'led_id'.
    :rtype: pd.DataFrame

    :raises FileNotFoundError: If the HDF file is not found.
    """
    file_path = os.path.join(path, 'analysis', f'channel{channel}','all_parameters.h5',)
    try:
        fit_parameters = pd.read_hdf(file_path, key='channel_values')
    except FileNotFoundError:
        average_all_fitpar(channel)
        fit_parameters = pd.read_hdf(file_path, key='channel_values')
    fit_parameters.set_index(['img_id', 'led_id'], inplace=True)
    return fit_parameters


def average_all_fitpar(channel, n_summarize=2, num_ref_imgs=10) -> None:  # TODO: rename variables within function
    """
    Averages all fit parameters for a given channel and writes them to an HDF file.

    :param channel: Channel number for which data is to be averaged.
    :type channel: int
    :param n_summarize: Number of images to average over.
    :type n_summarize: int
    :param num_ref_imgs: Number of reference images (Excluded from averaging).
    :type num_ref_imgs: int
    """
    fit_parameters = read_hdf(channel)
    fit_parameters_grouped = fit_parameters.groupby(['led_array_id', 'led_id'])
    avg_dataset_list = []
    for name, dataset in fit_parameters_grouped:
        head = dataset.iloc[:num_ref_imgs]
        tail = dataset.iloc[num_ref_imgs:]
        avg_dataset = tail.groupby(np.arange(len(tail)) // n_summarize).mean()
        avg_dataset["img_id"] = range(num_ref_imgs + 1, num_ref_imgs + len(avg_dataset) + 1)
        avg_dataset["led_id"] = name[1]
        avg_dataset.set_index(['img_id', 'led_id'], inplace=True)
        combined_dataset = pd.concat([head, avg_dataset])
        avg_dataset_list.append(combined_dataset)
    all_fitpar = pd.concat(avg_dataset_list)
    all_fitpar[["led_array_id", "sum_col_val"]] = all_fitpar[["led_array_id", "sum_col_val"]].astype(int)
    try: # TODO: Resolve
        all_fitpar[["sum_col_val_cc"]] = all_fitpar[["sum_col_val_cc"]].astype(int)
    except:
        pass
    all_fitpar.reset_index(inplace=True)
    file_path = os.path.join('analysis', f'channel{channel}', 'all_parameters_avg.h5'),
    all_fitpar.to_hdf(file_path, key='channel_values', format='table', append=True)


def extend_hdf(channel: int, quantity: str, values: np.ndarray) -> None:
    """
    Extends existing binary HDF file by adding new data columns.

    :param channel: Channel number for which the HDF is to be read extended.
    :type channel: int
    :param quantity: New column name to add.
    :type quantity: str
    :param values: Data values for the new column.
    :type values:  np.ndarray
    """
    file = os.path.join('analysis', f'channel{channel}', 'all_parameters.h5')
    fit_parameters = pd.read_hdf(file, key='channel_values')
    fit_parameters[quantity] = values
    fit_parameters.to_hdf(file, key='channel_values', format='table')


def create_binary_data(channel: int, path: str = '.') -> None:
    """
    Creates binary file from the CSV files for a specified channel and writes to an HDF file.

    :param channel: Channel number for which binary data is to be created.
    :type channel: int
    :param path: Root directory of the LEDSA simulation. Defaults to the current directory.
    :type path: str
    :raises ValueError: If not a single image of the channel has analysis results.
    """
    prev_dir = os.getcwd()
    try:
        os.chdir(path)
        config = ConfigData()
        columns = _get_column_names(channel)

        fit_params_list = []

        # find time and fit parameter for every image

        first_img_id = int(config['analyse_photo']['first_img_analysis_id'])
        last_img_id = int(config['analyse_photo']['last_img_analysis_id'])

        if config['DEFAULT']['num_img_overflow'] != 'None':
            max_id = int(config['DEFAULT']['num_img_overflow'])
        else:
            max_id = 10 ** 7
        number_of_images = (max_id + last_img_id - first_img_id) % max_id + 1
        number_of_images //= int(config['analyse_photo']['num_skip_imgs']) + 1
        print('Loading fit parameters...')
        skipped_img_ids = []
        for image_id in range(1, number_of_images + 1):
            try:
                in_file_path = os.path.join('analysis', f'channel{channel}', f'{image_id}_led_positions.csv')
                parameters = ledsa.core.file_handling.read_table(in_file_path, delim=',', silent=True)
            except (FileNotFoundError, IOError):
                # A missing result file means the image was not analysed, for example because it could not be
                # read. The image is left out entirely instead of being filled with NaNs, so that the
                # extinction coefficients are never computed from them.
                skipped_img_ids.append(image_id)
                continue

            parameters = parameters[parameters[:, 0].argsort()]  # sort for led_id
            parameters = _append_coordinates(parameters)
            fit_params_fragment = _param_array_to_dataframe(parameters, image_id, columns)
            fit_params_list.append(fit_params_fragment)

        if not fit_params_list:
            raise ValueError(f'None of the {number_of_images} images of channel {channel} has analysis '
                             f'results. Run the data extraction (-s3 or -s3_fast) before the analysis.')
        fit_params = pd.concat(fit_params_list, ignore_index=True, sort=False)

        print(f'{number_of_images - len(skipped_img_ids)} of {number_of_images} loaded.')
        if skipped_img_ids:
            print(f'No analysis results for the image IDs {skipped_img_ids}, so they are left out of the '
                  f'extinction coefficients. Images that could not be read are listed in '
                  f'{CORRUPTED_IMAGES_FILE_PATH}.')
        fit_params['img_id'] = fit_params['img_id'].astype(int)
        fit_params['led_id'] = fit_params['led_id'].astype(int)
        fit_params['led_array_id'] = fit_params['led_array_id'].astype(int)
        fit_params['max_col_val'] = fit_params['max_col_val'].astype(int)
        fit_params['sum_col_val'] = fit_params['sum_col_val'].astype(int)
        out_file_path = os.path.join('analysis', f'channel{channel}', 'all_parameters.h5')
        fit_params.to_hdf(out_file_path, key='channel_values', format='table', append=True)
    finally:
        os.chdir(prev_dir)

# Column names of the step 3 result files (<img_id>_led_positions.csv). The files do not contain img_id, it is
# prepended when they are loaded, so a file with n columns has n + 1 names.
BASE_COLUMNS = ["img_id", "led_id", "led_array_id", "sum_col_val", "mean_col_val", "max_col_val"]
BGSUB_COLUMNS = ["bgsub_sum_col_val"]
SIDEBG_COLUMNS = ["sidebg_sum_col_val", "sidebg_pedestal", "bgsub2_sum_col_val"]
LED_CENTER_COLUMNS = ["led_center_x", "led_center_y"]
FIT_COLUMNS = ["x", "y", "dx", "dy", "A", "alpha", "wx", "wy", "fit_success", "fit_fun", "fit_nfev", "fit_time"]

# known layouts by the number of names (including img_id)
COLUMN_LAYOUTS = {
    len(BASE_COLUMNS): BASE_COLUMNS,  # -s3_fast, before the local background subtraction
    len(BASE_COLUMNS + BGSUB_COLUMNS): BASE_COLUMNS + BGSUB_COLUMNS,  # -s3_fast, before the side band pedestal
    len(BASE_COLUMNS + BGSUB_COLUMNS + SIDEBG_COLUMNS): BASE_COLUMNS + BGSUB_COLUMNS + SIDEBG_COLUMNS,  # -s3_fast
    len(BASE_COLUMNS + BGSUB_COLUMNS + LED_CENTER_COLUMNS + FIT_COLUMNS):
        BASE_COLUMNS + BGSUB_COLUMNS + LED_CENTER_COLUMNS + FIT_COLUMNS,  # -s3 with fit, before the side band pedestal
    len(BASE_COLUMNS + BGSUB_COLUMNS + SIDEBG_COLUMNS + LED_CENTER_COLUMNS + FIT_COLUMNS):
        BASE_COLUMNS + BGSUB_COLUMNS + SIDEBG_COLUMNS + LED_CENTER_COLUMNS + FIT_COLUMNS,  # -s3 with fit
}
# Fitted files written before the delimiter between the main and the fit part was added: max_col_value and
# led_center_x are glued into one field, so the file has one column less and the names are shifted from there on.
LEGACY_FIT_COLUMNS = BASE_COLUMNS + LED_CENTER_COLUMNS + FIT_COLUMNS[:-1]


def _get_column_names(channel: int) -> List[str]:
    """
    Get the column names for the specified channel from the number of columns of the CSV files.

    :param channel: Channel number for which column names are to be determined.
    :type channel: int
    :return: List of column names.
    :rtype: List[str]
    :raises ValueError: If the file cannot be parsed or the number of columns matches no known layout.
    """
    file_path = os.path.join('analysis', f'channel{channel}', '1_led_positions.csv')
    try:
        parameters = ledsa.core.file_handling.read_table(file_path, delim=',', silent=True, atleast_2d=True)
    except ValueError as error:
        # fitted files written with bgsub_sum_col_value but before the delimiter fix glue two floats into one field
        raise ValueError(f"{file_path} cannot be read ({error}). If it was written with fits by a LEDSA version "
                         f"that had bgsub_sum_col_value but no delimiter between the main and the fit columns, "
                         f"bgsub_sum_col_value and led_center_x are glued into one field. Run step 3 again.") from error
    num_names = parameters.shape[1] + 1  # + 1 for img_id, which is prepended on load
    if num_names in COLUMN_LAYOUTS:
        columns = list(COLUMN_LAYOUTS[num_names])
    elif num_names == len(LEGACY_FIT_COLUMNS):
        print(f"Warning: {file_path} was written with fits by an older LEDSA version, in which max_col_value and "
              f"led_center_x are glued into one field. sum_col_val and mean_col_val are valid, max_col_val and all "
              f"fit columns are not. Run step 3 again to obtain valid fit results.")
        columns = list(LEGACY_FIT_COLUMNS)
    else:
        columns = _get_old_columns(parameters)
        if not columns:
            raise ValueError(f"{file_path} has {parameters.shape[1]} columns, which matches no known layout of the "
                             f"step 3 result files.")
    columns.extend(["width", "height"])
    return columns


def _get_old_columns(params: np.ndarray) -> List[str]:
    """
    Provides column names for older file structures for backward compatibility.

    :param params: Array containing parameter data to infer column structure.
    :type params: np.ndarray
    :return: List of old column names based on the provided parameter data.
    :rtype: List[str]
    """
    columns = []
    if params.shape[1] == 15:
        columns = ["img_id", "led_id", "led_array_id",
                   "x", "y", "dx", "dy", "A", "alpha", "wx", "wy", "fit_success", "fit_fun", "fit_nfev",
                   "sum_col_val", "mean_col_val"]
    if params.shape[1] == 4:
        columns = ["img_id", "led_id", "led_array_id",
                   "sum_col_val", "mean_col_val"]
    return columns


def _param_array_to_dataframe(array: Union[np.ndarray, List[List]], img_id: int,
                              column_names: List[str]) -> pd.DataFrame:
    """
    Convert an array of parameters to a pandas DataFrame.

    :param array: Array of parameters.
    :type array: Union[np.ndarray, List[List]]
    :param img_id: Image ID to append to each row.
    :type img_id: int
    :param column_names: List of column names for the DataFrame.
    :type column_names: List[str]
    :return: DataFrame representation of the provided array.
    :rtype: pd.DataFrame
    """
    appended_array = np.empty((np.shape(array)[0], np.shape(array)[1] + 1))
    appended_array[:, 0] = img_id
    appended_array[:, 1:] = array
    fit_params = pd.DataFrame(appended_array, columns=column_names)
    return fit_params


def _append_coordinates(params: np.ndarray) -> np.ndarray:
    """
    Append LED coordinates to the parameter array if available.

    :param params: Array of parameters.
    :type params: np.ndarray
    :return: Updated parameter array with LED coordinates appended.
    :rtype: np.ndarray

    :raised: FileNotFoundError if led_search_areas_with_coordinates.csv' file is not found.
    """
    ac = _append_coordinates
    if "coord" not in ac.__dict__:
        try:
            file_path =  os.path.join('analysis', 'led_search_areas_with_coordinates.csv')
            ac.coord = ledsa.core.file_handling.read_table(file_path, delim=',', silent=True)[:, [0, -2, -1]]
        except (FileNotFoundError, IOError):
            ac.coord = False

    if type(ac.coord) == bool:
        return _append_nans_to_params(params)
    else:
        return _append_coordinates_to_params(params, ac.coord)


def _append_nans_to_params(params: np.ndarray) -> np.ndarray:
    """
    Append NaN values to the parameter array.

    :param params: Array of parameters.
    :type params: np.ndarray
    :return: Updated parameter array with NaN values appended.
    :rtype: np.ndarray
    """
    p_with_nans = np.empty((np.shape(params)[0], np.shape(params)[1] + 2))
    p_with_nans[:] = np.NaN
    p_with_nans[:, :-2] = params
    return p_with_nans


def _append_coordinates_to_params(params: np.ndarray, coord: np.ndarray) -> np.ndarray:
    """
    Append coordinates to the parameter array.

    :param params: Array of parameters.
    :type params: np.ndarray
    :param coordinates: Array of LED coordinates.
    :type coordinates: np.ndarray
    :return: Updated parameter array with coordinates appended.
    :rtype: np.ndarray
    """
    p_with_c = np.empty((np.shape(params)[0], np.shape(params)[1] + 2))
    p_with_c[:, :-2] = params

    if p_with_c.shape[0] != coord.shape[0]:
        mask = np.zeros(coord.shape)
        for led_id in p_with_c[:, 0]:
            mask = np.logical_or(mask, np.repeat((coord[:, 0] == led_id), coord.shape[1]).reshape(coord.shape))
        coord = np.reshape(coord[mask], (params.shape[0], coord.shape[1]))

    p_with_c[:, -2:] = coord[:, -2:]
    return p_with_c