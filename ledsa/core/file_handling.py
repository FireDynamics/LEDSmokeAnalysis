import os
import re
import warnings
from pathlib import Path
from typing import Dict, List, Union

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

def get_extinction_coefficients_dir(solver: str, reference_property: str,
                                   root: Union[str, Path] = '.') -> Path:
    """
    Build the directory the extinction coefficients of one solver and one quantity are stored in.

    Every analysed quantity gets a directory of its own, so that the results of an analysis run
    with several quantities stay apart from each other.

    :param solver: Solver the coefficients were computed with.
    :type solver: str
    :param reference_property: Quantity the coefficients were computed from.
    :type reference_property: str
    :param root: Root directory of the LEDSA simulation, defaults to the current directory.
    :type root: Union[str, Path]
    :return: Directory of the extinction coefficient files.
    :rtype: Path
    """
    return Path(root) / 'analysis' / 'extinction_coefficients' / str(solver) / str(reference_property)


def get_extinction_coefficients_file_path(solver: str, channel: int, reference_property: str, led_array: int,
                                          root: Union[str, Path] = '.') -> Path:
    """
    Build the path of the extinction coefficients of one solver, quantity, channel and LED array.

    :param solver: Solver the coefficients were computed with.
    :type solver: str
    :param channel: Camera channel the coefficients were computed from.
    :type channel: int
    :param reference_property: Quantity the coefficients were computed from.
    :type reference_property: str
    :param led_array: LED array the coefficients belong to.
    :type led_array: int
    :param root: Root directory of the LEDSA simulation, defaults to the current directory.
    :type root: Union[str, Path]
    :return: Path of the extinction coefficient file.
    :rtype: Path
    """
    file_name = (f'extinction_coefficients_{solver}_channel_{channel}_{reference_property}'
                 f'_led_array_{led_array}.csv')
    return get_extinction_coefficients_dir(solver, reference_property, root) / file_name


# Number of consecutive images that are summarized into one when average_images is set in
# config_analysis.ini. The option is documented as the average of two consecutive images.
NUM_IMGS_PER_AVERAGE = 2
# Number of reference images at the beginning of an experiment that are left unaveraged
DEFAULT_NUM_REF_IMGS = 10


def build_avg_img_groups(img_ids: List[int], n_summarize: int = NUM_IMGS_PER_AVERAGE,
                         num_ref_imgs: int = DEFAULT_NUM_REF_IMGS,
                         ref_img_ids: List[int] = None) -> List[List[int]]:
    """
    Group the analysed images into the images of the averaged data set.

    Every reference image forms a group of its own and is therefore kept as it is, while the
    images between them are summarized in groups of n_summarize. A group never reaches across a
    reference image, so the reference images keep the place in the sequence they were taken in,
    whether they lie at the beginning of the experiment or anywhere else.

    :param img_ids: IDs of the analysed images, in the order they were taken.
    :type img_ids: List[int]
    :param n_summarize: Number of consecutive images that are summarized into one.
    :type n_summarize: int
    :param num_ref_imgs: Number of reference images at the beginning of the experiment, used when
        ref_img_ids is None.
    :type num_ref_imgs: int
    :param ref_img_ids: IDs of the images the LED intensities are normalized with, which are kept
        unaveraged. None: the first num_ref_imgs images are kept.
    :type ref_img_ids: List[int] or None
    :return: IDs of the images every averaged image is computed from, in the order they were taken.
    :rtype: List[List[int]]
    """
    kept_img_ids = set(ref_img_ids) if ref_img_ids is not None else set(img_ids[:num_ref_imgs])
    groups = []
    summarized = []
    for img_id in img_ids:
        if img_id in kept_img_ids:
            if summarized:
                groups.append(summarized)
                summarized = []
            groups.append([img_id])
            continue
        summarized.append(img_id)
        if len(summarized) == n_summarize:
            groups.append(summarized)
            summarized = []
    if summarized:
        groups.append(summarized)
    return groups


def avg_img_ids_of_ref_imgs(ref_img_ids: List[int], img_ids: List[int],
                            n_summarize: int = NUM_IMGS_PER_AVERAGE) -> List[int]:
    """
    Translate the IDs of the reference images into the IDs they carry in the averaged data set.

    The averaged images are numbered from 1 again, so the IDs the configuration names would point
    at other images after the averaging. The reference images themselves are kept unaveraged, so
    each of them is still an image of its own and only its number changes.

    :param ref_img_ids: IDs of the reference images among the analysed images.
    :type ref_img_ids: List[int]
    :param img_ids: IDs of the analysed images, in the order they were taken.
    :type img_ids: List[int]
    :param n_summarize: Number of consecutive images that are summarized into one.
    :type n_summarize: int
    :return: IDs of the reference images in the averaged data set.
    :rtype: List[int]
    :raises ValueError: If none of the reference images is among the analysed images.
    """
    groups = build_avg_img_groups(img_ids, n_summarize=n_summarize, ref_img_ids=ref_img_ids)
    requested = set(ref_img_ids)
    avg_img_id_of = {group[0]: avg_img_id for avg_img_id, group in enumerate(groups, start=1)
                     if len(group) == 1 and group[0] in requested}
    missing = [img_id for img_id in ref_img_ids if img_id not in avg_img_id_of]
    if missing:
        msg = (f'The reference images {missing} are not among the analysed images, so they cannot '
               f'be kept out of the averaging.')
        if len(missing) == len(ref_img_ids):
            raise ValueError(f'{msg} Set ref_img_indices in config_analysis.ini or '
                             f'ref_img_experiment_ids in config.ini to images that were analysed.')
        warnings.warn(f'{msg} The remaining ones are used.', category=UserWarning, stacklevel=2)
    return [avg_img_id_of[img_id] for img_id in ref_img_ids if img_id in avg_img_id_of]


def _averaging_parameters(n_summarize: int, num_ref_imgs: int, ref_img_ids: List[int]) -> dict:
    """Parameters the averaged data were computed with, kept next to them in the binary."""
    named = ref_img_ids is not None
    return {'n_summarize': n_summarize,
            # the number of reference images is only read when none of them is named
            'num_ref_imgs': None if named else num_ref_imgs,
            'ref_img_ids': sorted(ref_img_ids) if named else None}


def _averaged_binary_was_built_differently(file_path: Union[str, Path], n_summarize: int,
                                           num_ref_imgs: int, ref_img_ids: List[int]) -> bool:
    """
    Check whether the averaged binary was computed with other reference images.

    A binary of an earlier run is newer than the data it was computed from, so the modification
    times alone would hand out averages of other reference images after a change of the
    configuration. A binary written before the parameters were kept with the data counts as
    different and is computed again.
    """
    try:
        with pd.HDFStore(file_path, mode='r') as store:
            stored_parameters = getattr(store.get_storer('channel_values').attrs,
                                        'averaging_parameters', None)
    except (AttributeError, KeyError, OSError):
        return True
    return stored_parameters != _averaging_parameters(n_summarize, num_ref_imgs, ref_img_ids)


def get_binary_file_path(channel: int, path: Union[str, Path] = '.', averaged: bool = False) -> str:
    """
    Build the path of the binary data of one channel.

    The averaged binary lies next to the one of the single images it is computed from, so that an
    analysis with average_images set never reads the unaveraged data by accident.

    :param channel: Camera channel the binary data belong to.
    :type channel: int
    :param path: Root directory of the LEDSA simulation, defaults to the current directory.
    :type path: Union[str, Path]
    :param averaged: If True, the path of the averaged binary is built.
    :type averaged: bool
    :return: Path of the binary file.
    :rtype: str
    """
    file_name = 'all_parameters_avg.h5' if averaged else 'all_parameters.h5'
    return os.path.join(path, 'analysis', f'channel{channel}', file_name)


def read_analysis_img_ids(path: Union[str, Path] = '.') -> List[int]:
    """
    Read the analysis IDs of the analysed images, in the order they were taken.

    :param path: Root directory of the LEDSA simulation, defaults to the current directory.
    :type path: Union[str, Path]
    :return: Analysis IDs of the images the data extraction worked on.
    :rtype: List[int]
    """
    file_path = os.path.join(path, 'analysis', 'image_infos_analysis.csv')
    return pd.read_csv(file_path)['#ID'].tolist()


def create_analysis_infos_avg(n_summarize: int = NUM_IMGS_PER_AVERAGE,
                              num_ref_imgs: int = DEFAULT_NUM_REF_IMGS,
                              ref_img_ids: List[int] = None,
                              path: Union[str, Path] = '.') -> None:  # TODO: Move funtion somewhere else
    """
    Generate a CSV file with the image information of the averaged images from the existing
    image_infos_analysis.csv file. The reference images are kept as they are, the remaining ones
    are summarized in groups of n_summarize. Contains image name and experiment time.

    The averaged images are numbered from 1 again. Called with the same n_summarize, num_ref_imgs
    and ref_img_ids as :func:`average_all_fitpar`, which averages the intensities of the very same
    images, both files hold the same image IDs and the same number of images.

    :param n_summarize: Number of consecutive images that are summarized into one.
    :type n_summarize: int
    :param num_ref_imgs: Number of reference images at the beginning of the experiment, used when
        ref_img_ids is None.
    :type num_ref_imgs: int
    :param ref_img_ids: IDs of the images the LED intensities are normalized with, which are not
        averaged. None: the first num_ref_imgs images are kept.
    :type ref_img_ids: List[int] or None
    :param path: Root directory of the LEDSA simulation, defaults to the current directory.
    :type path: Union[str, Path]
    """
    in_file_path = os.path.join(path, 'analysis', 'image_infos_analysis.csv')
    image_infos = pd.read_csv(in_file_path)
    img_names = image_infos['Name'].tolist()
    exp_time = image_infos['Experiment_Time[s]']
    img_ids = image_infos['#ID'].tolist()
    groups = build_avg_img_groups(img_ids, n_summarize=n_summarize, num_ref_imgs=num_ref_imgs,
                                  ref_img_ids=ref_img_ids)
    # the images of a group are the rows the group is built from, so the names and the times are
    # taken by position and a group of one image keeps the name and the time of that image
    row_of_img_id = {img_id: row for row, img_id in enumerate(img_ids)}
    rows_per_group = [[row_of_img_id[img_id] for img_id in group] for group in groups]
    combined_img_names = ["/".join(img_names[row] for row in rows) for rows in rows_per_group]
    combined_exp_times = pd.Series(
        [exp_time.iloc[rows].mean() for rows in rows_per_group], index=range(len(groups)))
    ids = range(1, len(combined_img_names) + 1)
    img_infos_analysis_avg = pd.DataFrame(
        {"#ID": ids, "Name": combined_img_names, "Experiment_Time[s]": combined_exp_times})
    out_file_path = os.path.join(path, 'analysis', 'image_infos_analysis_avg.csv')
    # written without the row index, so that the file has the same columns as the one of the
    # single images it is derived from
    img_infos_analysis_avg.to_csv(out_file_path, index=False)


def read_experiment_times(path: Union[str, os.PathLike] = '.', average_images: bool = False) -> Dict[int, float]:
    """
    Read the experiment time of every image of a LEDSA run from its image infos.

    :param path: Root directory of the LEDSA run, defaults to the current directory.
    :type path: str or os.PathLike
    :param average_images: Whether the image infos of the averaged images are read.
    :type average_images: bool
    :return: Experiment time [s] of every image ID.
    :rtype: dict[int, float]
    :raises FileNotFoundError: If the image infos file does not exist.
    """
    info_file = 'image_infos_analysis_avg.csv' if average_images else 'image_infos_analysis.csv'
    file_path = os.path.join(path, 'analysis', info_file)
    image_infos = pd.read_csv(file_path)
    return {int(img_id): float(experiment_time)
            for img_id, experiment_time in zip(image_infos['#ID'], image_infos['Experiment_Time[s]'])}


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


def _binary_data_is_outdated(binary_file_path: str, channel: int, path: str = '.') -> bool:
    """
    Tell whether the binary data of a channel are older than one of the files they are built from.

    The binary data hold the step 3 result files of a channel together with the LED coordinates, cut to
    the image range of config.ini, so all three are sources. A source that is newer than the binary data
    means they were built before it and no longer describe it.

    Only the modification times are compared. A source rewritten within the same clock tick as the binary
    data is therefore not recognised, and a change to config.ini that does not concern the image range
    causes a rebuild that was not needed. The rebuild costs the loading time of the result files, while a
    missed source would be answered with the values of the earlier run.

    :param binary_file_path: Path of the binary HDF file of the channel.
    :type binary_file_path: str
    :param channel: Channel number whose result files are compared.
    :type channel: int
    :param path: Root directory of the LEDSA simulation. Defaults to the current directory.
    :type path: str
    :return: True if the binary data exist and one of their sources is newer.
    :rtype: bool
    """
    if not os.path.exists(binary_file_path):
        # Nothing to compare against. The caller builds the binary data because they are missing.
        return False
    binary_mtime = os.path.getmtime(binary_file_path)
    source_paths = [os.path.join(path, 'config.ini'),
                    os.path.join(path, 'analysis', 'led_search_areas_with_coordinates.csv')]
    result_dir_path = os.path.join(path, 'analysis', f'channel{channel}')
    if os.path.isdir(result_dir_path):
        source_paths.extend(os.path.join(result_dir_path, file_name)
                            for file_name in os.listdir(result_dir_path)
                            if re.search(r"\d+_led_positions\.csv", file_name) is not None)
    for source_path in source_paths:
        try:
            if os.path.getmtime(source_path) > binary_mtime:
                return True
        except OSError:
            # A source that is not there cannot have outdated the binary data.
            continue
    return False


def read_hdf(channel: int, path='.') -> pd.DataFrame:
    """
    Reads data from an HDF file for a given channel. If the binary does not exist or is older than the
    files it is built from, it is created.

    :param channel: Channel number for which data is to be read.
    :type channel: int
    :param path: Directory path where the HDF is stored, defaults to the current directory.
    :type path: str
    :return: DataFrame with multi-index 'img_id' and 'led_id'.
    :rtype: pd.DataFrame

    :raises FileNotFoundError: If the HDF file is not found.
    """
    file_path = get_binary_file_path(channel, path=path)
    if _binary_data_is_outdated(file_path, channel, path=path):
        # The binary data are only a cache of the step 3 result files. Without this they were read as
        # they were, so an analysis after the data extraction had run again silently reported the
        # extinction coefficients of the values of the earlier run.
        print(f'{file_path} is older than the files it is built from and is rebuilt.')
        create_binary_data(channel, path=path)
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


def read_hdf_avg(channel: int, path='.', n_summarize: int = NUM_IMGS_PER_AVERAGE,
                 num_ref_imgs: int = DEFAULT_NUM_REF_IMGS,
                 ref_img_ids: List[int] = None) -> pd.DataFrame:
    """
    Reads averaged data from an HDF file for a given channel. If the averaged binary is not up to
    date, it is created from the binary of the single images.

    The averaged binary has a file name of its own, so that the averaged data are returned
    whenever they are asked for. It is rebuilt whenever it is missing, older than the binary of the
    single images it is computed from, older than the step 3 result files those are built from, or
    older than config_analysis.ini, which holds the counts the images are summarized with. So an
    analysis never reads the averaged data of an earlier extraction run or of other counts.

    :param channel: Channel number for which data is to be read.
    :type channel: int
    :param path: Directory path where the HDF is stored, defaults to the current directory.
    :type path: str
    :param n_summarize: Number of consecutive images that are averaged into one.
    :type n_summarize: int
    :param num_ref_imgs: Number of reference images at the beginning of the experiment, used when
        ref_img_ids is None.
    :type num_ref_imgs: int
    :param ref_img_ids: IDs of the images the LED intensities are normalized with, which are not
        averaged. None: the first num_ref_imgs images are kept.
    :type ref_img_ids: List[int] or None
    :return: DataFrame with multi-index 'img_id' and 'led_id'.
    :rtype: pd.DataFrame
    """
    file_path = get_binary_file_path(channel, path=path, averaged=True)
    if not os.path.exists(file_path):
        outdated = True
    else:
        # The averaged data are built from the binary of the single images, which is itself only a
        # cache of the step 3 result files, so both are sources. Comparing the averaged data with
        # the single image binary alone would miss a data extraction run that has not been read
        # back yet, since that binary is rebuilt only when it is read. config_analysis.ini is a
        # source as well, because num_ref_images and n_summarize decide which images are summarized
        # into which, so averages computed with other counts no longer answer what it asks for. A
        # save of the config that leaves both counts alone costs a rebuild that was not needed,
        # which is the loading time of the channel.
        sources = [get_binary_file_path(channel, path=path),
                   os.path.join(path, 'config_analysis.ini')]
        # The reference images can also be named in config.ini or, for the stacked analysis, in
        # config_stacked.ini, which does not lie next to the data at all, so the parameters the
        # averages were computed with are compared with the ones they are asked for as well.
        outdated = _binary_data_is_outdated(file_path, channel, path=path) or any(
            os.path.exists(source) and os.path.getmtime(file_path) < os.path.getmtime(source)
            for source in sources) or _averaged_binary_was_built_differently(
                file_path, n_summarize, num_ref_imgs, ref_img_ids)
    if outdated:
        average_all_fitpar(channel, n_summarize=n_summarize, num_ref_imgs=num_ref_imgs,
                           ref_img_ids=ref_img_ids, path=path)
    try:
        fit_parameters = pd.read_hdf(file_path, key='channel_values')
    except KeyError:
        # An averaged binary written before the key was named carries 'table', the key read_hdf
        # still allows for the binary of the single images. Only a checkout between 787fd26 and
        # ff1843e wrote one, since v0.9.0 already carried the path that never reached this file, so
        # this is the same courtesy read_hdf extends and not a case anyone is likely to hit.
        average_all_fitpar(channel, n_summarize=n_summarize, num_ref_imgs=num_ref_imgs,
                           ref_img_ids=ref_img_ids, path=path)
        fit_parameters = pd.read_hdf(file_path, key='channel_values')
    fit_parameters.set_index(['img_id', 'led_id'], inplace=True)
    return fit_parameters


def average_all_fitpar(channel, n_summarize: int = NUM_IMGS_PER_AVERAGE,
                       num_ref_imgs: int = DEFAULT_NUM_REF_IMGS,
                       ref_img_ids: List[int] = None,
                       path: Union[str, Path] = '.') -> None:  # TODO: rename variables within function
    """
    Averages all fit parameters for a given channel and writes them to an HDF file.

    The reference images of every LED are kept as they are, the remaining ones are averaged in
    groups of n_summarize, and the averaged images are numbered from 1 again. Called with the same
    n_summarize, num_ref_imgs and ref_img_ids as :func:`create_analysis_infos_avg`, the binary and
    the averaged image infos hold the same image IDs and the same number of images.

    :param channel: Channel number for which data is to be averaged.
    :type channel: int
    :param n_summarize: Number of images to average over.
    :type n_summarize: int
    :param num_ref_imgs: Number of reference images at the beginning of the experiment, used when
        ref_img_ids is None.
    :type num_ref_imgs: int
    :param ref_img_ids: IDs of the images the LED intensities are normalized with (excluded from
        averaging). None: the first num_ref_imgs images are kept.
    :type ref_img_ids: List[int] or None
    :param path: Root directory of the LEDSA simulation, defaults to the current directory.
    :type path: Union[str, Path]
    """
    fit_parameters = read_hdf(channel, path=path)
    fit_parameters_grouped = fit_parameters.groupby(['led_array_id', 'led_id'])
    avg_dataset_list = []
    for name, dataset in fit_parameters_grouped:
        # the images of one LED are summarized in the order they were taken, whatever order they
        # lie in the binary
        dataset = dataset.sort_index(level='img_id')
        groups = build_avg_img_groups(dataset.index.get_level_values('img_id').tolist(),
                                      n_summarize=n_summarize, num_ref_imgs=num_ref_imgs,
                                      ref_img_ids=ref_img_ids)
        if not groups:
            continue
        # a reference image is a group of its own, so its mean is the image itself and it passes
        # the averaging unchanged
        group_of_row = np.concatenate([np.full(len(group), group_idx)
                                       for group_idx, group in enumerate(groups)])
        avg_dataset = dataset.groupby(group_of_row).mean()
        avg_dataset["img_id"] = range(1, len(groups) + 1)
        avg_dataset["led_id"] = name[1]
        avg_dataset.set_index(['img_id', 'led_id'], inplace=True)
        avg_dataset_list.append(avg_dataset)
    all_fitpar = pd.concat(avg_dataset_list)
    all_fitpar["led_array_id"] = all_fitpar["led_array_id"].astype(int)
    all_fitpar.reset_index(inplace=True)
    out_file_path = get_binary_file_path(channel, path=path, averaged=True)
    if os.path.exists(out_file_path):
        # The averaged data are derived from the binary of the single images, so they are
        # rewritten from scratch. Appending would keep every LED of the earlier run next to the
        # new one, and after a change of the columns it fails in the middle of the write.
        print(f'{out_file_path} of an earlier run is replaced.')
        os.remove(out_file_path)
    all_fitpar.to_hdf(out_file_path, key='channel_values', format='table')
    with pd.HDFStore(out_file_path, mode='a') as store:
        # kept with the data, so that an analysis with other reference images computes the averages
        # again instead of reading those of the earlier run
        store.get_storer('channel_values').attrs.averaging_parameters = _averaging_parameters(
            n_summarize, num_ref_imgs, ref_img_ids)


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
    file = get_binary_file_path(channel)
    fit_parameters = pd.read_hdf(file, key='channel_values')
    fit_parameters[quantity] = values
    fit_parameters.to_hdf(file, key='channel_values', format='table')


def _number_of_analysed_images(config) -> int:
    """
    Number of images the data extraction gave an analysis ID.

    The analysis IDs are a gapless sequence from 1 on, so the highest one is the number of images.
    It is read from the image infos, which name the images the extraction really worked on, the
    reference images that num_skip_imgs leaves out included. Only when they are missing, the
    number is computed from the image range of the configuration, as every version before them did.

    :param config: Configuration data of the experiment.
    :type config: ConfigData
    :return: Number of analysed images.
    :rtype: int
    """
    file_path = os.path.join('analysis', 'image_infos_analysis.csv')
    if os.path.exists(file_path):
        return int(pd.read_csv(file_path)['#ID'].max())
    first_img_id = int(config['analyse_photo']['first_img_analysis_id'])
    last_img_id = int(config['analyse_photo']['last_img_analysis_id'])
    if config['DEFAULT']['num_img_overflow'] != 'None':
        max_id = int(config['DEFAULT']['num_img_overflow'])
    else:
        max_id = 10 ** 7
    number_of_images = (max_id + last_img_id - first_img_id) % max_id + 1
    img_increment = int(config['analyse_photo']['num_skip_imgs']) + 1
    # rounded up, because the first image is always analysed: the IDs 1 to 11 with every second
    # image skipped are the six images 1, 3, 5, 7, 9 and 11, not five
    return (number_of_images + img_increment - 1) // img_increment


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
        first_file_path = _find_first_result_file(channel)
        columns = _get_column_names(channel)
        # the names carry the img_id that is prepended on load and the coordinates that are appended below,
        # the result file itself has neither
        num_columns_of_first_file = len(columns) - 3

        fit_params_list = []

        # find time and fit parameter for every image

        number_of_images = _number_of_analysed_images(config)
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

            # The column names are taken from the first result file and used for all of them, so files of
            # different width cannot be loaded together. They occur when step 3 was run again for part of
            # the images after a LEDSA version that writes further columns was installed.
            if parameters.shape[-1] != num_columns_of_first_file:
                raise ValueError(f'{in_file_path} has {parameters.shape[-1]} columns, while '
                                 f'{first_file_path} has {num_columns_of_first_file} columns. The result '
                                 f'files of channel {channel} were written with different column layouts, '
                                 f'so the column names of the first file do not fit the others. Run the '
                                 f'data extraction (-s3 or -s3_fast) again for every image of the channel.')

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
        # max_col_val and sum_col_val stay float: the black level is a fraction of a count and the
        # values are negative where an LED is darker than the black level of the sensor
        out_file_path = get_binary_file_path(channel)
        if os.path.exists(out_file_path):
            # The binary data are derived from the step 3 files, so they are rewritten from scratch.
            # Appending would keep every LED of the earlier run next to the new one, and after a change of
            # the columns it fails in the middle of the write.
            print(f'{out_file_path} of an earlier run is replaced.')
            os.remove(out_file_path)
        fit_params.to_hdf(out_file_path, key='channel_values', format='table')
    finally:
        os.chdir(prev_dir)

# Column names of the step 3 result files (<img_id>_led_positions.csv). The files do not contain img_id, it is
# prepended when they are loaded, so a file with n columns has n + 1 names.
BASE_COLUMNS = ["img_id", "led_id", "led_array_id", "sum_col_val", "mean_col_val", "max_col_val"]
BGSUB_COLUMNS = ["bgsub_sum_col_val"]
SIDEBG_COLUMNS = ["sidebg_sum_col_val", "sidebg_pedestal", "bgsub2_sum_col_val"]
NOISE_SUM_COLUMNS = ["noise_sum_col_val"]
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
        BASE_COLUMNS + BGSUB_COLUMNS + SIDEBG_COLUMNS + LED_CENTER_COLUMNS + FIT_COLUMNS,
        # -s3 with fit, before the optical black
    len(BASE_COLUMNS + BGSUB_COLUMNS + SIDEBG_COLUMNS + NOISE_SUM_COLUMNS):
        BASE_COLUMNS + BGSUB_COLUMNS + SIDEBG_COLUMNS + NOISE_SUM_COLUMNS,  # -s3_fast
    len(BASE_COLUMNS + BGSUB_COLUMNS + SIDEBG_COLUMNS + NOISE_SUM_COLUMNS + LED_CENTER_COLUMNS
        + FIT_COLUMNS):
        BASE_COLUMNS + BGSUB_COLUMNS + SIDEBG_COLUMNS + NOISE_SUM_COLUMNS + LED_CENTER_COLUMNS
        + FIT_COLUMNS,  # -s3 with fit
}
# noise column of a reference property, used to weight the LEDs in the linear solver
NOISE_COLUMNS = {"sum_col_val": "noise_sum_col_val"}
# Fitted files written before the delimiter between the main and the fit part was added: max_col_value and
# led_center_x are glued into one field, so the file has one column less and the names are shifted from there on.
LEGACY_FIT_COLUMNS = BASE_COLUMNS + LED_CENTER_COLUMNS + FIT_COLUMNS[:-1]


def _find_first_result_file(channel: int) -> str:
    """
    Find the step 3 result file with the lowest image ID of a channel.

    An image that could not be read has no result file at all, so the lowest analysed image ID is not
    necessarily 1 and the result file of image 1 does not have to exist.

    :param channel: Channel number whose result files are searched.
    :type channel: int
    :return: Path of the result file with the lowest image ID.
    :rtype: str
    :raises FileNotFoundError: If the channel has no result file.
    """
    dir_path = os.path.join('analysis', f'channel{channel}')
    img_ids = []
    if os.path.isdir(dir_path):
        for file_name in os.listdir(dir_path):
            img = re.search(r"(\d+)_led_positions.csv", file_name)
            if img is not None:
                img_ids.append(int(img.group(1)))
    if not img_ids:
        raise FileNotFoundError(f'No analysis results in {dir_path}. Run the data extraction (-s3 or '
                                f'-s3_fast) before the analysis.')
    return os.path.join(dir_path, f'{min(img_ids)}_led_positions.csv')


def _get_column_names(channel: int) -> List[str]:
    """
    Get the column names for the specified channel from the number of columns of the CSV files.

    :param channel: Channel number for which column names are to be determined.
    :type channel: int
    :return: List of column names.
    :rtype: List[str]
    :raises ValueError: If the file cannot be parsed or the number of columns matches no known layout.
    :raises FileNotFoundError: If the channel has no result file.
    """
    file_path = _find_first_result_file(channel)
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