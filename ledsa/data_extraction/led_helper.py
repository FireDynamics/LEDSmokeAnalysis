from datetime import timedelta, datetime

import exifread
from ledsa.core.ConfigData import ConfigData

import os
import numpy as np
import matplotlib.pyplot as plt
import rawpy

from typing import List

sep = os.path.sep


# """
# ------------------------------------
# File management
# ------------------------------------
# """

# should handle all exception for opening files
# when exception is thrown, ask if std conf-file should be used or user input
def load_file(filename: str, delim=' ', dtype='float', atleast_2d=False, silent=False) -> np.ndarray:
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


def read_file(filename: str, channel: int, colordepth=14) -> np.ndarray:
    """
    Returns a 2D array of channel values depending on the colordepth.
    8bit is default range for JPG. Bayer array is a 2D array where
    all channel values except the selected channel are masked.
    """
    extension = os.path.splitext(filename)[-1]
    if extension in ['.JPG','.JPEG', '.jpg', '.jpeg', '.PNG', '.png']:
        data = plt.imread(filename)
    elif extension in ['.CR2']:
        with rawpy.imread(filename) as raw:
            data = raw.raw_image_visible.copy()
            filter_array = raw.raw_colors_visible
            black_level = raw.black_level_per_channel[channel]
            white_level = raw.white_level
        channel_range = 2 ** colordepth - 1
        channel_array = data.astype(np.int16) - black_level
        channel_array = (channel_array * (channel_range / (white_level - black_level))).astype(np.int16)
        channel_array = np.clip(channel_array, 0, channel_range)
        if channel == 0 or channel == 2:
            channel_array = np.where(filter_array == channel, channel_array, 0)
        elif channel == 1:
            channel_array = np.where((filter_array == 1) | (filter_array == 3), channel_array, 0)
        return channel_array
    return data[:, :, channel]


# """
# ------------------------------------
# Image ID Functions and Conversions
# ------------------------------------
# """

def get_img_name(img_id: int) -> np.ndarray:
    infos = load_file('.{}analysis{}image_infos_analysis.csv'.format(sep, sep), ',', 'str',
                      silent=True, atleast_2d=True)
    for i in range(infos.shape[0]):
        if int(infos[i, 0]) == int(img_id):
            return infos[i, 1]
    raise NameError("Could not find an image name to id {}.".format(img_id))


def get_img_id(img_name: str) -> int:
    infos = load_file('.{}analysis{}image_infos_analysis.csv'.format(sep, sep), ',', 'str',
                      silent=True, atleast_2d=True)
    for i in range(infos.shape[0]):
        if infos[i, 1] == img_name:
            return infos[i, 0]
    raise NameError("Could not find an image id for {}.".format(img_name))


def get_last_img_id() -> int:
    infos = load_file('.{}analysis{}image_infos_analysis.csv'.format(sep, sep), ',', 'str',
                      silent=True, atleast_2d=True)
    return int(infos[-1, 0])


#TODO: check if time can be int (float comparisson)
def get_img_id_from_time(time: float) -> int:
    infos = load_file('.{}analysis{}image_infos_analysis.csv'.format(sep, sep), ',', 'str', silent=True)
    for i in range(infos.shape[0]):
        if float(infos[i, 3]) == time:
            return int(infos[i, 0])
    raise NameError("Could not find an image id at {}s.".format(time))


def get_time_from_img_id(img_id: int) -> int:
    infos = load_file('.{}analysis{}image_infos_analysis.csv'.format(sep, sep), ',', 'str', silent=True)
    for i in range(infos.shape[0]):
        if float(infos[i, 0]) == img_id:
            return int(float(infos[i, 3]))
    raise NameError("Could not find a time to image {}.".format(img_id))


# """
# ------------------------------------
# Outsourced logic - init
# ------------------------------------
# """

def create_needed_directories(channels: List[int]) -> None:
    if not os.path.exists('plots'):
        os.mkdir('plots')
        print("Directory plots created ")
    if not os.path.exists('analysis'):
        os.mkdir('analysis')
        print("Directory analysis created ")
    for channel in channels:
        if not os.path.exists('analysis{}channel{}'.format(sep, channel)):
            os.mkdir('analysis{}channel{}'.format(sep, channel))
            print("Directory analysis{}channel{} created".format(sep, channel))


def request_config_parameters(config: ConfigData) -> None:
    if config['DEFAULT']['time_img'] == 'None' and \
            config['DEFAULT']['exif_time_infront_real_time'] == 'None':
        config.in_time_img()
        config.save()
    if config['find_search_areas']['reference_img'] == 'None':
        config.in_ref_img()
        config.save()
    if config['DEFAULT']['exif_time_infront_real_time'] == 'None':
        config.in_time_diff_to_img_time()
        config.save()
    if config['DEFAULT']['img_name_string'] == 'None':
        config.in_img_name_string()
        config.save()
    if config['DEFAULT']['first_img'] == 'None':
        config.in_first_img()
        config.save()
    if config['DEFAULT']['last_img'] == 'None':
        config.in_last_img()
        config.save()
    if config['DEFAULT']['num_of_arrays'] == 'None':
        config.in_num_of_arrays()
        config.save()


def generate_image_infos_csv(config: ConfigData, build_experiment_infos=False, build_analysis_infos=False) -> None:
    config_switch = []
    if build_experiment_infos:
        config_switch.append('DEFAULT')
    if build_analysis_infos:
        config_switch.append('analyse_photo')
    for build_type in config_switch:
        if config['DEFAULT']['start_time'] == 'None':
            config.get_start_time()
            config.save()
        img_data = _build_img_data_string(build_type, config)

        if build_type == 'DEFAULT':
            _save_experiment_infos(img_data)
        if build_type == 'analyse_photo':
            _save_analysis_infos(img_data)


def _calc_experiment_and_real_time(build_type, config, tag, img_number):
    exif = _get_exif(config['DEFAULT']['img_directory'] +
                     config['DEFAULT']['img_name_string'].format(int(img_number)), tag)
    if not exif:
        raise ValueError("No EXIF metadata found")

    if f"EXIF {tag}" not in exif:
        raise ValueError("No EXIF time found")
    date, time_meta = exif[f"EXIF {tag}"].values.split(' ')
    date_time_img = _get_datetime_from_str(date, time_meta)

    experiment_time = date_time_img - config.get_datetime()
    experiment_time = experiment_time.total_seconds()
    time_diff = config[build_type]['exif_time_infront_real_time'].split('.')
    if len(time_diff) == 1:
        time_diff.append('0')
    time = date_time_img - timedelta(seconds=int(time_diff[0]), milliseconds=int(time_diff[1]))
    return experiment_time, time


def _get_exif(filename, tag):
    with open(filename, 'rb') as f:
        exif = exifread.process_file(f, details=False, stop_tag=tag)
    return exif


def _get_datetime_from_str(date, time):
    if date.find(":") != -1:
        date_time = datetime.strptime(date + ' ' + time, '%Y:%m:%d %H:%M:%S')
    else:
        date_time = datetime.strptime(date + ' ' + time, '%d.%m.%Y %H:%M:%S')
    return date_time


def _find_img_number_list(first, last, increment, number_string_length=4):
    if last >= first:
        return [str(ele).zfill(number_string_length) for ele in range(first, last + 1, increment)]

    largest_number = 0
    for i in range(number_string_length):
        largest_number += 9 * 10 ** i
    print(largest_number)
    num_list = [str(ele).zfill(number_string_length) for ele in range(first, largest_number + 1, increment)]
    num_list.extend([str(ele).zfill(number_string_length) for ele in
                     range(increment - (largest_number - int(num_list[-1])), last + 1, increment)])
    return num_list


def _build_img_data_string(build_type, config):
    img_data = ''
    img_idx = 1
    first_img = config.getint(build_type, 'first_img')
    last_img = config.getint(build_type, 'last_img')
    img_increment = config.getint(build_type, 'skip_imgs') + 1 if build_type == 'analyse_photo' else 1
    img_number_list = _find_img_number_list(first_img, last_img, img_increment)
    for img_number in img_number_list:
        tag = 'DateTimeOriginal'
        experiment_time, time = _calc_experiment_and_real_time(build_type, config, tag, img_number)
        img_data += (str(img_idx) + ',' + config[build_type]['img_name_string'].format(int(img_number)) +
                     ',' + time.strftime('%H:%M:%S') + ',' + str(experiment_time) + '\n')
        img_idx += 1
    return img_data


def _save_analysis_infos(img_data):
    out_file = open('.{}analysis{}image_infos_analysis.csv'.format(sep, sep), 'w')
    out_file.write("#ID,Name,Time[s],Experiment_Time[s]\n")
    out_file.write(img_data)
    out_file.close()


def _save_experiment_infos(img_data):
    out_file = open('image_infos.csv', 'w')
    out_file.write("#Count,Name,Time[s],Experiment_Time[s]\n")
    out_file.write(img_data)
    out_file.close()
