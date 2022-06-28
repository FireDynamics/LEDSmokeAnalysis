import os
import re
import time
from typing import List

import numpy as np
import rawpy
import scipy.optimize
from matplotlib import pyplot as plt

from ledsa.core.ConfigData import ConfigData
from ledsa.data_extraction.LEDAnalysisData import LEDAnalysisData
from ledsa.core.file_handling import read_table, sep
from ledsa.core.image_handling import get_img_name
from ledsa.data_extraction.model import target_function


def generate_analysis_data(img_filename: str, channel: int, search_areas: np.ndarray, line_indices: List[List[int]],
                           conf: ConfigData, fit_leds=True, debug=False, debug_led=None) -> List[LEDAnalysisData]:
    data = read_img('{}{}'.format(conf['img_directory'], img_filename), channel=channel)
    window_radius = int(conf['window_radius'])
    img_analysis_data = []

    if debug:
        analysis_res = _generate_led_analysis_data(conf, channel, data, debug, debug_led, img_filename, 0, search_areas,
                                                   window_radius, fit_leds)
        return analysis_res

    for led_array_idx in range(int(conf['num_of_arrays'])):
        print('processing LED array ', led_array_idx, '...')
        for iled in line_indices[led_array_idx]:
            if iled % (int(conf['skip_leds']) + 1) == 0:
                led_analysis_data = _generate_led_analysis_data(conf, channel, data, debug, iled, img_filename,
                                                                led_array_idx, search_areas, window_radius, fit_leds)
                img_analysis_data.append(led_analysis_data)
    return img_analysis_data


def create_fit_result_file(img_data: np.ndarray, img_id: int, channel: int) -> None:
    img_infos = read_table('analysis{}image_infos_analysis.csv'.format(sep), dtype='str', delim=',', silent=True,
                           atleast_2d=True)
    root = os.getcwd()
    root = root.split(sep)
    img_filename = get_img_name(img_id)

    _save_results_in_file(channel, img_data, img_filename, img_id, img_infos, root)


def create_imgs_to_process_file() -> None:
    image_infos = read_table('.{}analysis{}image_infos_analysis.csv'.format(sep, sep), dtype='str', delim=',',
                             atleast_2d=True)
    img_filenames = image_infos[:, 1]
    out_file = open('images_to_process.csv', 'w')
    for img in img_filenames:
        out_file.write('{}\n'.format(img))
    out_file.close()


def find_and_save_not_analysed_imgs(channel: int) -> None:
    image_infos = read_table('.{}analysis{}image_infos_analysis.csv'.format(sep, sep), dtype='str', delim=',',
                             atleast_2d=True)
    all_imgs = image_infos[:, 1]
    processed_img_ids = _find_analysed_img_ids(channel)
    processed_imgs = np.frompyfunc(get_img_name, 1, 1)(processed_img_ids)
    remaining_imgs = set(all_imgs)-set(processed_imgs)

    _save_list_of_remaining_imgs_needed_to_be_processed(remaining_imgs)


def _generate_led_analysis_data(conf, channel, data, debug, iled, img_filename, led_array_idx, search_areas,
                                window_radius, fit_leds=True):
    led_data = LEDAnalysisData(iled, led_array_idx, fit_leds)
    center_search_area_x = int(search_areas[iled, 1])
    center_search_area_y = int(search_areas[iled, 2])
    search_area = np.index_exp[center_search_area_x - window_radius:
                               center_search_area_x + window_radius,
                               center_search_area_y - window_radius:
                               center_search_area_y + window_radius]

    if fit_leds:
        start_time = time.process_time()
        led_data.fit_results, mesh = _fit_model_to_led(data[search_area])
        end_time = time.process_time()
        led_data.fit_time = end_time - start_time
        led_data.led_center_x = led_data.fit_results.x[0] + center_search_area_x - window_radius
        led_data.led_center_y = led_data.fit_results.x[1] + center_search_area_y - window_radius
        if debug:
            return led_data.fit_results.x
        if not led_data.fit_results.success:  # A > 255 or A < 0:
            _log_warnings(img_filename, channel, led_data, center_search_area_x, center_search_area_y,
                          data[search_area].shape, window_radius, conf)

    led_data.mean_color_value = np.mean(data[search_area])
    led_data.sum_color_value = np.sum(data[search_area])
    led_data.max_color_value = np.amax(data[search_area])

    return led_data


def _save_results_in_file(channel, img_data, img_filename, img_id, img_infos, root):
    out_file = open(f'analysis{sep}channel{channel}{sep}{img_id}_led_positions.csv', 'w')
    header = _create_header(channel, img_id, img_filename, img_infos, root, img_data[0].fit_leds)
    out_file.write(header)
    for led_data in img_data:
        out_file.write(str(led_data))
    out_file.close()


def _find_analysed_img_ids(channel):
    processed_imgs = []
    directory_content = os.listdir('.{}analysis{}channel{}'.format(sep, sep, channel))
    for file_name in directory_content:
        img = re.search(r"(\d+)_led_positions.csv", file_name)
        if img is not None:
            processed_imgs.append(int(img.group(1)))
    return processed_imgs


def _save_list_of_remaining_imgs_needed_to_be_processed(remaining_imgs):
    out_file = open('images_to_process.csv', 'w')
    for i in list(remaining_imgs):
        out_file.write('{}\n'.format(i))
    out_file.close()


def _fit_model_to_led(search_area):
    nx = search_area.shape[0]
    ny = search_area.shape[1]

    center_x = nx // 2
    center_y = ny // 2
    x0 = np.array([center_x, center_y, 2., 2., 200., 1.0, 1.0, 1.0])
    x = np.linspace(0.5, nx - 0.5, nx)
    y = np.linspace(0.5, ny - 0.5, ny)
    mesh = np.meshgrid(x, y)
    res = scipy.optimize.minimize(target_function, x0,
                                  args=(search_area, mesh), method='nelder-mead',
                                  options={'xtol': 1e-8, 'disp': False,
                                           'adaptive': False, 'maxiter': 10000})
    return res, mesh


def _log_warnings(img_filename, channel, led_data, cx, cy, size_of_search_area, window_radius, conf):
    res = ' '.join(np.array_str(led_data.fit_results.x).split()).replace('[ ', '[').replace(' ]', ']').replace(' ', ',')
    img_file_path = conf['img_directory'] + img_filename

    log = f'Irregularities while fitting:\n    {img_file_path} {led_data.led_id} {led_data.led_array} {res} ' \
          f'{led_data.fit_results.success} {led_data.fit_results.fun} {led_data.fit_results.nfev} ' \
          f'{size_of_search_area[0]} {size_of_search_area[1]} {led_data.led_center_x} {led_data.led_center_y} ' \
          f'{window_radius} {cx} {cy} {channel}'
    if not os.path.exists('.{}logfiles'.format(sep)):
        os.makedirs('.{}logfiles'.format(sep))
    logfile = open('.{}logfiles{}warnings.log'.format(sep, sep), 'a')
    logfile.write(log)
    logfile.write('\n')
    logfile.close()


def _create_header(channel, img_id, img_filename, img_infos, root, fit_leds):
    out_str = f'# image root = {root[-1]}, photo file name = {img_filename}, '
    out_str += f"channel = {channel}, "
    out_str += f"time[s] = {img_infos[int(img_id) - 1][3]}\n"
    out_str += "# id,line,sum_col_value,average_col_value,max_col_value"
    if fit_leds:
        out_str += ",led_center_x, led_center_y"
        out_str += ",x,y,dx,dy,A,alpha,wx,wy,fit_success,fit_fun,fit_nfev,fit_time"
        out_str += "// all spatial quantities in pixel coordinates\n"
    else:
        out_str += "\n"
    return out_str


def read_img(filename: str, channel: int, color_depth=14) -> np.ndarray:
    """
    Returns a 2D array of channel values depending on the color depth.
    8bit is default range for JPG. Bayer array is a 2D array where
    all channel values except the selected channel are masked.
    """
    extension = os.path.splitext(filename)[-1]
    data = []
    if extension in ['.JPG', '.JPEG', '.jpg', '.jpeg', '.PNG', '.png']:
        data = plt.imread(filename)
    elif extension in ['.CR2']:
        with rawpy.imread(filename) as raw:
            data = raw.raw_image_visible.copy()
            filter_array = raw.raw_colors_visible
            black_level = raw.black_level_per_channel[channel]
            white_level = raw.white_level
        channel_range = 2 ** color_depth - 1
        channel_array = data.astype(np.int16) - black_level
        channel_array = (channel_array * (channel_range / (white_level - black_level))).astype(np.int16)
        channel_array = np.clip(channel_array, 0, channel_range)
        if channel == 0 or channel == 2:
            channel_array = np.where(filter_array == channel, channel_array, 0)
        elif channel == 1:
            channel_array = np.where((filter_array == 1) | (filter_array == 3), channel_array, 0)
        return channel_array
    return data[:, :, channel]
