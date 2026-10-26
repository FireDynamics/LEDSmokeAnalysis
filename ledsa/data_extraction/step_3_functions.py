import os
import re
import time
from typing import List, Optional, Tuple

import numpy as np
import scipy.optimize

from ledsa.core.ConfigData import ConfigData
from ledsa.core.file_handling import read_table
from ledsa.core.image_handling import get_img_name
from ledsa.core.image_reading import read_channel_data_and_mask_from_img
from ledsa.data_extraction.LEDAnalysisData import LEDAnalysisData
from ledsa.data_extraction.model import target_function
from ledsa.data_extraction.sideband_background import (DEFAULT_SIDEBAND_DISTANCES, DEFAULT_SIDEBAND_FLOOR,
                                                        estimate_sideband_background, get_sideband_pixels,
                                                        subtract_pedestal)


def generate_analysis_data(img_filename: str, channel: int, search_areas: np.ndarray, line_indices: List[List[int]],
                           conf: ConfigData, fit_leds=True, debug=False, debug_led=None) -> List[LEDAnalysisData]:
    """
    Generate LED analysis data for the given image.

    :param img_filename: The filename of the image to be analyzed.
    :type img_filename: str
    :param channel: The color channel to be considered during analysis.
    :type channel: int
    :param search_areas: A numpy array containing the search areas for LEDs.
    :type search_areas: np.ndarray
    :param line_indices: IDs indicating the LEDs in the arrays.
    :type line_indices: List[List[int]]
    :param conf: Configuration data for analysis.
    :type conf: ConfigData
    :param fit_leds: Whether to fit the LED model to the data. Default is True.
    :type fit_leds: bool
    :param debug: If True, the function will run in debug mode. Default is False.
    :type debug: bool
    :param debug_led: The specific LED to debug. Default is None.
    :type debug_led: Optional[int]
    :return: A list of LEDAnalysisData objects containing analysis results.
    :rtype: List[LEDAnalysisData]
    """
    file_path = os.path.join(conf['DEFAULT']['img_directory'], img_filename)
    data, channel_mask = read_channel_data_and_mask_from_img(file_path, channel=channel)
    search_area_radius = int(conf['find_search_areas']['search_area_radius'])
    img_analysis_data = []

    if debug:
        analysis_res = _generate_led_analysis_data(conf, channel, data, debug, debug_led, img_filename, 0, search_areas,
                                                   search_area_radius, fit_leds)
        return analysis_res

    sideband_distances, sideband_floor, sideband_exclusion_margin = get_sideband_parameters(conf)
    sideband_pixels = get_sideband_pixels(search_areas, _get_unmerged_led_arrays(conf, line_indices), data.shape,
                                          search_area_radius, sideband_distances, sideband_exclusion_margin)

    num_of_arrays = len(line_indices)
    for led_array_idx in range(num_of_arrays):
        for iled in line_indices[led_array_idx]:
            if iled % (int(conf['analyse_photo']['num_skip_leds']) + 1) == 0:
                led_analysis_data = _generate_led_analysis_data(conf, channel, data, debug, iled, img_filename,
                                                                led_array_idx, search_areas, search_area_radius, fit_leds,
                                                                channel_mask=channel_mask,
                                                                sideband_pixels=sideband_pixels[iled],
                                                                sideband_floor=sideband_floor)
                img_analysis_data.append(led_analysis_data)
    return img_analysis_data


def get_sideband_parameters(conf: ConfigData) -> Tuple[Tuple[float, float], float, Optional[int]]:
    """
    Read the parameters of the side band pedestal estimation from the config. Config files written before the
    side band estimation was added do not contain the options, the defaults are used then.

    :param conf: Configuration data.
    :type conf: ConfigData
    :return: Distances (inner, outer) of the side bands across the strip in pixels, lower limit of the corrected
        sum as fraction of the box sum and margin for the exclusion of other search areas (None: R).
    :rtype: tuple[tuple[float, float], float, int or None]
    """
    distances = DEFAULT_SIDEBAND_DISTANCES
    floor = DEFAULT_SIDEBAND_FLOOR
    exclusion_margin = None
    if conf is not None:
        if conf.get('analyse_photo', 'sideband_distances') != 'None':
            distances = tuple(float(d) for d in conf.get('analyse_photo', 'sideband_distances').split())
            if len(distances) != 2 or not 0 <= distances[0] < distances[1]:
                raise ValueError(f'sideband_distances must be two distances 0 <= inner < outer, got {distances}.')
        if conf.get('analyse_photo', 'sideband_floor') != 'None':
            floor = float(conf.get('analyse_photo', 'sideband_floor'))
        if conf.get('analyse_photo', 'sideband_exclusion_margin') != 'None':
            exclusion_margin = int(conf.get('analyse_photo', 'sideband_exclusion_margin'))
    return distances, floor, exclusion_margin


def _get_unmerged_led_arrays(conf: ConfigData, line_indices: List[List[int]]) -> List[List[int]]:
    """
    Return the LED arrays as found in step 2. If LED arrays are merged for the analysis, the merged arrays may
    contain several strips, which would spoil the estimate of the local strip direction.

    :param conf: Configuration data.
    :type conf: ConfigData
    :param line_indices: LED ids of the (possibly merged) LED arrays used in the analysis.
    :type line_indices: List[List[int]]
    :return: LED ids of the LED arrays as found in step 2.
    :rtype: List[List[int]]
    """
    if conf is None or conf['analyse_positions']['merge_led_array_indices'] == 'None':
        return line_indices
    led_arrays = []
    for i in range(int(conf['analyse_positions']['num_arrays'])):
        file_path = os.path.join('analysis', f'led_array_indices_{i:03}.csv')
        led_arrays.append(read_table(file_path, dtype='int', silent=True))
    return led_arrays


def create_fit_result_file(img_data: List[LEDAnalysisData], img_id: int, channel: int) -> None: # TODO: rename because misleading
    """
      Create a result file for a single image, containing the pixel values and, if applicable, the fit results of all LEDs.

      :param img_data: A list, containing LEDAnalysisData objects with pixel values and, if applicable, the fit results of all LEDs.
      :type img_data: List[LEDAnalysisData]
      :param img_id: Identifier for the image.
      :type img_id: int
      :param channel: Color channel being analyzed.
      :type channel: int
      """
    file_path = os.path.join('analysis', 'image_infos_analysis.csv')
    img_infos = read_table(file_path, dtype='str', delim=',', silent=True, atleast_2d=True)
    basename = os.path.basename(os.getcwd())
    img_filename = get_img_name(img_id)

    _save_results_in_file(channel, img_data, img_filename, img_id, img_infos, basename)


def create_imgs_to_process_file() -> None:
    """
    Create a file with filenames of images that need to be processed.
    """
    file_path = os.path.join('analysis', 'image_infos_analysis.csv')
    image_infos = read_table(file_path, dtype='str', delim=',', atleast_2d=True)
    img_filenames = image_infos[:, 1]
    out_file = open('images_to_process.csv', 'w', encoding='utf-8')
    for img in img_filenames:
        out_file.write('{}\n'.format(img))
    out_file.close()


def find_and_save_not_analysed_imgs(channel: int) -> None:
    """
    Find and save filenames of images that have not yet been analyzed.

    :param channel: Channel to check for analyzed images.
    :type channel: int
    """
    file_path = os.path.join('analysis', 'image_infos_analysis.csv')
    image_infos = read_table(file_path, dtype='str', delim=',', atleast_2d=True)
    all_imgs = image_infos[:, 1]
    processed_img_ids = _find_analysed_img_ids(channel)
    processed_imgs = np.frompyfunc(get_img_name, 1, 1)(processed_img_ids)
    remaining_imgs = set(all_imgs)-set(processed_imgs)

    _save_list_of_remaining_imgs_needed_to_be_processed(remaining_imgs)


def _generate_led_analysis_data(conf: ConfigData, channel: int, data: np.ndarray, debug: bool, iled: int, img_filename: str, led_array_idx: int, search_areas: np.ndarray, search_area_radius: int, fit_leds: bool = True,
                                channel_mask: Optional[np.ndarray] = None,
                                sideband_pixels: Optional[Tuple[np.ndarray, np.ndarray]] = None,
                                sideband_floor: float = DEFAULT_SIDEBAND_FLOOR) -> LEDAnalysisData:
    """
    Generate analysis data for a specific LED.

    :param conf: Configuration data.
    :type conf: ConfigData
    :param channel: Color Channel for which analysis should be generated.
    :type channel: int
    :param data: Array representing the image data.
    :type data: np.ndarray
    :param debug: If True, enters debug mode.
    :type debug: bool
    :param iled: ID of the LED for which analysis should be generated.
    :type iled: int
    :param img_filename: Name of the image.
    :type img_filename: str
    :param led_array_idx: Index of the LED array where the LED is on.
    :type led_array_idx: int
    :param search_areas: Array containing the pixel positions of all search areas on the image.
    :type search_areas: np.ndarray
    :param search_area_radius: Radius of the search area.
    :type search_area_radius: int
    :param fit_leds: If True, the LED is fitted to a model function.
    :type fit_leds: bool
    :param channel_mask: Boolean mask of the pixels belonging to the color channel (Bayer pattern for raw files).
        None if all pixels belong to the channel.
    :type channel_mask: np.ndarray or None
    :param sideband_pixels: Row and column indices of the side band pixels of the LED, None if not available.
    :type sideband_pixels: tuple[np.ndarray, np.ndarray] or None
    :param sideband_floor: Lower limit of the side band and ring corrected sums as fraction of the box sum.
    :type sideband_floor: float
    :return: Analysis data for the LED.
    :rtype: LEDAnalysisData
    """
    led_data = LEDAnalysisData(iled, led_array_idx, fit_leds)
    center_search_area_x = int(search_areas[iled, 1])
    center_search_area_y = int(search_areas[iled, 2])
    search_area = np.index_exp[center_search_area_x - search_area_radius:
                               center_search_area_x + search_area_radius,
                               center_search_area_y - search_area_radius:
                               center_search_area_y + search_area_radius]

    if fit_leds:
        start_time = time.process_time()
        led_data.fit_results, mesh = _fit_model_to_led(data[search_area])
        end_time = time.process_time()
        led_data.fit_time = end_time - start_time
        led_data.led_center_x = led_data.fit_results.x[0] + center_search_area_x - search_area_radius
        led_data.led_center_y = led_data.fit_results.x[1] + center_search_area_y - search_area_radius
        if debug:
            return led_data.fit_results.x
        if not led_data.fit_results.success:  # A > 255 or A < 0:
            _log_warnings(img_filename, channel, led_data, center_search_area_x, center_search_area_y,
                          data[search_area].shape, search_area_radius, conf)

    led_data.mean_color_value = np.mean(data[search_area])
    led_data.sum_color_value = np.sum(data[search_area])
    led_data.max_color_value = np.amax(data[search_area])
    background = _estimate_local_background(data[search_area])
    led_data.bgsub_sum_color_value = float(np.sum(np.clip(data[search_area].astype(np.float64) - background,
                                                          0.0, None)))

    # the pedestals below are subtracted as N * b with N the number of channel pixels in the box
    box_mask = None if channel_mask is None else channel_mask[search_area]
    num_channel_pixels = data[search_area].size if box_mask is None else int(np.count_nonzero(box_mask))

    # border ring as for bgsub, but with the estimator of the side bands: mean over the channel pixels including 0;
    # same lower limit as for the side band correction
    ring_background = _estimate_local_background_mean(data[search_area], box_mask)
    led_data.bgsub2_sum_color_value = subtract_pedestal(float(led_data.sum_color_value),
                                                        num_channel_pixels * ring_background, sideband_floor)

    # b = mean over the channel pixels in the side bands next to the LED strip
    sideband_background = estimate_sideband_background(data, channel_mask, sideband_pixels)
    led_data.sidebg_pedestal = num_channel_pixels * sideband_background
    led_data.sidebg_sum_color_value = subtract_pedestal(float(led_data.sum_color_value), led_data.sidebg_pedestal,
                                                        sideband_floor)

    return led_data


def _estimate_local_background(search_area_data: np.ndarray, border_width: int = 2) -> float:
    """
    Estimate the local background of a search area from the median of its border pixels.

    The LED sits at the center of the search area, so the outer border ring samples the
    scene background (stray light, smoke path radiance). Pixels with value 0 are excluded
    because raw Bayer arrays mask all pixels of foreign color channels with 0 and the
    black level subtraction clips at 0.

    :param search_area_data: Part of the image where the LED is located.
    :type search_area_data: np.ndarray
    :param border_width: Width of the border ring in pixels.
    :type border_width: int
    :return: Estimated background value per pixel. 0 if no valid border pixels exist.
    :rtype: float
    """
    if search_area_data.shape[0] <= 2 * border_width or search_area_data.shape[1] <= 2 * border_width:
        return 0.0
    border_mask = np.ones(search_area_data.shape, dtype=bool)
    border_mask[border_width:-border_width, border_width:-border_width] = False
    border_pixels = search_area_data[border_mask]
    border_pixels = border_pixels[border_pixels > 0]
    if border_pixels.size == 0:
        return 0.0
    return float(np.median(border_pixels))


def _estimate_local_background_mean(search_area_data: np.ndarray, channel_mask: Optional[np.ndarray] = None,
                                    border_width: int = 2) -> float:
    """
    Estimate the local background of a search area as the mean over the channel pixels of its border ring.

    Unlike _estimate_local_background, pixels with value 0 are kept if they belong to the color channel: they are
    noise clipped at 0 after the black level subtraction and are part of the box sum in the same way. Pixels of
    foreign Bayer channels are excluded via the channel mask. The result is meant to be subtracted as N * b from the
    box sum (N channel pixels in the box), which is linear in the pixel values, instead of pixel-wise with clipping.
    Note that the border ring lies in the halo of the LED for small search areas.

    :param search_area_data: Part of the image where the LED is located.
    :type search_area_data: np.ndarray
    :param channel_mask: Boolean mask of the channel pixels in the search area, None if all pixels belong to the
        channel.
    :type channel_mask: np.ndarray or None
    :param border_width: Width of the border ring in pixels.
    :type border_width: int
    :return: Mean background per channel pixel, NaN if the border ring contains no channel pixel.
    :rtype: float
    """
    if search_area_data.shape[0] <= 2 * border_width or search_area_data.shape[1] <= 2 * border_width:
        return np.nan
    border_mask = np.ones(search_area_data.shape, dtype=bool)
    border_mask[border_width:-border_width, border_width:-border_width] = False
    if channel_mask is not None:
        border_mask &= channel_mask
    if not border_mask.any():
        return np.nan
    return float(np.mean(search_area_data[border_mask], dtype=np.float64))


def _save_results_in_file(channel: int, img_data: LEDAnalysisData, img_filename: str, img_id: str, img_infos: np.ndarray, basename: str) -> None:
    """
    Save analysis results to a file.

    :param channel: Color channel being analyzed.
    :type channel: int
    :param img_data: A list, containing LEDAnalysisData objects with pixel values and, if applicable, the fit results of all LEDs.
    :type img_data: LEDAnalysisData
    :param img_filename: Name of the image.
    :type img_filename: str
    :param img_id: Identifier for the image.
    :type img_id: int
    :param img_infos: An array containing the image IDs, image filenames, capture time and experiment time for all images.
    :type img_infos: np.ndarray
    :param basename: Base name for the file to save.
    :type basename: str
    """
    file_path = os.path.join('analysis', f'channel{channel}', f'{img_id}_led_positions.csv')
    out_file = open(file_path, 'w', encoding='utf-8')
    header = _create_header(channel, img_id, img_filename, img_infos, basename, img_data[0].fit_leds)
    out_file.write(header)
    for led_data in img_data:
        out_file.write(str(led_data))
    out_file.close()


def _find_analysed_img_ids(channel):
    """
    Find and save filenames of images that have not yet been analyzed.

    :param channel: Channel to check for analyzed images.
    :return: List of IDs of analyzed images.
    :rtype: List[int]
    """
    processed_imgs = []
    dir_path = os.path.join('analysis', f'channel{channel}')
    directory_content = os.listdir(dir_path)
    for file_name in directory_content:
        img = re.search(r"(\d+)_led_positions.csv", file_name)
        if img is not None:
            processed_imgs.append(int(img.group(1)))
    return processed_imgs


def _save_list_of_remaining_imgs_needed_to_be_processed(remaining_imgs): # TODO: Why extra function needed?
    """
     Save a list of filenames of images that still need to be processed.

    :param remaining_imgs: List of remaining image filenames.
    :type remaining_imgs: list[str]
    """
    out_file = open('images_to_process.csv', 'w', encoding='utf-8')
    for i in list(remaining_imgs):
        out_file.write('{}\n'.format(i))
    out_file.close()


def _fit_model_to_led(search_area: np.ndarray) -> Tuple[scipy.optimize.OptimizeResult, List[np.ndarray]]:
    """
    Fit a model to the LED in a specific search area.

    :param search_area: Part of the image where the LED is located.
    :type search_area: np.ndarray
    :return: Result of the fit and mesh of the search area.
    :rtype: tuple[scipy.optimize.OptimizeResult, List[np.ndarray]]
    """
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
                                  options={'xatol': 1e-8, 'disp': False,
                                           'adaptive': False, 'maxiter': 10000})
    return res, mesh


def _log_warnings(img_filename, channel, led_data, cx, cy, size_of_search_area, search_area_radius, conf) -> None:
    """
    Log warnings that occur during LED fitting.

    :param img_filename: Name of the image.
    :param channel: Color channel being analyzed.
    :param led_data: Analysis data for the LED.
    :type led_data: LEDAnalysisData
    :param cx:  X pixel coordinate of the LED on the image.
    :type cx: int
    :param cy: Y pixel coordinate of the LED on the image.
    :type cy: int
    :param size_of_search_area: Dimensions of the search area.
    :type size_of_search_area: tuple[int, int]
    :param search_area_radius: Radius of the search area.
    :type search_area_radius: int
    :param conf: Configuration data for the analysis.
    :type conf: ConfigData
    """
    res = ' '.join(np.array_str(led_data.fit_results.x).split()).replace('[ ', '[').replace(' ]', ']').replace(' ', ',')
    img_file_path = os.path.join(conf['DEFAULT']['img_directory'], img_filename)

    log = f'Irregularities while fitting:\n    {img_file_path} {led_data.led_id} {led_data.led_array} {res} ' \
          f'{led_data.fit_results.success} {led_data.fit_results.fun} {led_data.fit_results.nfev} ' \
          f'{size_of_search_area[0]} {size_of_search_area[1]} {led_data.led_center_x} {led_data.led_center_y} ' \
          f'{search_area_radius} {cx} {cy} {channel}'
    dir_path = 'logfiles'
    if not os.path.exists(dir_path):
        os.makedirs(dir_path)
    file_path = os.path.join(dir_path, 'warnings.log')
    logfile = open(file_path, 'a', encoding='utf-8')
    logfile.write(log)
    logfile.write('\n')
    logfile.close()


def _create_header(channel: int, img_id: str, img_filename: str, img_infos: np.ndarray, basename: str, fit_leds: str):
    """
    Create a header for the analysis result file.

    :param channel: Color channel being analyzed.
    :type channel: int
    :param img_id: Identifier for the image.
    :type img_id: str
    :param img_filename: Name of the image.
    :type img_filename: str
    :param img_infos: An array containing the image IDs, image filenames, capture time and experiment time for all images.
    :type img_infos: np.ndarray
    :param basename: Base name for the file.
    :type basename: str
    :param fit_leds: If True, includes fit data in the header.
    :type fit_leds: bool
    :return: String of the header.
    :rtype: str
    """
    out_str = f'# image root = {basename}, photo file name = {img_filename}, '
    out_str += f"channel = {channel}, "
    out_str += f"time[s] = {img_infos[int(img_id) - 1][3]}\n"
    out_str += "# id,line,sum_col_value,average_col_value,max_col_value,bgsub_sum_col_value"
    out_str += ",sidebg_sum_col_value,sidebg_pedestal,bgsub2_sum_col_value"
    if fit_leds:
        out_str += ",led_center_x, led_center_y"
        out_str += ",x,y,dx,dy,A,alpha,wx,wy,fit_success,fit_fun,fit_nfev,fit_time"
        out_str += "// all spatial quantities in pixel coordinates\n"
    else:
        out_str += "\n"
    return out_str
