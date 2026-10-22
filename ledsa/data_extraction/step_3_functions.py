import os
import re
import time
from typing import Dict, List, Tuple

import numpy as np
import scipy.optimize

from ledsa.core.ConfigData import ConfigData
from ledsa.core.file_handling import read_table
from ledsa.core.image_handling import get_img_name
from ledsa.core.image_reading import read_channel_data_from_img
from ledsa.data_extraction.LEDAnalysisData import LEDAnalysisData
from ledsa.data_extraction.model import target_function

# Column indices of the LED id and of the LED center coordinates in the '{img_id}_led_positions.csv' files
LED_ID_COLUMN = 0
LED_CENTER_COLUMNS = [5, 6]


def generate_analysis_data(img_filename: str, channel: int, search_areas: np.ndarray, line_indices: List[List[int]],
                           conf: ConfigData, fit_leds=True, track_led_movement=False, debug=False,
                           debug_led=None) -> List[LEDAnalysisData]:
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
    :param track_led_movement: If True, the LED center is determined from the brightest pixels of the search area
                               instead of being fixed to the search area center. Ignored if the LEDs are fitted,
                               because the fit locates the LED itself. Default is False.
    :type track_led_movement: bool
    :param debug: If True, the function will run in debug mode. Default is False.
    :type debug: bool
    :param debug_led: The specific LED to debug. Default is None.
    :type debug_led: Optional[int]
    :return: A list of LEDAnalysisData objects containing analysis results.
    :rtype: List[LEDAnalysisData]
    """
    file_path = os.path.join(conf['DEFAULT']['img_directory'], img_filename)
    data = read_channel_data_from_img(file_path, channel=channel)
    search_area_radius = int(conf['find_search_areas']['search_area_radius'])
    img_analysis_data = []

    if debug:
        analysis_res = _generate_led_analysis_data(conf, channel, data, debug, debug_led, img_filename, 0, search_areas,
                                                   search_area_radius, fit_leds, track_led_movement)
        return analysis_res

    num_of_arrays = len(line_indices)
    for led_array_idx in range(num_of_arrays):
        for iled in line_indices[led_array_idx]:
            if iled % (int(conf['analyse_photo']['num_skip_leds']) + 1) == 0:
                led_analysis_data = _generate_led_analysis_data(conf, channel, data, debug, iled, img_filename,
                                                                led_array_idx, search_areas, search_area_radius,
                                                                fit_leds, track_led_movement)
                img_analysis_data.append(led_analysis_data)
    return img_analysis_data


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
    out_file = open('images_to_process.csv', 'w')
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
    processed_imgs = set(np.frompyfunc(get_img_name, 1, 1)(processed_img_ids))
    # Filtering instead of a set difference keeps the images in chronological order, which LED movement tracking needs
    remaining_imgs = [img for img in all_imgs if img not in processed_imgs]

    _save_list_of_remaining_imgs_needed_to_be_processed(remaining_imgs)


def sort_img_filenames_by_id(img_filenames: List[str]) -> List[str]:
    """
    Sort image filenames by their analysis image ID, which orders them chronologically.

    :param img_filenames: Names of the image files to sort.
    :type img_filenames: List[str]
    :return: The image filenames ordered by increasing analysis image ID.
    :rtype: List[str]
    :raises KeyError: If an image filename is not listed in 'image_infos_analysis.csv'.
    """
    file_path = os.path.join('analysis', 'image_infos_analysis.csv')
    img_infos = read_table(file_path, dtype='str', delim=',', silent=True, atleast_2d=True)
    img_id_of_filename = {img_filename: int(img_id) for img_id, img_filename in zip(img_infos[:, 0], img_infos[:, 1])}
    return sorted(img_filenames, key=lambda img_filename: img_id_of_filename[img_filename])


def init_tracked_search_areas(search_areas: np.ndarray, channel: int, seed_from_analysed_imgs=False) -> np.ndarray:
    """
    Create the search areas used for LED movement tracking of a single channel.

    The tracked search areas are a floating point copy of the static search areas, so that LED positions can be
    stored with sub pixel accuracy. On a restart they are seeded with the LED positions of the most recently
    analysed image, so that the tracking continues where the interrupted run stopped.

    :param search_areas: Array containing the LED IDs and the pixel positions of all search areas.
    :type search_areas: np.ndarray
    :param channel: Color channel for which the search areas are tracked.
    :type channel: int
    :param seed_from_analysed_imgs: If True, the LED positions of the most recently analysed image are used as start values.
    :type seed_from_analysed_imgs: bool
    :return: Array containing the LED IDs and the pixel positions of all tracked search areas.
    :rtype: np.ndarray
    """
    tracked_search_areas = search_areas.astype(float)
    if not seed_from_analysed_imgs:
        return tracked_search_areas

    analysed_img_ids = _find_analysed_img_ids(channel)
    if not analysed_img_ids:
        return tracked_search_areas

    file_path = os.path.join('analysis', f'channel{channel}', f'{max(analysed_img_ids)}_led_positions.csv')
    led_positions = read_table(file_path, delim=',', silent=True, atleast_2d=True)
    if led_positions.shape[1] <= max(LED_CENTER_COLUMNS):
        print(f'{file_path} does not contain LED centers. LED movement tracking of channel {channel} restarts from '
              f'the search areas of the reference image.')
        return tracked_search_areas

    row_of_led_id = _get_row_of_led_id(tracked_search_areas)
    for led_position in led_positions:
        row = row_of_led_id.get(int(led_position[LED_ID_COLUMN]))
        if row is not None:
            tracked_search_areas[row, 1:3] = led_position[LED_CENTER_COLUMNS]
    print(f'LED movement tracking of channel {channel} continues with the LED positions of {file_path}.')
    return tracked_search_areas


def update_tracked_search_areas(tracked_search_areas: np.ndarray, img_data: List[LEDAnalysisData]) -> None:
    """
    Move the tracked search areas to the LED positions found in the current image.

    LEDs that were not analysed, for example because they are skipped by num_skip_leds, keep their search area.
    Since the LED position is the brightest pixel within the search area, a search area can move at most by the
    search area radius from one image to the next.

    :param tracked_search_areas: Array containing the LED IDs and the pixel positions of all tracked search areas.
                                 Updated in place.
    :type tracked_search_areas: np.ndarray
    :param img_data: A list, containing LEDAnalysisData objects with the LED centers found in the current image.
    :type img_data: List[LEDAnalysisData]
    """
    row_of_led_id = _get_row_of_led_id(tracked_search_areas)
    for led_data in img_data:
        row = row_of_led_id.get(int(led_data.led_id))
        if row is not None:
            tracked_search_areas[row, 1:3] = (led_data.led_center_x, led_data.led_center_y)


def _generate_led_analysis_data(conf: ConfigData, channel: int, data: np.ndarray, debug: bool, iled: int, img_filename: str, led_array_idx: int, search_areas: np.ndarray, search_area_radius: int, fit_leds: bool = True, track_led_movement: bool = False) -> LEDAnalysisData:
    """
    Generate analysis data for a specific LED.

    :param conf: Configuration data.
    :type conf: ConfigData
    :param channel: Color channel for which analysis should be generated.
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
    :param track_led_movement: If True, the LED center is set to the position of the brightest pixels within the
                               search area instead of the search area center. Ignored if the LED is fitted.
    :type track_led_movement: bool
    :return: Analysis data for the LED.
    :rtype: LEDAnalysisData
    """
    led_data = LEDAnalysisData(iled, led_array_idx, fit_leds)
    center_search_area_x = int(round(search_areas[iled, 1]))
    center_search_area_y = int(round(search_areas[iled, 2]))

    # Clip the search area to the image, so that LEDs close to the image border do not wrap around
    x_min = max(0, center_search_area_x - search_area_radius)
    x_max = min(data.shape[0], center_search_area_x + search_area_radius)
    y_min = max(0, center_search_area_y - search_area_radius)
    y_max = min(data.shape[1], center_search_area_y + search_area_radius)
    search_area = np.index_exp[x_min:x_max, y_min:y_max]

    if fit_leds:
        start_time = time.process_time()
        led_data.fit_results, mesh = _fit_model_to_led(data[search_area])
        end_time = time.process_time()
        led_data.fit_time = end_time - start_time
        led_data.led_center_x = led_data.fit_results.x[0] + x_min
        led_data.led_center_y = led_data.fit_results.x[1] + y_min
        if debug:
            return led_data.fit_results.x
        if not led_data.fit_results.success:  # A > 255 or A < 0:
            _log_warnings(img_filename, channel, led_data, center_search_area_x, center_search_area_y,
                          data[search_area].shape, search_area_radius, conf)
    elif track_led_movement:
        led_data.led_center_x, led_data.led_center_y = _find_max_pixel_position(data[search_area], x_min, y_min)
    else:
        led_data.led_center_x = center_search_area_x
        led_data.led_center_y = center_search_area_y

    led_data.mean_color_value = np.mean(data[search_area])
    led_data.sum_color_value = np.sum(data[search_area])
    led_data.max_color_value = np.amax(data[search_area])

    return led_data


def _find_max_pixel_position(search_area_data: np.ndarray, x_offset: int, y_offset: int) -> Tuple[float, float]:
    """
    Find the pixel position of the LED within a search area.

    The center of all pixels holding the maximum value is used instead of the first maximum found, because the core
    of a bright LED is usually saturated over several pixels. Picking the first maximum would bias the position
    towards the upper left corner of that plateau and, since the position is fed back into the next image, would
    make the search area drift.

    :param search_area_data: Part of the image data belonging to the search area.
    :type search_area_data: np.ndarray
    :param x_offset: X pixel coordinate of the first pixel of the search area on the image.
    :type x_offset: int
    :param y_offset: Y pixel coordinate of the first pixel of the search area on the image.
    :type y_offset: int
    :return: X and Y pixel coordinate of the LED on the image.
    :rtype: tuple[float, float]
    """
    max_positions = np.argwhere(search_area_data == np.amax(search_area_data))
    max_position_x, max_position_y = max_positions.mean(axis=0)
    return max_position_x + x_offset, max_position_y + y_offset


def _get_row_of_led_id(search_areas: np.ndarray) -> Dict[int, int]:
    """
    Map every LED ID to its row in the search area array.

    :param search_areas: Array containing the LED IDs and the pixel positions of all search areas.
    :type search_areas: np.ndarray
    :return: Row index of every LED ID.
    :rtype: Dict[int, int]
    """
    return {int(led_id): row for row, led_id in enumerate(search_areas[:, LED_ID_COLUMN])}


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
    out_file = open(file_path, 'w')
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
    out_file = open('images_to_process.csv', 'w')
    for i in list(remaining_imgs):
        out_file.write('{}\n'.format(i))
    out_file.close()


def _fit_model_to_led(search_area: np.ndarray) -> Tuple[scipy.optimize.OptimizeResult, List[np.ndarray]]:
    """
    Fit a model to the LED in a specific search area.

    The mesh uses matrix indexing, so that the fitted x and y coordinates refer to the first and the second axis of
    the search area, as everywhere else in LEDSA, and so that search areas clipped at the image border, which are
    not square, can be fitted as well.

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
    mesh = np.meshgrid(x, y, indexing='ij')
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
    logfile = open(file_path, 'a')
    logfile.write(log)
    logfile.write('\n')
    logfile.close()


def _create_header(channel: int, img_id: str, img_filename: str, img_infos: np.ndarray, basename: str, fit_leds: bool) -> str:
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
    out_str += "# id,line,sum_col_value,average_col_value,max_col_value"
    out_str += ",led_center_x,led_center_y"
    if fit_leds:
        out_str += ",x,y,dx,dy,A,alpha,wx,wy,fit_success,fit_fun,fit_nfev,fit_time"
    out_str += " // all spatial quantities in pixel coordinates\n"
    return out_str
