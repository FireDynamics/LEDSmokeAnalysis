import os
import re
import time
from importlib.metadata import version
from typing import List, NamedTuple, Optional, Sequence, Tuple

import numpy as np
import scipy.optimize

from ledsa.core.ConfigData import ConfigData
from ledsa.core.file_handling import read_table
from ledsa.core.image_handling import get_img_name
from ledsa.core.image_reading import ImageChannels
from ledsa.data_extraction.LEDAnalysisData import LEDAnalysisData
from ledsa.data_extraction.farfield_background import (FarfieldGeometry, farfield_pedestals,
                                                       get_farfield_geometry, mean_of_channel_pixels,
                                                       parse_farfield_distances)
from ledsa.data_extraction.model import target_function


def generate_analysis_data(img_filename: str, channel: int, search_areas: np.ndarray, line_indices: List[List[int]],
                           conf: ConfigData, fit_leds=True, debug=False, debug_led=None,
                           img_channels: Optional[ImageChannels] = None) -> List[LEDAnalysisData]:
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
    :param img_channels: The image, already read for the analysis of several channels. If None, the image is
        read from the image directory of the config.
    :type img_channels: Optional[ImageChannels]
    :return: A list of LEDAnalysisData objects containing analysis results.
    :rtype: List[LEDAnalysisData]
    """
    if img_channels is None:
        file_path = os.path.join(conf['DEFAULT']['img_directory'], img_filename)
        img_channels = ImageChannels(file_path, rotation=conf.get('DEFAULT', 'img_rotation'))
    search_area_radius = int(conf['find_search_areas']['search_area_radius'])

    if debug:
        data, _ = img_channels.channel_data_and_mask(channel)
        analysis_res = _generate_led_analysis_data(conf, channel, data, debug, debug_led, img_filename, 0, search_areas,
                                                   search_area_radius, fit_leds)
        return analysis_res

    num_skip_leds = int(conf['analyse_photo']['num_skip_leds'])
    leds = [(int(iled), led_array_idx) for led_array_idx, led_array in enumerate(line_indices)
            for iled in led_array if iled % (num_skip_leds + 1) == 0]
    led_ids = np.array([iled for iled, _ in leds], dtype=int)
    led_array_ids = np.array([led_array_idx for _, led_array_idx in leds], dtype=int)

    farfield = _get_farfield_geometry(conf, search_areas, line_indices, img_channels.shape, search_area_radius)

    # only the pixels of the search areas and of the far-field bands are read, which are a few percent of the image
    analysis_pixels = _get_analysis_pixels(search_areas, led_ids, img_channels.shape, search_area_radius, farfield)
    data, channel_mask = img_channels.channel_data_and_mask(channel, analysis_pixels.pixels)
    pedestals = None
    if farfield is not None:
        bands = np.s_[analysis_pixels.num_box_pixels:]
        band_levels = mean_of_channel_pixels(data[bands], channel_mask[bands],
                                             analysis_pixels.num_band_pixels.ravel()).reshape(-1, 2)
        pedestals = farfield_pedestals(band_levels, farfield.strip_of_led[led_ids],
                                       farfield.position_on_strip[led_ids])

    img_analysis_data = [None] * len(leds)
    for positions, shape, start in analysis_pixels.box_groups:
        boxes = np.s_[start:start + positions.size * shape[0] * shape[1]]
        stack_shape = (positions.size,) + shape
        group_data = _analyse_search_areas(
            conf, channel, img_filename, led_ids[positions], led_array_ids[positions],
            analysis_pixels.centers[positions], search_area_radius, fit_leds, data[boxes].reshape(stack_shape),
            channel_mask[boxes].reshape(stack_shape), None if pedestals is None else pedestals[positions])
        for position, led_analysis_data in zip(positions, group_data):
            img_analysis_data[position] = led_analysis_data
    return img_analysis_data


class AnalysisPixels(NamedTuple):
    """
    The pixels step 3 reads from an image, in one selection so that the image calibrates them once for all
    channels: the search areas of the analysed LEDs, stacked by their shape, followed by their far-field bands
    if the pedestal is estimated.
    """

    pixels: Tuple[np.ndarray, np.ndarray]   # row and column indices
    centers: np.ndarray                     # center (row, column) of the search area of every analysed LED
    # per shape of the search areas: the positions of their LEDs in the analysis, the shape and the index of the
    # first pixel of the stack in pixels
    box_groups: List[Tuple[np.ndarray, Tuple[int, int], int]]
    num_box_pixels: int                     # the band pixels follow the search areas, LED after LED, side after side
    num_band_pixels: np.ndarray             # number of pixels of the two far-field bands of every analysed LED


# pixels of the analysis, cached per process because they only depend on the LED positions
_analysis_pixels_cache = {}


def _get_analysis_pixels(search_areas: np.ndarray, led_ids: np.ndarray, img_shape: Tuple[int, int],
                         search_area_radius: int, farfield: Optional[FarfieldGeometry] = None) -> AnalysisPixels:
    """
    Cached _build_analysis_pixels. The same object is returned for every image and channel, which lets the image
    keep the pixels it calibrated for the next channel.

    :return: See _build_analysis_pixels.
    """
    key = (search_areas.tobytes(), search_areas.shape, led_ids.tobytes(), tuple(img_shape), search_area_radius)
    # the far-field geometry comes from a cache of its own, which builds a new one when its key changes; it is
    # kept with the entry, so that its identity cannot pass to another one
    if key not in _analysis_pixels_cache or _analysis_pixels_cache[key][0] is not farfield:
        _analysis_pixels_cache.clear()
        _analysis_pixels_cache[key] = (farfield, _build_analysis_pixels(search_areas, led_ids, img_shape,
                                                                        search_area_radius, farfield))
    return _analysis_pixels_cache[key][1]


def _build_analysis_pixels(search_areas: np.ndarray, led_ids: np.ndarray, img_shape: Tuple[int, int],
                           search_area_radius: int, farfield: Optional[FarfieldGeometry] = None) -> AnalysisPixels:
    """
    Collect the pixels of the search areas of the analysed LEDs and, if given, of their far-field bands.

    The search area is cut from the image as data[row - R:row + R, col - R:col + R], so an area at the border
    of the image is cut short as the slice is. Its shape differs from that of the others, and the search areas
    are therefore stacked by their shape.

    :param search_areas: Array with rows (LED id, pixel row, pixel column), the row index equals the LED id.
    :type search_areas: np.ndarray
    :param led_ids: Ids of the analysed LEDs, in the order of the analysis.
    :type led_ids: np.ndarray
    :param img_shape: Shape of the image.
    :type img_shape: tuple[int, int]
    :param search_area_radius: Radius R of the search areas.
    :type search_area_radius: int
    :param farfield: The far-field bands of the LEDs, None if the pedestal is not estimated.
    :type farfield: FarfieldGeometry or None
    :return: The pixels and where the analysis finds the search area and the bands of every LED in them.
    :rtype: AnalysisPixels
    """
    centers = search_areas[led_ids, 1:3].astype(int)
    row_ranges = [range(*slice(row - search_area_radius, row + search_area_radius).indices(img_shape[0]))
                  for row in centers[:, 0]]
    col_ranges = [range(*slice(col - search_area_radius, col + search_area_radius).indices(img_shape[1]))
                  for col in centers[:, 1]]
    positions_by_shape = {}
    for position, (row_range, col_range) in enumerate(zip(row_ranges, col_ranges)):
        positions_by_shape.setdefault((len(row_range), len(col_range)), []).append(position)

    rows, cols, box_groups, start = [np.zeros(0, dtype=np.intp)], [np.zeros(0, dtype=np.intp)], [], 0
    for shape, positions in positions_by_shape.items():
        positions = np.array(positions, dtype=int)
        stack_shape = (positions.size,) + shape
        first_rows = np.array([row_ranges[position].start for position in positions], dtype=np.intp)
        first_cols = np.array([col_ranges[position].start for position in positions], dtype=np.intp)
        rows.append(np.broadcast_to(first_rows[:, None, None] + np.arange(shape[0])[:, None], stack_shape).ravel())
        cols.append(np.broadcast_to(first_cols[:, None, None] + np.arange(shape[1]), stack_shape).ravel())
        box_groups.append((positions, shape, start))
        start += positions.size * shape[0] * shape[1]

    num_band_pixels = np.zeros((led_ids.size, 2), dtype=np.intp)
    if farfield is not None:
        for position, led_id in enumerate(led_ids):
            if farfield.pixels[led_id] is None:
                continue
            for side, (band_rows, band_cols) in enumerate(farfield.pixels[led_id]):
                rows.append(band_rows)
                cols.append(band_cols)
                num_band_pixels[position, side] = band_rows.size
    return AnalysisPixels((np.concatenate(rows), np.concatenate(cols)), centers, box_groups, start,
                          num_band_pixels)


def _get_farfield_geometry(conf: ConfigData, search_areas: np.ndarray, line_indices: List[List[int]],
                           img_shape: Tuple[int, int], search_area_radius: int) -> Optional[FarfieldGeometry]:
    """
    The far-field bands of the LEDs if farfield_distances of the config asks for the pedestal. Config files
    written before the option existed do not contain it, the pedestal is not estimated then.

    :param conf: Configuration data.
    :type conf: ConfigData
    :param search_areas: Array with rows (LED id, pixel row, pixel column), the row index equals the LED id.
    :type search_areas: np.ndarray
    :param line_indices: LED ids of the LED arrays used in the analysis.
    :type line_indices: List[List[int]]
    :param img_shape: Shape of the image.
    :type img_shape: tuple[int, int]
    :param search_area_radius: Radius R of the search areas.
    :type search_area_radius: int
    :return: The far-field geometry, None if the pedestal is not estimated.
    :rtype: FarfieldGeometry or None
    """
    distances = parse_farfield_distances(conf.get('analyse_photo', 'farfield_distances', fallback='None'),
                                         search_area_radius)
    if distances is None:
        return None
    return get_farfield_geometry(search_areas, _get_unmerged_led_arrays(conf, line_indices), img_shape,
                                 search_area_radius, distances)


def _get_unmerged_led_arrays(conf: ConfigData, line_indices: List[List[int]]) -> List[List[int]]:
    """
    Return the LED arrays as found in step 2. If LED arrays are merged for the analysis, the merged arrays may
    contain several strips, which would spoil the strip direction and the smoothing along the strip.

    :param conf: Configuration data.
    :type conf: ConfigData
    :param line_indices: LED ids of the (possibly merged) LED arrays used in the analysis.
    :type line_indices: List[List[int]]
    :return: LED ids of the LED arrays as found in step 2.
    :rtype: List[List[int]]
    """
    if conf.get('analyse_positions', 'merge_led_array_indices', fallback='None') == 'None':
        return line_indices
    led_arrays = []
    for i in range(int(conf['analyse_positions']['num_arrays'])):
        file_path = os.path.join('analysis', f'led_array_indices_{i:03}.csv')
        # read_table stays silent, it would report every file of every image; its error message is unspecific
        try:
            led_arrays.append(read_table(file_path, dtype='int', silent=True))
        except OSError as error:
            raise OSError(f'{file_path} could not be read ({error}). The LED arrays of step 2 are needed for '
                          f'the far-field pedestal (farbg_sum_col_val), because merge_led_array_indices merges '
                          f'LED arrays that may contain several strips. Run step 2 again to write the files, '
                          f'or unset merge_led_array_indices or farfield_distances.') from error
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


def find_and_save_not_analysed_imgs(channels: Sequence[int]) -> None:
    """
    Find and save filenames of images that have not yet been analyzed in all of the channels.

    An image is analysed for one channel after the other, so an interrupted run can leave it with the result files
    of only some of its channels. Such an image is analysed again for all of them.

    :param channels: Channels to check for analyzed images.
    :type channels: Sequence[int]
    """
    file_path = os.path.join('analysis', 'image_infos_analysis.csv')
    image_infos = read_table(file_path, dtype='str', delim=',', atleast_2d=True)
    analysed_img_ids = set.intersection(*(set(_find_analysed_img_ids(channel)) for channel in channels))
    remaining_imgs = [img_filename for img_id, img_filename in image_infos[:, :2]
                      if int(img_id) not in analysed_img_ids]

    _save_list_of_remaining_imgs_needed_to_be_processed(remaining_imgs)


def find_fit_leds_for_restart(channels: Sequence[int]) -> bool:
    """
    Find whether the restart of an interrupted step 3 fits the LEDs: if the interrupted run did (-s3), not if it
    did not (-s3_fast). The analysis loads all result files of a channel with the same columns, so the resumed
    images need the columns of the images analysed before. The mode is read from the headers of the result files,
    not from their number of columns, which differs between LEDSA versions. If none of the channels has a result
    file yet, the LEDs are not fitted.

    :param channels: Channels to be resumed.
    :type channels: Sequence[int]
    :return: True if the restart fits the LEDs, False if not.
    :rtype: bool
    :raises ValueError: If the result files of a channel, or those of different channels, were written both with
        and without the fits.
    """
    fit_leds_of_channels = {}
    for channel in channels:
        files_by_fit_leds = {True: [], False: []}
        for _, file_path in _find_result_files(channel):
            has_fits = _result_file_has_fits(file_path)
            if has_fits is not None:
                files_by_fit_leds[has_fits].append(file_path)
        fitted_files, unfitted_files = files_by_fit_leds[True], files_by_fit_leds[False]
        if fitted_files and unfitted_files:
            raise ValueError(f'The result files of channel {channel} were written both with the LED fits (-s3, '
                             f'{len(fitted_files)} files, e.g. {fitted_files[0]}) and without them (-s3_fast, '
                             f'{len(unfitted_files)} files, e.g. {unfitted_files[0]}), so the restart cannot tell '
                             f'in which mode to resume. Delete the files of the mode you do not want and run -re '
                             f'again, or run -s3 or -s3_fast for the channel again.')
        if fitted_files or unfitted_files:
            fit_leds_of_channels[channel] = bool(fitted_files)

    if len(set(fit_leds_of_channels.values())) > 1:
        modes = ', '.join(f'channel {channel} {"with" if fit_leds else "without"}'
                          for channel, fit_leds in fit_leds_of_channels.items())
        raise ValueError(f'The channels were analysed with and without the LED fits ({modes}), but a restart '
                         f'resumes all of its channels in one mode. Restart these channels separately, selecting '
                         f'them with -r, -g or -b.')
    fit_leds = True in fit_leds_of_channels.values()
    if not fit_leds_of_channels:
        print('There are no result files of an interrupted run yet, so step 3 runs without the LED fits, as by '
              '-s3_fast.')
    elif fit_leds:
        print('The result files of the interrupted run have LED fits, so step 3 is resumed with them, as by -s3.')
    else:
        print('The result files of the interrupted run have no LED fits, so step 3 is resumed without them, as by '
              '-s3_fast.')
    return fit_leds


def _generate_led_analysis_data(conf: ConfigData, channel: int, data: np.ndarray, debug: bool, iled: int, img_filename: str, led_array_idx: int, search_areas: np.ndarray, search_area_radius: int, fit_leds: bool = True) -> LEDAnalysisData:
    """
    Generate analysis data for a specific LED from the arrays of the whole image, as _analyse_search_areas does
    for many.

    :param conf: Configuration data.
    :type conf: ConfigData
    :param channel: Color Channel for which analysis should be generated.
    :type channel: int
    :param data: Array representing the image data.
    :type data: np.ndarray
    :param debug: If True, the fit results are returned instead of the analysis data.
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
    :return: Analysis data for the LED.
    :rtype: LEDAnalysisData
    """
    center = np.array([[int(search_areas[iled, 1]), int(search_areas[iled, 2])]])
    search_area = np.index_exp[center[0, 0] - search_area_radius:center[0, 0] + search_area_radius,
                               center[0, 1] - search_area_radius:center[0, 1] + search_area_radius]
    led_data, = _analyse_search_areas(conf, channel, img_filename, [iled], [led_array_idx], center,
                                      search_area_radius, fit_leds, data[search_area][None])
    if debug and fit_leds:
        return led_data.fit_results.x
    return led_data


def _analyse_search_areas(conf: ConfigData, channel: int, img_filename: str, led_ids: Sequence[int],
                          led_array_ids: Sequence[int], centers: np.ndarray, search_area_radius: int, fit_leds: bool,
                          boxes: np.ndarray, box_masks: Optional[np.ndarray] = None,
                          pedestals: Optional[np.ndarray] = None) -> List[LEDAnalysisData]:
    """
    Generate the analysis data of the LEDs of a stack of search areas of one shape.

    Every quantity is taken along the axes of the stack, so that an image of a thousand LEDs needs a few numpy
    calls per quantity instead of a few per LED and quantity. Only the fit is made LED by LED.

    :param conf: Configuration data, used to log irregular fits.
    :type conf: ConfigData
    :param channel: Color channel being analysed.
    :type channel: int
    :param img_filename: Name of the image.
    :type img_filename: str
    :param led_ids: ID of the LED of every search area.
    :type led_ids: Sequence[int]
    :param led_array_ids: Index of the LED array of every LED.
    :type led_array_ids: Sequence[int]
    :param centers: Center (row, column) of every search area in the image.
    :type centers: np.ndarray
    :param search_area_radius: Radius of the search areas.
    :type search_area_radius: int
    :param fit_leds: If True, every LED is fitted to a model function.
    :type fit_leds: bool
    :param boxes: Channel values of the search areas, 0 outside the channel, of shape (number of LEDs, rows,
        columns).
    :type boxes: np.ndarray
    :param box_masks: Whether each pixel of the search areas belongs to the channel, of the shape of boxes. Only
        needed with pedestals.
    :type box_masks: np.ndarray or None
    :param pedestals: Far-field pedestal per channel pixel of every LED, NaN for an LED without one, see
        farfield_pedestals. None if the pedestal is not estimated.
    :type pedestals: np.ndarray or None
    :return: Analysis data for every LED.
    :rtype: List[LEDAnalysisData]
    """
    means = np.mean(boxes, axis=(1, 2))
    sums = np.sum(boxes, axis=(1, 2))
    maxima = np.amax(boxes, axis=(1, 2))
    if pedestals is not None:
        # the pedestal of the search area is N * b with N its number of channel pixels; an LED without a far
        # field keeps its plain sum, so that the corrected sum is a number for every LED
        box_pedestals = np.sum(box_masks, axis=(1, 2)) * np.where(np.isfinite(pedestals), pedestals, 0.0)

    img_analysis_data = []
    for i, led_id in enumerate(led_ids):
        led_data = LEDAnalysisData(led_id, int(led_array_ids[i]), fit_leds)
        center_search_area_x, center_search_area_y = int(centers[i, 0]), int(centers[i, 1])
        if fit_leds:
            start_time = time.process_time()
            led_data.fit_results, mesh = _fit_model_to_led(boxes[i])
            end_time = time.process_time()
            led_data.fit_time = end_time - start_time
            led_data.led_center_x = led_data.fit_results.x[0] + center_search_area_x - search_area_radius
            led_data.led_center_y = led_data.fit_results.x[1] + center_search_area_y - search_area_radius
            if not led_data.fit_results.success:  # A > 255 or A < 0:
                _log_warnings(img_filename, channel, led_data, center_search_area_x, center_search_area_y,
                              boxes[i].shape, search_area_radius, conf)

        led_data.mean_color_value = means[i]
        led_data.sum_color_value = sums[i]
        led_data.max_color_value = maxima[i]
        if pedestals is not None:
            led_data.farbg_pedestal = float(box_pedestals[i])
            led_data.farbg_sum_color_value = float(sums[i]) - led_data.farbg_pedestal
        img_analysis_data.append(led_data)
    return img_analysis_data


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
    header = _create_header(channel, img_id, img_filename, img_infos, basename, img_data[0].fit_leds,
                            img_data[0].farbg_sum_color_value is not None)
    out_file.write(header)
    for led_data in img_data:
        out_file.write(str(led_data))
    out_file.close()


def _find_analysed_img_ids(channel):
    """
    Find the IDs of the images that have a result file in a channel.

    :param channel: Channel to check for analyzed images.
    :return: List of IDs of analyzed images.
    :rtype: List[int]
    """
    return [img_id for img_id, _ in _find_result_files(channel)]


def _find_result_files(channel: int) -> List[Tuple[int, str]]:
    """
    Find the result files of step 3 in a channel.

    :param channel: Channel whose result files are searched.
    :type channel: int
    :return: Image ID and path of every result file, sorted by image ID. Empty if the channel has no directory yet.
    :rtype: List[Tuple[int, str]]
    """
    dir_path = os.path.join('analysis', f'channel{channel}')
    if not os.path.isdir(dir_path):
        return []
    result_files = []
    for file_name in os.listdir(dir_path):
        img = re.search(r"(\d+)_led_positions.csv", file_name)
        if img is not None:
            result_files.append((int(img.group(1)), os.path.join(dir_path, file_name)))
    return sorted(result_files)


def _result_file_has_fits(file_path: str) -> Optional[bool]:
    """
    Tell from its header whether a result file of step 3 was written with the LED fits.

    Only a file written with the fits names their columns in the header line of the column names, see
    _create_header. fit_success is one of them in every version of the header.

    :param file_path: Path of the result file.
    :type file_path: str
    :return: True if the file was written with the fits, False if without them, None if it has no header line of
        the column names, for example because it is empty.
    :rtype: Optional[bool]
    """
    # only the column names are of interest, a directory or image name in another encoding must not stop the restart
    with open(file_path, 'r', encoding='utf-8', errors='replace') as result_file:
        for line in result_file:
            if not line.startswith('#'):
                break
            # the fit columns are followed by a comment after //
            column_names = [name.strip() for name in line.lstrip('#').split('//')[0].split(',')]
            if column_names[0] == 'id':
                return 'fit_success' in column_names
    return None


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


def _create_header(channel: int, img_id: str, img_filename: str, img_infos: np.ndarray, basename: str, fit_leds: str,
                   farfield: bool = False):
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
    :param farfield: If True, includes the far-field pedestal columns in the header.
    :type farfield: bool
    :return: String of the header.
    :rtype: str
    """
    out_str = f'# image root = {basename}, photo file name = {img_filename}, '
    out_str += f"channel = {channel}, "
    out_str += f"time[s] = {img_infos[int(img_id) - 1][3]}, "
    out_str += f"LEDSA {version('ledsa')}\n"
    out_str += "# id,line,sum_col_value,average_col_value,max_col_value"
    if farfield:
        out_str += ",farbg_sum_col_value,farbg_pedestal"
    if fit_leds:
        out_str += ",led_center_x, led_center_y"
        out_str += ",x,y,dx,dy,A,alpha,wx,wy,fit_success,fit_fun,fit_nfev,fit_time"
        out_str += "// all spatial quantities in pixel coordinates\n"
    else:
        out_str += "\n"
    return out_str
