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
from ledsa.data_extraction.model import target_function
from ledsa.data_extraction.sideband_background import (DEFAULT_SIDEBAND_DISTANCES, DEFAULT_BACKGROUND_FLOOR,
                                                        estimate_sideband_background, estimate_sideband_backgrounds,
                                                        get_sideband_pixels, subtract_pedestal)


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
        data, _, _ = img_channels.channel_data_mask_and_noise(channel)
        analysis_res = _generate_led_analysis_data(conf, channel, data, debug, debug_led, img_filename, 0, search_areas,
                                                   search_area_radius, fit_leds)
        return analysis_res

    sideband_distances, background_floor, sideband_exclusion_margin = get_sideband_parameters(conf)
    sideband_pixels = get_sideband_pixels(search_areas, _get_unmerged_led_arrays(conf, line_indices),
                                          img_channels.shape, search_area_radius, sideband_distances,
                                          sideband_exclusion_margin)

    num_skip_leds = int(conf['analyse_photo']['num_skip_leds'])
    leds = [(int(iled), led_array_idx) for led_array_idx, led_array in enumerate(line_indices)
            for iled in led_array if iled % (num_skip_leds + 1) == 0]
    led_ids = np.array([iled for iled, _ in leds], dtype=int)
    led_array_ids = np.array([led_array_idx for _, led_array_idx in leds], dtype=int)

    # only the pixels of the search areas and side bands are read, which are a few percent of the image
    analysis_pixels = _get_analysis_pixels(search_areas, led_ids, sideband_pixels, img_channels.shape,
                                           search_area_radius)
    data, channel_mask, pixel_noise = img_channels.channel_data_mask_and_noise(channel, analysis_pixels.pixels)
    sidebands = np.s_[analysis_pixels.num_box_pixels:]
    sideband_backgrounds = estimate_sideband_backgrounds(data[sidebands], channel_mask[sidebands],
                                                         analysis_pixels.num_sideband_pixels)

    img_analysis_data = [None] * len(leds)
    for positions, shape, start in analysis_pixels.box_groups:
        boxes = np.s_[start:start + positions.size * shape[0] * shape[1]]
        stack_shape = (positions.size,) + shape
        group_data = _analyse_search_areas(
            conf, channel, img_filename, led_ids[positions], led_array_ids[positions],
            analysis_pixels.centers[positions], search_area_radius, fit_leds,
            data[boxes].reshape(stack_shape), channel_mask[boxes].reshape(stack_shape),
            None if pixel_noise is None else pixel_noise[boxes].reshape(stack_shape),
            sideband_backgrounds[positions], background_floor)
        for position, led_analysis_data in zip(positions, group_data):
            img_analysis_data[position] = led_analysis_data
    return img_analysis_data


class AnalysisPixels(NamedTuple):
    """
    The pixels step 3 reads from an image, in one selection so that the image calibrates them once for all
    channels: the search areas of the analysed LEDs, stacked by their shape, followed by their side bands.
    """

    pixels: Tuple[np.ndarray, np.ndarray]   # row and column indices
    centers: np.ndarray                     # center (row, column) of the search area of every analysed LED
    # per shape of the search areas: the positions of their LEDs in the analysis, the shape and the index of the
    # first pixel of the stack in pixels
    box_groups: List[Tuple[np.ndarray, Tuple[int, int], int]]
    num_box_pixels: int                     # the side band pixels follow the search areas, LED after LED
    num_sideband_pixels: np.ndarray         # number of side band pixels of every analysed LED


# pixels of the analysis, cached per process because they only depend on the LED positions
_analysis_pixels_cache = {}


def _get_analysis_pixels(search_areas: np.ndarray, led_ids: np.ndarray,
                         sideband_pixels: List[Optional[Tuple[np.ndarray, np.ndarray]]],
                         img_shape: Tuple[int, int], search_area_radius: int) -> AnalysisPixels:
    """
    Cached _build_analysis_pixels. The same object is returned for every image and channel, which lets the image
    keep the pixels it calibrated for the next channel.

    :return: See _build_analysis_pixels.
    """
    key = (search_areas.tobytes(), search_areas.shape, led_ids.tobytes(), tuple(img_shape), search_area_radius)
    # the side band pixels come from a cache of their own, which builds a new list when its key changes; the
    # list is kept with the entry, so that its identity cannot pass to another one
    if key not in _analysis_pixels_cache or _analysis_pixels_cache[key][0] is not sideband_pixels:
        _analysis_pixels_cache.clear()
        _analysis_pixels_cache[key] = (sideband_pixels, _build_analysis_pixels(
            search_areas, led_ids, sideband_pixels, img_shape, search_area_radius))
    return _analysis_pixels_cache[key][1]


def _build_analysis_pixels(search_areas: np.ndarray, led_ids: np.ndarray,
                           sideband_pixels: List[Optional[Tuple[np.ndarray, np.ndarray]]],
                           img_shape: Tuple[int, int], search_area_radius: int) -> AnalysisPixels:
    """
    Collect the pixels of the search areas and side bands of the analysed LEDs.

    The search area is cut from the image as data[row - R:row + R, col - R:col + R], so an area at the border
    of the image is cut short as the slice is. Its shape differs from that of the others, and the search areas
    are therefore stacked by their shape.

    :param search_areas: Array with rows (LED id, pixel row, pixel column), the row index equals the LED id.
    :type search_areas: np.ndarray
    :param led_ids: Ids of the analysed LEDs, in the order of the analysis.
    :type led_ids: np.ndarray
    :param sideband_pixels: Per LED id the row and column indices of its side band pixels, see get_sideband_pixels.
    :type sideband_pixels: list[tuple[np.ndarray, np.ndarray] or None]
    :param img_shape: Shape of the image.
    :type img_shape: tuple[int, int]
    :param search_area_radius: Radius R of the search areas.
    :type search_area_radius: int
    :return: The pixels and where the analysis finds the search area and the side bands of every LED in them.
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

    num_sideband_pixels = np.zeros(led_ids.size, dtype=np.intp)
    for position, led_id in enumerate(led_ids):
        if sideband_pixels[led_id] is not None:
            rows.append(sideband_pixels[led_id][0])
            cols.append(sideband_pixels[led_id][1])
            num_sideband_pixels[position] = sideband_pixels[led_id][0].size
    return AnalysisPixels((np.concatenate(rows), np.concatenate(cols)), centers, box_groups, start,
                          num_sideband_pixels)


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
    floor = DEFAULT_BACKGROUND_FLOOR
    exclusion_margin = None
    if conf is not None:
        if conf.get('analyse_photo', 'sideband_distances') != 'None':
            distances = tuple(float(d) for d in conf.get('analyse_photo', 'sideband_distances').split())
            if len(distances) != 2 or not 0 <= distances[0] < distances[1]:
                raise ValueError(f'sideband_distances must be two distances 0 <= inner < outer, got {distances}.')
        if conf.get('analyse_photo', 'background_floor') != 'None':
            floor = float(conf.get('analyse_photo', 'background_floor'))
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
        # read_table stays silent, it would report every file of every image; its error message is unspecific
        try:
            led_arrays.append(read_table(file_path, dtype='int', silent=True))
        except OSError as error:
            raise OSError(f'{file_path} could not be read ({error}). The LED arrays of step 2 are needed to '
                          f'estimate the strip direction of sidebg_sum_col_val, because merge_led_array_indices '
                          f'merges LED arrays that may contain several strips. Run step 2 again to write the '
                          f'files, or unset merge_led_array_indices.') from error
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
                                pixel_noise: Optional[np.ndarray] = None,
                                sideband_pixels: Optional[Tuple[np.ndarray, np.ndarray]] = None,
                                background_floor: float = DEFAULT_BACKGROUND_FLOOR) -> LEDAnalysisData:
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
    :param pixel_noise: Read noise of every pixel, None if the image has no optical black margin. If
        None, the noise of the sum stays NaN.
    :type pixel_noise: np.ndarray or None
    :param channel_mask: Boolean mask of the pixels belonging to the color channel (Bayer pattern for raw files).
        None if all pixels belong to the channel.
    :type channel_mask: np.ndarray or None
    :param sideband_pixels: Row and column indices of the side band pixels of the LED, None if not available.
    :type sideband_pixels: tuple[np.ndarray, np.ndarray] or None
    :param background_floor: Lower limit of the side band and ring corrected sums as fraction of the box sum.
    :type background_floor: float
    :return: Analysis data for the LED.
    :rtype: LEDAnalysisData
    """
    center = np.array([[int(search_areas[iled, 1]), int(search_areas[iled, 2])]])
    search_area = np.index_exp[center[0, 0] - search_area_radius:center[0, 0] + search_area_radius,
                               center[0, 1] - search_area_radius:center[0, 1] + search_area_radius]
    box_mask = np.ones(data[search_area].shape, dtype=bool) if channel_mask is None else channel_mask[search_area]
    led_data, = _analyse_search_areas(conf, channel, img_filename, [iled], [led_array_idx], center,
                                      search_area_radius, fit_leds, data[search_area][None], box_mask[None],
                                      None if pixel_noise is None else pixel_noise[search_area][None],
                                      [estimate_sideband_background(data, channel_mask, sideband_pixels)],
                                      background_floor)
    if debug and fit_leds:
        return led_data.fit_results.x
    return led_data


def _analyse_search_areas(conf: ConfigData, channel: int, img_filename: str, led_ids: Sequence[int],
                          led_array_ids: Sequence[int], centers: np.ndarray, search_area_radius: int, fit_leds: bool,
                          boxes: np.ndarray, box_masks: np.ndarray, box_noise: Optional[np.ndarray],
                          sideband_backgrounds: Sequence[float],
                          background_floor: float = DEFAULT_BACKGROUND_FLOOR) -> List[LEDAnalysisData]:
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
    :param box_masks: Whether each pixel of the search areas belongs to the channel, of the shape of boxes.
    :type box_masks: np.ndarray
    :param box_noise: Read noise of each pixel of the search areas, 0 outside the channel, None if the image has
        no optical black margin. If None, the noise of the sums stays NaN.
    :type box_noise: np.ndarray or None
    :param sideband_backgrounds: Mean pedestal per channel pixel in the side bands of every LED, see
        estimate_sideband_background.
    :type sideband_backgrounds: Sequence[float]
    :param background_floor: Lower limit of the side band and ring corrected sums as fraction of the box sum.
    :type background_floor: float
    :return: Analysis data for every LED.
    :rtype: List[LEDAnalysisData]
    """
    means = np.mean(boxes, axis=(1, 2))
    sums = np.sum(boxes, axis=(1, 2))
    maxima = np.amax(boxes, axis=(1, 2))

    # The median of the border ring, subtracted from every channel pixel of the box. It used to be
    # clipped at zero pixel by pixel, which puts back the offset that the unclipped data remove.
    ring_medians, ring_means = _estimate_ring_backgrounds(boxes, box_masks)
    bgsub_sums = np.sum(boxes.astype(np.float64) - ring_medians[:, None, None] * box_masks, axis=(1, 2))

    # the noise is 0 outside the channel, so the variance of the sum needs no mask
    noise_sums = None if box_noise is None else np.sqrt(np.sum(box_noise.astype(np.float64) ** 2, axis=(1, 2)))

    # the pedestals below are subtracted as N * b with N the number of channel pixels in the box
    num_channel_pixels = np.count_nonzero(box_masks, axis=(1, 2))

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
        led_data.bgsub_sum_color_value = float(bgsub_sums[i])
        if noise_sums is not None:
            led_data.noise_sum_color_value = float(noise_sums[i])

        # border ring as for bgsub, but with the estimator of the side bands: mean over the channel pixels including 0;
        # same lower limit as for the side band correction
        led_data.bgsub2_sum_color_value = subtract_pedestal(float(sums[i]), num_channel_pixels[i] * ring_means[i],
                                                            background_floor)

        # b = mean over the channel pixels in the side bands next to the LED strip
        led_data.sidebg_pedestal = float(num_channel_pixels[i] * sideband_backgrounds[i])
        led_data.sidebg_sum_color_value = subtract_pedestal(float(sums[i]), led_data.sidebg_pedestal,
                                                            background_floor)
        img_analysis_data.append(led_data)
    return img_analysis_data


def _estimate_ring_backgrounds(boxes: np.ndarray, box_masks: Optional[np.ndarray] = None,
                               border_width: int = 2) -> Tuple[np.ndarray, np.ndarray]:
    """
    Estimate the local background of every search area of a stack from the channel pixels of its border ring,
    as their median and as their mean.

    The LED sits at the center of the search area, so the outer border ring samples the scene background
    (stray light, smoke path radiance). Pixels of foreign Bayer colors are excluded through the channel
    mask and no longer by their value: the data are not clipped at zero any more, so a pixel of the
    channel may well be 0 or negative, and dropping those would raise the median.

    The median is subtracted from the box pixel by pixel (bgsub), the mean as N * b from the box sum (N channel
    pixels in the box, bgsub2), which is linear in the pixel values like the side band correction. Note that the
    border ring lies in the halo of the LED for small search areas.

    :param boxes: Search areas of one shape, stacked along the first axis.
    :type boxes: np.ndarray
    :param box_masks: Boolean mask of the channel pixels of the search areas, None if all pixels belong to the
        channel.
    :type box_masks: np.ndarray or None
    :param border_width: Width of the border ring in pixels.
    :type border_width: int
    :return: The median of every search area, 0 if its ring contains no channel pixel, and the mean, NaN then.
    :rtype: tuple[np.ndarray, np.ndarray]
    """
    medians = np.zeros(boxes.shape[0])
    means = np.full(boxes.shape[0], np.nan)
    if boxes.shape[1] <= 2 * border_width or boxes.shape[2] <= 2 * border_width:
        return medians, means
    in_ring = np.ones(boxes.shape, dtype=bool)
    in_ring[:, border_width:-border_width, border_width:-border_width] = False
    if box_masks is not None:
        in_ring &= box_masks
    num_ring_pixels = np.count_nonzero(in_ring, axis=(1, 2))
    # The rings of a Bayer pattern hold the same number of channel pixels wherever the search area lies, so
    # the search areas are grouped by that number and the estimates are taken along the axis of a group.
    for num in np.unique(num_ring_pixels[num_ring_pixels > 0]):
        group = num_ring_pixels == num
        ring_pixels = boxes[group][in_ring[group]].reshape(-1, num)
        medians[group] = np.median(ring_pixels, axis=1)
        means[group] = np.mean(ring_pixels, axis=1, dtype=np.float64)
    return medians, means


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
    out_str += f"time[s] = {img_infos[int(img_id) - 1][3]}, "
    out_str += f"LEDSA {version('ledsa')}\n"
    out_str += "# id,line,sum_col_value,average_col_value,max_col_value,bgsub_sum_col_value"
    out_str += ",sidebg_sum_col_value,sidebg_pedestal,bgsub2_sum_col_value"
    out_str += ",noise_sum_col_value"
    if fit_leds:
        out_str += ",led_center_x, led_center_y"
        out_str += ",x,y,dx,dy,A,alpha,wx,wy,fit_success,fit_fun,fit_nfev,fit_time"
        out_str += "// all spatial quantities in pixel coordinates\n"
    else:
        out_str += "\n"
    return out_str
