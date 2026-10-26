"""
Pedestal (additive background) of the LED search areas, estimated in side bands next to the LED strips.

The search area of an LED contains, besides the LED itself, an additive pedestal: stray light, light emitted or
scattered by the smoke and noise clipped at 0 after the black level subtraction of raw images. Under dense smoke the
LED dims while the pedestal remains, so the plain box sum (sum_col_val) overestimates the transmission. The border
ring of the search area (bgsub_sum_col_val) is not suited to measure the pedestal because it lies in the halo of the
LED itself. The side bands used here run parallel to the strip, at a distance of typically 25 to 60 px across the
strip and over +-R along the strip (R = search_area_radius), so they sample the scene next to the LED but not the LED.

The strip direction is estimated locally for each LED from its nearest neighbours in the same LED array, so vertical,
horizontal and oblique strips are supported. Pixels close to any other search area are excluded from the bands, so
neighbouring strips do not enter the estimate.
"""
from typing import List, Optional, Sequence, Tuple

import numpy as np

DEFAULT_SIDEBAND_DISTANCES = (25, 60)
DEFAULT_SIDEBAND_FLOOR = 0.05
NUM_DIRECTION_NEIGHBOURS = 4

# geometry of the side bands, cached per process because it only depends on the LED positions
_sideband_cache = {}


def estimate_strip_directions(search_areas: np.ndarray, led_arrays: Sequence[Sequence[int]],
                              num_neighbours: int = NUM_DIRECTION_NEIGHBOURS) -> np.ndarray:
    """
    Estimate the local direction of the LED strip at each LED from its nearest neighbours in the same LED array.

    The direction is the principal axis of the positions of the LED and its num_neighbours nearest neighbours of the
    same array. It is oriented such that the row component is positive (or, for horizontal strips, the column
    component).

    :param search_areas: Array with rows (LED id, pixel row, pixel column), the row index equals the LED id.
    :type search_areas: np.ndarray
    :param led_arrays: LED ids of each LED array.
    :type led_arrays: Sequence[Sequence[int]]
    :param num_neighbours: Number of neighbours used to estimate the direction.
    :type num_neighbours: int
    :return: Array of shape (number of LEDs, 2) with the unit vector (row, column) along the strip, NaN for LEDs
        that are not part of an array with at least two LEDs.
    :rtype: np.ndarray
    """
    directions = np.full((search_areas.shape[0], 2), np.nan)
    for led_ids in led_arrays:
        led_ids = np.atleast_1d(np.asarray(led_ids, dtype=int))
        if led_ids.size < 2:
            continue
        positions = search_areas[led_ids, 1:3].astype(np.float64)
        num_points = min(num_neighbours, led_ids.size - 1) + 1
        for i, led_id in enumerate(led_ids):
            distances = np.hypot(*(positions - positions[i]).T)
            nearest = np.argsort(distances, kind='stable')[:num_points]
            points = positions[nearest] - positions[nearest].mean(axis=0)
            direction = np.linalg.svd(points, full_matrices=False)[2][0]
            if direction[0] < 0 or (direction[0] == 0 and direction[1] < 0):
                direction = -direction
            directions[led_id] = direction
    return directions


def build_sideband_pixels(search_areas: np.ndarray, directions: np.ndarray, img_shape: Tuple[int, int],
                          search_area_radius: int, distances: Tuple[float, float] = DEFAULT_SIDEBAND_DISTANCES,
                          exclusion_margin: Optional[int] = None) -> List[Optional[Tuple[np.ndarray, np.ndarray]]]:
    """
    Determine the pixels of the two side bands of each LED.

    With t the coordinate along the strip and s the coordinate across the strip, both relative to the center of the
    search area, a pixel belongs to the side bands if ``-R <= t < R`` and ``lo <= abs(s) < hi`` (with the same
    half-open convention as the search area, i.e. ``lo <= s < hi`` or ``-hi <= s < -lo``). For a vertical strip this is
    rows r-R..r+R-1 and columns c+lo..c+hi-1 and c-hi..c-lo-1. Pixels outside the image and pixels within
    R + exclusion_margin (Chebyshev distance) of the center of any other search area are excluded.

    :param search_areas: Array with rows (LED id, pixel row, pixel column), the row index equals the LED id.
    :type search_areas: np.ndarray
    :param directions: Unit vectors along the strip per LED, see estimate_strip_directions.
    :type directions: np.ndarray
    :param img_shape: Shape of the image.
    :type img_shape: tuple[int, int]
    :param search_area_radius: Radius R of the search areas.
    :type search_area_radius: int
    :param distances: Inner and outer distance (lo, hi) of the side bands across the strip in pixels.
    :type distances: tuple[float, float]
    :param exclusion_margin: Margin added to R for the exclusion of other search areas. Defaults to R.
    :type exclusion_margin: int or None
    :return: Per LED id a tuple of row and column indices of the side band pixels, None if the strip direction of
        the LED is unknown.
    :rtype: list[tuple[np.ndarray, np.ndarray] or None]
    """
    radius = int(search_area_radius)
    inner, outer = float(distances[0]), float(distances[1])
    if exclusion_margin is None:
        exclusion_margin = radius
    exclusion = radius + int(exclusion_margin)
    reach = int(np.ceil(np.hypot(outer, radius))) + 1
    d_row, d_col = np.mgrid[-reach:reach + 1, -reach:reach + 1]
    centers = search_areas[:, 1:3].astype(np.int64)
    sideband_pixels = [None] * search_areas.shape[0]
    for led_id in range(search_areas.shape[0]):
        along = directions[led_id]
        if not np.all(np.isfinite(along)):
            continue
        across = np.array([-along[1], along[0]])
        # rounding avoids that pixels exactly on a band border flip due to floating point errors
        t = np.round(d_row * along[0] + d_col * along[1], 6)
        s = np.round(d_row * across[0] + d_col * across[1], 6)
        in_band = (t >= -radius) & (t < radius) & (((s >= inner) & (s < outer)) | ((s >= -outer) & (s < -inner)))
        rows = centers[led_id, 0] + d_row[in_band]
        cols = centers[led_id, 1] + d_col[in_band]
        keep = (rows >= 0) & (rows < img_shape[0]) & (cols >= 0) & (cols < img_shape[1])
        nearby = np.abs(centers - centers[led_id]).max(axis=1) <= reach + exclusion
        nearby[led_id] = False
        for row_other, col_other in centers[nearby]:
            keep &= ~((rows >= row_other - exclusion) & (rows < row_other + exclusion) &
                      (cols >= col_other - exclusion) & (cols < col_other + exclusion))
        sideband_pixels[led_id] = (rows[keep].astype(np.intp), cols[keep].astype(np.intp))
    return sideband_pixels


def get_sideband_pixels(search_areas: np.ndarray, led_arrays: Sequence[Sequence[int]], img_shape: Tuple[int, int],
                        search_area_radius: int, distances: Tuple[float, float] = DEFAULT_SIDEBAND_DISTANCES,
                        exclusion_margin: Optional[int] = None) -> List[Optional[Tuple[np.ndarray, np.ndarray]]]:
    """
    Cached combination of estimate_strip_directions and build_sideband_pixels. The geometry only depends on the
    LED positions and is therefore computed once per process and reused for all images.

    :return: See build_sideband_pixels.
    """
    key = (search_areas.tobytes(), search_areas.shape,
           tuple(tuple(int(i) for i in np.atleast_1d(led_ids)) for led_ids in led_arrays),
           tuple(img_shape), int(search_area_radius), tuple(float(d) for d in distances), exclusion_margin)
    if key not in _sideband_cache:
        _sideband_cache.clear()
        directions = estimate_strip_directions(search_areas, led_arrays)
        _sideband_cache[key] = build_sideband_pixels(search_areas, directions, img_shape, search_area_radius,
                                                     distances, exclusion_margin)
    return _sideband_cache[key]


def estimate_sideband_background(data: np.ndarray, channel_mask: Optional[np.ndarray],
                                 sideband_pixels: Optional[Tuple[np.ndarray, np.ndarray]]) -> float:
    """
    Estimate the pedestal per channel pixel as the mean over the channel pixels in the side bands.

    Pixels of the channel with value 0 are included: the black level subtraction clips noise at 0 and the box sum
    contains these clipped pixels in the same way, so the mean is the matching estimator for the pedestal in the box.
    Foreign Bayer channels are excluded via the channel mask, not via the pixel value.

    :param data: Channel array of the image.
    :type data: np.ndarray
    :param channel_mask: Boolean mask of the channel pixels, None if all pixels belong to the channel.
    :type channel_mask: np.ndarray or None
    :param sideband_pixels: Row and column indices of the side band pixels of the LED.
    :type sideband_pixels: tuple[np.ndarray, np.ndarray] or None
    :return: Mean pedestal per channel pixel, NaN if the side bands contain no channel pixel.
    :rtype: float
    """
    if sideband_pixels is None or sideband_pixels[0].size == 0:
        return np.nan
    values = data[sideband_pixels]
    if channel_mask is not None:
        values = values[channel_mask[sideband_pixels]]
    if values.size == 0:
        return np.nan
    return float(np.mean(values, dtype=np.float64))


def subtract_pedestal(sum_color_value: float, pedestal: float, floor: float = DEFAULT_SIDEBAND_FLOOR) -> float:
    """
    Subtract the pedestal N * B from the box sum S with a lower limit: max(S - N * B, floor * S).

    Without the lower limit the optical depth would be unbounded where the LED signal vanishes in the pedestal;
    values where the limit applies carry no information on the extinction and should be treated as invalid.

    :param sum_color_value: Box sum S.
    :type sum_color_value: float
    :param pedestal: Pedestal N * B of the box.
    :type pedestal: float
    :param floor: Lower limit as fraction of S.
    :type floor: float
    :return: Pedestal corrected box sum, NaN if the pedestal is NaN.
    :rtype: float
    """
    return float(np.maximum(sum_color_value - pedestal, floor * sum_color_value))
