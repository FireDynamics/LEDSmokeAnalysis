"""
Pedestal of the LED search areas, estimated in the far field of the LED strips.

The sum over a search area (sum_col_val) holds, besides the light of the LED, light that does not dim with the
LED: the dark level of the sensor where it differs from the optical black the black level is taken from, ambient
light, and light that smoke scatters into the line of sight. In dense smoke the LED dims while this pedestal
stays, so the plain sum overestimates the transmission and the extinction coefficients saturate.

The pedestal is estimated for every image where no light of an LED is expected, in the far field of the strips:

1. Two bands run parallel to the strip, one on each side, over +-R along the strip (R = search_area_radius) and
   from the inner to the outer far-field distance across it. Pixels closer than the inner distance to any LED of
   any strip are left out, so neighbouring strips do not enter the bands.
2. The level of a band is the mean of its pixels of the color channel.
3. The levels of each side are smoothed along the strip, with a running median and a running mean over the LEDs.
   The pedestal varies slowly along the strip, so this removes the noise of the estimate without changing it:
   the pedestal corrected sum is as noisy as the plain sum.
4. The pedestal of an LED is the lower of the levels of its two sides. An object in one of the bands, another
   light source or the brighter side of a gradient can only raise a level, so the lower side is the safe one.
5. N times this pedestal is subtracted from the sum of the search area, with N the number of pixels of the color
   channel in the search area.

Why the far field and not the neighbourhood of the LED: around an LED there is light of the LED itself, the wings
of its image and, behind a window or in hot gases, a halo that can reach a hundred pixels and more. A pedestal
measured there subtracts transmitted light and raises the extinction coefficients, the more the denser the smoke.
The far field holds none of it. Stray light that peaks at the strip is brighter at the LED than in the far field, so
the far-field level is a lower bound of the pedestal: the corrected sum lies between the plain sum and the light
of the LED in the search area. The bound holds for the dark level and for scattered light. Ambient light on a
background that is brighter beside the strip than behind the LED would raise the far-field level above the
pedestal of the search area.

What the correction does not do: where the image of an LED widens beyond its search area, the search area loses
light of the LED. That error has the opposite sign and is no pedestal; the correction leaves it as it is.
"""
from typing import List, NamedTuple, Optional, Sequence, Tuple

import numpy as np
from scipy.spatial import cKDTree

# LEDs the levels are smoothed over along the strip: running median, then running mean
SMOOTHING_MEDIAN_LEDS = 15
SMOOTHING_MEAN_LEDS = 5
NUM_DIRECTION_NEIGHBOURS = 4


class FarfieldGeometry(NamedTuple):
    """The far-field bands of the LEDs of an image and the place of every LED on its strip."""

    # per LED id the row and column indices of the pixels of the band on each of the two sides of the strip,
    # None for an LED without a strip direction
    pixels: List[Optional[Tuple[Tuple[np.ndarray, np.ndarray], Tuple[np.ndarray, np.ndarray]]]]
    strip_of_led: np.ndarray        # per LED id the index of its strip, -1 for an LED on no strip
    position_on_strip: np.ndarray   # per LED id its coordinate along the strip in pixels


# geometry of the bands, cached per process because it only depends on the LED positions
_farfield_cache = {}


def parse_farfield_distances(value: Optional[str], search_area_radius: int) -> Optional[Tuple[float, float]]:
    """
    Read the far-field distances from their config value.

    :param value: Value of farfield_distances: the inner and the outer distance of the bands across the strip in
        pixels, or None.
    :type value: str or None
    :param search_area_radius: Radius R of the search areas.
    :type search_area_radius: int
    :return: The distances (inner, outer), None if the pedestal is not estimated.
    :rtype: tuple[float, float] or None
    :raises ValueError: If the value is neither None nor two distances R <= inner < outer.
    """
    if value is None or str(value).strip().lower() in ('none', ''):
        return None
    try:
        distances = tuple(float(distance) for distance in str(value).split())
    except ValueError:
        distances = ()
    if len(distances) != 2 or not search_area_radius <= distances[0] < distances[1]:
        raise ValueError(f"farfield_distances of config.ini holds '{value}', while it takes the inner and the "
                         f"outer distance of the far-field bands across the LED strip in pixels, with "
                         f"search_area_radius ({search_area_radius}) <= inner < outer, or None.")
    return distances


def estimate_strip_directions(search_areas: np.ndarray, led_arrays: Sequence[Sequence[int]],
                              num_neighbours: int = NUM_DIRECTION_NEIGHBOURS) -> np.ndarray:
    """
    Estimate the local direction of the LED strip at each LED from its nearest neighbours in the same LED array.

    The direction is the principal axis of the positions of the LED and its num_neighbours nearest neighbours of the
    same array. It is oriented such that the row component is positive (or, for horizontal strips, the column
    component), so the two sides of a strip are the same for all of its LEDs.

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


def build_farfield_geometry(search_areas: np.ndarray, led_arrays: Sequence[Sequence[int]],
                            img_shape: Tuple[int, int], search_area_radius: int,
                            distances: Tuple[float, float]) -> FarfieldGeometry:
    """
    Determine the pixels of the two far-field bands of each LED and the place of each LED on its strip.

    With t the coordinate along the strip and s the coordinate across it, both relative to the center of the
    search area, a pixel belongs to the band of one side if ``-R <= t < R`` and ``inner <= s < outer``, and to that
    of the other side if ``-outer <= s < -inner``. For a vertical strip these are the rows r-R..r+R-1 and the
    columns c+inner..c+outer-1 and c-outer..c-inner-1. Pixels outside the image and pixels closer than the inner
    distance to the center of any search area are left out.

    :param search_areas: Array with rows (LED id, pixel row, pixel column), the row index equals the LED id.
    :type search_areas: np.ndarray
    :param led_arrays: LED ids of each LED strip, as found in step 2.
    :type led_arrays: Sequence[Sequence[int]]
    :param img_shape: Shape of the image.
    :type img_shape: tuple[int, int]
    :param search_area_radius: Radius R of the search areas.
    :type search_area_radius: int
    :param distances: Inner and outer distance of the bands across the strip in pixels.
    :type distances: tuple[float, float]
    :return: The band pixels and the place on its strip of every LED.
    :rtype: FarfieldGeometry
    """
    radius = int(search_area_radius)
    inner, outer = float(distances[0]), float(distances[1])
    directions = estimate_strip_directions(search_areas, led_arrays)
    centers = search_areas[:, 1:3].astype(np.float64)
    tree = cKDTree(centers)
    reach = int(np.ceil(np.hypot(outer, radius))) + 1
    d_row, d_col = np.mgrid[-reach:reach + 1, -reach:reach + 1]

    num_leds = search_areas.shape[0]
    pixels = [None] * num_leds
    strip_of_led = np.full(num_leds, -1, dtype=int)
    position_on_strip = np.zeros(num_leds)
    for strip, led_ids in enumerate(led_arrays):
        led_ids = np.atleast_1d(np.asarray(led_ids, dtype=int))
        if led_ids.size < 2:
            continue
        # one direction for the coordinate along the strip, so that it orders the LEDs of a curved strip as well
        mean_direction = np.nanmean(directions[led_ids], axis=0)
        strip_of_led[led_ids] = strip
        position_on_strip[led_ids] = centers[led_ids] @ mean_direction
    for led_id in range(num_leds):
        along = directions[led_id]
        if not np.all(np.isfinite(along)):
            continue
        across = np.array([-along[1], along[0]])
        # rounding avoids that pixels exactly on a band border flip due to floating point errors
        t = np.round(d_row * along[0] + d_col * along[1], 6)
        s = np.round(d_row * across[0] + d_col * across[1], 6)
        in_window = (t >= -radius) & (t < radius)
        sides = []
        for in_band in (in_window & (s >= inner) & (s < outer), in_window & (s >= -outer) & (s < -inner)):
            rows = int(centers[led_id, 0]) + d_row[in_band]
            cols = int(centers[led_id, 1]) + d_col[in_band]
            keep = (rows >= 0) & (rows < img_shape[0]) & (cols >= 0) & (cols < img_shape[1])
            rows, cols = rows[keep], cols[keep]
            if rows.size:
                distance_to_nearest_led, _ = tree.query(np.column_stack((rows, cols)))
                keep = distance_to_nearest_led >= inner
                rows, cols = rows[keep], cols[keep]
            sides.append((rows.astype(np.intp), cols.astype(np.intp)))
        pixels[led_id] = (sides[0], sides[1])
    return FarfieldGeometry(pixels, strip_of_led, position_on_strip)


def get_farfield_geometry(search_areas: np.ndarray, led_arrays: Sequence[Sequence[int]], img_shape: Tuple[int, int],
                          search_area_radius: int, distances: Tuple[float, float]) -> FarfieldGeometry:
    """
    Cached build_farfield_geometry. The geometry only depends on the LED positions and is therefore computed once
    per process and reused for all images.

    :return: See build_farfield_geometry.
    """
    key = (search_areas.tobytes(), search_areas.shape,
           tuple(tuple(int(i) for i in np.atleast_1d(led_ids)) for led_ids in led_arrays),
           tuple(img_shape), int(search_area_radius), tuple(float(d) for d in distances))
    if key not in _farfield_cache:
        _farfield_cache.clear()
        _farfield_cache[key] = build_farfield_geometry(search_areas, led_arrays, img_shape, search_area_radius,
                                                       distances)
    return _farfield_cache[key]


def mean_of_channel_pixels(values: np.ndarray, channel_mask: np.ndarray, num_pixels: Sequence[int]) -> np.ndarray:
    """
    Mean over the pixels of the color channel in every band, from the pixels of all bands.

    :param values: Channel values of the pixels of all bands, the pixels of one band after those of the previous
        one.
    :type values: np.ndarray
    :param channel_mask: Whether each of these pixels belongs to the channel.
    :type channel_mask: np.ndarray
    :param num_pixels: Number of pixels of every band.
    :type num_pixels: Sequence[int]
    :return: Mean per channel pixel of every band, NaN for a band without a pixel of the channel.
    :rtype: np.ndarray
    """
    num_pixels = np.asarray(num_pixels, dtype=np.intp)
    means = np.full(num_pixels.size, np.nan)
    # reduceat cannot sum an empty segment, and the segments left are contiguous without the empty ones
    has_pixels = num_pixels > 0
    if not has_pixels.any():
        return means
    starts = (np.cumsum(num_pixels) - num_pixels)[has_pixels]
    sums = np.add.reduceat(np.where(channel_mask, values, 0), starts, dtype=np.float64)
    counts = np.add.reduceat(channel_mask, starts, dtype=np.intp)
    with np.errstate(invalid='ignore', divide='ignore'):
        means[has_pixels] = sums / counts  # 0 / 0 is NaN
    return means


def _running(values: np.ndarray, size: int, function) -> np.ndarray:
    """A centred running statistic that ignores NaN and takes fewer values at the ends, NaN where it has none."""
    half = size // 2
    padded = np.concatenate((np.full(half, np.nan), values, np.full(half, np.nan)))
    windows = np.lib.stride_tricks.sliding_window_view(padded, 2 * half + 1)
    result = np.full(values.size, np.nan)
    has_values = np.isfinite(windows).any(axis=1)
    result[has_values] = function(windows[has_values], axis=1)
    return result


def smooth_along_strip(levels: np.ndarray, median_size: int = SMOOTHING_MEDIAN_LEDS,
                       mean_size: int = SMOOTHING_MEAN_LEDS) -> np.ndarray:
    """
    Smooth the band levels of one side of a strip along the strip: a running median, which drops the outliers of
    single LEDs, followed by a running mean. An LED without a level takes the one of its neighbours. At the ends
    of the strip the windows are cut short.

    :param levels: Level of every LED of the strip, in their order along the strip, NaN for an LED without one.
    :type levels: np.ndarray
    :param median_size: Number of LEDs of the running median.
    :type median_size: int
    :param mean_size: Number of LEDs of the running mean.
    :type mean_size: int
    :return: The smoothed levels, NaN where no LED within reach has a level.
    :rtype: np.ndarray
    """
    levels = np.asarray(levels, dtype=np.float64)
    if levels.size == 0:
        return levels
    return _running(_running(levels, median_size, np.nanmedian), mean_size, np.nanmean)


def farfield_pedestals(band_levels: np.ndarray, strip_of_led: np.ndarray,
                       position_on_strip: np.ndarray) -> np.ndarray:
    """
    The pedestal per channel pixel of every LED: the levels of the two sides, smoothed along the strip, and of
    them the lower one.

    :param band_levels: Level of the band on each of the two sides of every LED, shape (number of LEDs, 2), NaN
        for a band without a pixel of the channel.
    :type band_levels: np.ndarray
    :param strip_of_led: Index of the strip of every LED, -1 for an LED on no strip.
    :type strip_of_led: np.ndarray
    :param position_on_strip: Coordinate of every LED along its strip.
    :type position_on_strip: np.ndarray
    :return: Pedestal per channel pixel of every LED, NaN for an LED without a level on either side.
    :rtype: np.ndarray
    """
    band_levels = np.asarray(band_levels, dtype=np.float64)
    pedestals = np.full(band_levels.shape[0], np.nan)
    for strip in np.unique(strip_of_led[strip_of_led >= 0]):
        leds = np.flatnonzero(strip_of_led == strip)
        leds = leds[np.argsort(position_on_strip[leds], kind='stable')]
        smoothed = np.column_stack([smooth_along_strip(band_levels[leds, side]) for side in (0, 1)])
        # fmin takes the side that has a level where the other has none
        pedestals[leds] = np.fmin(smoothed[:, 0], smoothed[:, 1])
    return pedestals
