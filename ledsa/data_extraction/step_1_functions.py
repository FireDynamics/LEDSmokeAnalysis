from typing import Tuple

import numpy as np
from matplotlib import pyplot as plt

from ledsa.core.ConfigData import ConfigData


def find_search_areas(image: np.ndarray, window_radius=10, skip=10, threshold_factor=0.25) -> np.ndarray:
    """
    Find the search areas of the LEDs in a given image.

    :param image: A 2D numpy array representing the image.
    :type image: numpy.ndarray
    :param window_radius: The radius of the search area around each LED, defaults to 10.
    :type window_radius: int, optional
    :param skip: The distance to skip between individual LEDs while scanning the image, defaults to 10.
    :type skip: int, optional
    :param threshold_factor: A factor to calculate the pixel value threshold for identifying LEDs, defaults to 0.25.
    :type threshold_factor: float, optional
    :return: A numpy array containing the search areas for LEDs.
    :rtype: numpy.ndarray
    """
    print('finding led search areas')
    led_mask = _generate_mask_of_led_areas(image, threshold_factor)
    search_areas = _find_pos_of_max_col_val_per_area(image, led_mask, skip, window_radius)
    print("\nfound {} leds".format(search_areas.shape[0]))
    return search_areas


def add_search_areas_to_plot(search_areas: np.ndarray, ax: plt.axes, config: ConfigData) -> None:
    """
    Add search areas as red circles and LED IDs to a given matplotlib axis.

    :param search_areas: A numpy array containing LED search areas.
    :type search_areas: numpy.ndarray
    :param ax: A matplotlib axis where search areas should be plotted.
    :type ax: plt.axes
    :param config: An instance of ConfigData containing the configuration data.
    :type config: ConfigData
    """
    for i in range(search_areas.shape[0]):
        ax.add_patch(plt.Circle((search_areas[i, 2], search_areas[i, 1]),
                                radius=int(config['window_radius']),
                                color='Red', fill=False, alpha=0.25,
                                linewidth=0.1))
        ax.text(search_areas[i, 2] + int(config['window_radius']),
                search_areas[i, 1] + int(config['window_radius']) // 2,
                '{}'.format(search_areas[i, 0]), fontsize=1)


def _generate_mask_of_led_areas(image: np.ndarray, threshold_factor: float) -> np.ndarray:
    """
    Generates a binary mask indicating the potential positions of the search areas.

    :param image: A 2D numpy array representing the image.
    :type image: numpy.ndarray
    :param threshold_factor: A factor to calculate the pixel value threshold for identifying LEDs.
    :type threshold_factor: float
    :return: A binary mask (same size as the image) indicating potential search areas.
    :rtype: numpy.ndarray
    """
    im_mean = np.mean(image)
    im_max = np.max(image)
    th = threshold_factor * (im_max - im_mean)
    print("mean pixel value:", im_mean)
    print("max pixel value:", im_max)
    im_set = np.zeros_like(image)
    im_set[image > th] = 1
    return im_set


def _find_pos_of_max_col_val_per_area(image: np.ndarray, led_mask: np.ndarray, skip: int, window_radius: int) -> np.ndarray:
    """
    Iterates over the given mask to find the brightest pixel within each potential search areas.

    :param image: A 2D numpy array representing the image.
    :type image: numpy.ndarray
    :param led_mask: A binary mask indicating potential search areas.
    :type led_mask: numpy.ndarray
    :param skip: The distance to skip while scanning the image.
    :type skip: int
    :param window_radius: The radius of the search area around each LED center pixel.
    :type window_radius: int
    :return: A numpy array containing the ID and X Y pixel coordinates of each search area.
    :rtype: numpy.ndarray
    """
    search_areas_list = []
    led_id = 0
    for ix in range(window_radius, image.shape[0] - window_radius, skip):
        for iy in range(window_radius, image.shape[1] - window_radius, skip):
            if led_mask[ix, iy] != 0:
                max_x, max_y = _find_led_pos(image, ix, iy, window_radius)
                search_areas_list.append([led_id, max_x, max_y])
                led_id += 1
                _remove_led_from_mask(led_mask, ix, iy, window_radius)

                print('.', end='', flush=True)
    search_areas_array = np.array(search_areas_list)
    return search_areas_array


def _find_led_pos(image: np.ndarray, ix: int, iy: int, window_radius: int) -> Tuple[int, int]:
    """
    Find the position of the LED within a smaller area.

    :param image: A 2D numpy array representing the image.
    :type image: numpy.ndarray
    :param ix: Integer representing the x pixel coordinate on the image around which the window is defined.
    :type ix: int
    :param iy: Integer representing the y pixel coordinate on the image around which the window is defined.
    :type iy: int
    :param window_radius: The radius of the search area.
    :type window_radius: int
    :return: The position of the LED within the smaller area.
    :rtype: tuple
    """
    s_radius = window_radius // 2
    s = np.index_exp[ix - s_radius:ix + s_radius, iy - s_radius:iy + s_radius]
    res = np.unravel_index(np.argmax(image[s]), image[s].shape)
    max_x = ix - s_radius + res[0]
    max_y = iy - s_radius + res[1]
    return max_x, max_y


def _remove_led_from_mask(im_set: np.ndarray, ix: int, iy: int, window_radius: int) -> None:
    """
    Remove the detected LED from the mask to prevent repeated detection.

    :param im_set: The binary mask where LEDs are detected.
    :type im_set: numpy.ndarray
    :param ix: Integer representing the x index around which the window is defined.
    :type ix: int
    :param iy: Integer representing the y index around which the window is defined.
    :type iy: int
    :param window_radius: The radius of the window.
    :type window_radius: int
    """
    im_set[ix - window_radius:ix + window_radius, iy - window_radius:iy + window_radius] = 0
