import numpy as np
from matplotlib import pyplot as plt
import cv2


def find_search_areas(image: np.ndarray, search_area_radius, pixel_value_percentile=99.875, max_n_leds=1300) -> np.ndarray:
    """
    Identifies and extracts locations of LEDs in an image.

    :param image: The input image in which LEDs are to be searched. Expected to be a grayscale image.
    :type image: np.ndarray
    :param search_area_radius: The radius of the square area around each identified LED location.
    :type search_area_radius: int
    :param pixel_value_percentile: The percentile value to determine the brightness threshold for LED detection.
    :type pixel_value_percentile: float
    :param max_n_leds: The maximum number of LED locations to identify in the image.
    :type max_n_leds: int
    :return: A numpy array of identified LED locations, each represented as (LED ID, y-coordinate, x-coordinate).
    :rtype: np.ndarray
    """
    (_, max_pixel_value, _, max_pixel_loc) = cv2.minMaxLoc(image)
    threshold = np.percentile(image, pixel_value_percentile)
    search_areas_list = []
    print("Threshold pixel value:", threshold)
    print("Searching LEDs")
    led_id = 0
    image = image.copy()
    while max_pixel_value > threshold and led_id < max_n_leds:
        (_, max_pixel_value, _, max_pixel_loc) = cv2.minMaxLoc(image)
        if max_pixel_value > threshold:
            image[max_pixel_loc[1] - search_area_radius: max_pixel_loc[1] + search_area_radius, max_pixel_loc[0] - search_area_radius: max_pixel_loc[0] + search_area_radius] = 0
            search_areas_list.append((led_id, max_pixel_loc[1], max_pixel_loc[0]))
            print('.', end='', flush=True)
            led_id += 1
    print('\n')
    print(f"Found {led_id} LEDS")
    return np.array(search_areas_list)


def add_search_areas_to_plot(search_areas: np.ndarray, search_area_radius: int, ax: plt.axes) -> None:
    """
    Add search areas as red circles and LED IDs to a given matplotlib axis.

    :param search_areas: A numpy array containing LED search areas.
    :type search_areas: numpy.ndarray
    :param search_area_radius: The radius of the search areas to be displayed
    :type search_area_radius: int
    :param ax: A matplotlib axis where search areas should be plotted.
    :type ax: plt.axes
    """
    for i in range(search_areas.shape[0]):
        ax.add_patch(plt.Circle((search_areas[i, 2], search_areas[i, 1]),
                                radius=int(search_area_radius),
                                color='Red', fill=False, alpha=0.25,
                                linewidth=0.1))
        ax.text(search_areas[i, 2] + int(search_area_radius),
                search_areas[i, 1] + int(search_area_radius) // 2,
                '{}'.format(search_areas[i, 0]), fontsize=1)