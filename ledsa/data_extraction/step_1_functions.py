import numpy as np
from matplotlib import pyplot as plt

from ledsa.core.ConfigData import ConfigData


def find_search_areas(image: np.ndarray, window_radius=10, skip=10, threshold_factor=0.25) -> np.ndarray:
    print('finding led search areas')
    led_mask = _generate_mask_of_led_areas(image, threshold_factor)
    search_areas = _find_pos_of_max_col_val_per_area(image, led_mask, skip, window_radius)
    print("\nfound {} leds".format(search_areas.shape[0]))
    return search_areas


def add_search_areas_to_plot(search_areas: np.ndarray, ax: plt.axes, config: ConfigData) -> None:
    for i in range(search_areas.shape[0]):
        ax.add_patch(plt.Circle((search_areas[i, 2], search_areas[i, 1]),
                                radius=int(config['window_radius']),
                                color='Red', fill=False, alpha=0.25,
                                linewidth=0.1))
        ax.text(search_areas[i, 2] + int(config['window_radius']),
                search_areas[i, 1] + int(config['window_radius']) // 2,
                '{}'.format(search_areas[i, 0]), fontsize=1)


def _generate_mask_of_led_areas(image, threshold_factor):
    im_mean = np.mean(image)
    im_max = np.max(image)
    th = threshold_factor * (im_max - im_mean)
    print("mean pixel value:", im_mean)
    print("max pixel value:", im_max)
    im_set = np.zeros_like(image)
    im_set[image > th] = 1
    return im_set


def _find_pos_of_max_col_val_per_area(image, led_mask, skip, window_radius):
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


def _find_led_pos(image, ix, iy, window_radius):
    s_radius = window_radius // 2
    s = np.index_exp[ix - s_radius:ix + s_radius, iy - s_radius:iy + s_radius]
    res = np.unravel_index(np.argmax(image[s]), image[s].shape)
    max_x = ix - s_radius + res[0]
    max_y = iy - s_radius + res[1]
    return max_x, max_y


def _remove_led_from_mask(im_set, ix, iy, window_radius):
    im_set[ix - window_radius:ix + window_radius, iy - window_radius:iy + window_radius] = 0
