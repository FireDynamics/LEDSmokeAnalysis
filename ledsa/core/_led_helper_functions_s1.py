import numpy as np


def generate_mask_of_led_areas(image):
    im_mean = np.mean(image)
    im_max = np.max(image)
    th = 0.25 * (im_max - im_mean)
    print("mean pixel value:", im_mean)
    print("max pixel value:", im_max)
    im_set = np.zeros_like(image)
    im_set[image > th] = 1
    return im_set


def find_pos_of_max_col_val_per_area(image, led_mask, skip, window_radius):
    search_areas_list = []
    led_id = 0
    for ix in range(window_radius, image.shape[0] - window_radius, skip):
        for iy in range(window_radius, image.shape[1] - window_radius, skip):
            if led_mask[ix, iy] != 0:
                max_x, max_y = find_led_pos(image, ix, iy, window_radius)
                search_areas_list.append([led_id, max_x, max_y])
                led_id += 1
                remove_led_from_mask(led_mask, ix, iy, window_radius)

                print('.', end='', flush=True)
    search_areas_array = np.array(search_areas_list)
    return search_areas_array


def find_led_pos(image, ix, iy, window_radius):
    s_radius = window_radius // 2
    s = np.index_exp[ix - s_radius:ix + s_radius, iy - s_radius:iy + s_radius]
    res = np.unravel_index(np.argmax(image[s]), image[s].shape)
    max_x = ix - s_radius + res[0]
    max_y = iy - s_radius + res[1]
    return max_x, max_y


def remove_led_from_mask(im_set, ix, iy, window_radius):
    im_set[ix - window_radius:ix + window_radius, iy - window_radius:iy + window_radius] = 0