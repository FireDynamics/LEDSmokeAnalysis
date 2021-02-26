import numpy as np


def get_indices_of_outer_leds(config):
    if config['analyse_positions']['line_edge_indices'] == 'None':
        config.in_line_edge_indices()
        with open('config.ini', 'w') as configfile:
            config.write(configfile)
    line_edge_indices = config.get2dnparray('analyse_positions', 'line_edge_indices')
    # makes sure that line_edge_indices is a 2d list
    if len(line_edge_indices.shape) == 1:
        line_edge_indices = [line_edge_indices]
    return line_edge_indices


def calc_dists_between_led_arrays_and_search_areas(line_edge_indices, search_areas):
    distances_led_arrays_search_areas = np.zeros((search_areas.shape[0], len(line_edge_indices)))

    for line_edge_idx in range(len(line_edge_indices)):
        i1 = line_edge_indices[line_edge_idx][0]
        i2 = line_edge_indices[line_edge_idx][1]

        corner1 = np.array(search_areas[i1, 1:])
        corner2 = np.array(search_areas[i2, 1:])

        d = calc_dists_to_line_segment(search_areas[:, 1:], corner1, corner2)

        distances_led_arrays_search_areas[:, line_edge_idx] = d
    return distances_led_arrays_search_areas


def calc_dists_to_line_segment(points: np.ndarray, c1: np.ndarray, c2: np.ndarray) -> np.ndarray:
    d = np.zeros(points.shape[0])

    for led_id in range(points.shape[0]):
        led = points[led_id]

        dist_led_c1 = np.linalg.norm(led - c1)
        dist_led_c2 = np.linalg.norm(led - c2)
        segment_len = np.linalg.norm(c2 - c1)

        if np.all(c1 == led) or np.all(c2 == led):
            d[led_id] = 0
            continue

        if all(c1 == c2):
            d[led_id] = dist_led_c1
            continue

        led_on_wrong_side_of_c1 = np.arccos(np.dot((led - c1) / dist_led_c1, (c2 - c1) / segment_len)) > np.pi / 2
        if led_on_wrong_side_of_c1:
            d[led_id] = dist_led_c1
            continue

        led_on_wrong_side_of_c2 = np.arccos(np.dot((led - c2) / dist_led_c2, (c1 - c2) / segment_len)) > np.pi / 2
        if led_on_wrong_side_of_c2:
            d[led_id] = dist_led_c2
            continue

        d[led_id] = np.abs(np.cross(c1 - c2, c1 - led)) / segment_len
    return d


def match_leds_to_arrays_with_min_dist(dists_led_arrays_search_areas, edge_indices, config, search_areas):
    ignore_indices = get_indices_of_ignored_leds(config)
    # construct 2D list for LED indices sorted by line
    led_arrays = []
    for edge_idx in edge_indices:
        led_arrays.append([])

    for iled in range(search_areas.shape[0]):
        if iled in ignore_indices:
            continue

        idx_nearest_array = np.argmin(dists_led_arrays_search_areas[iled, :])
        led_arrays[idx_nearest_array].append(iled)
    return led_arrays


def get_indices_of_ignored_leds(config):
    if config['analyse_positions']['ignore_indices'] != 'None':
        ignore_indices = np.array([int(i) for i in config['analyse_positions']['ignore_indices'].split(' ')])
    else:
        ignore_indices = np.array([])
    return ignore_indices
