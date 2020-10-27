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

    xs = search_areas[:, 1]
    ys = search_areas[:, 2]

    for line_edge_idx in range(len(line_edge_indices)):
        i1 = line_edge_indices[line_edge_idx][0]
        i2 = line_edge_indices[line_edge_idx][1]

        p1x = xs[i1]
        p1y = ys[i1]
        p2x = xs[i2]
        p2y = ys[i2]

        pd = np.sqrt((p1x - p2x) ** 2 + (p1y - p2y) ** 2)
        d = np.abs(((p2y - p1y) * xs - (p2x - p1x) * ys
                    + p2x * p1y - p2y * p1x) / pd)

        distances_led_arrays_search_areas[:, line_edge_idx] = d
    return distances_led_arrays_search_areas


def match_leds_to_arrays_with_min_dist(dists_led_arrays_search_areas, edge_indices, config, search_areas):
    ignore_indices = get_indices_of_ignored_leds(config)

    xs = search_areas[:, 1]
    ys = search_areas[:, 2]

    num_leds = search_areas.shape[0]

    # construct 2D array for LED indices sorted by line
    led_arrays = []
    for edge_idx in edge_indices:
        led_arrays.append([])

    for iled in range(num_leds):
        if iled in ignore_indices:
            continue

        idx_nearest_array = np.argmin(dists_led_arrays_search_areas[iled, :])
        # TODO: ask Lukas for need of following code

        # for il_repeat in range(len(edge_indices)):
        #     i1 = edge_indices[idx_nearest_array][0]
        #     i2 = edge_indices[idx_nearest_array][1]
        #
        #     x_outer_led1 = xs[i1]
        #     y_outer_led1 = ys[i1]
        #     x_outer_led2 = xs[i2]
        #     y_outer_led2 = ys[i2]
        #
        #     x_led = xs[iled]
        #     y_led = ys[iled]
        #
        #     dist_led_outer_led1 = np.sqrt((x_outer_led1 - x_led) ** 2 + (y_outer_led1 - y_led) ** 2)
        #     dist_led_outer_led2 = np.sqrt((x_outer_led2 - x_led) ** 2 + (y_outer_led2 - y_led) ** 2)
        #     dist_outer_leds = np.sqrt((x_outer_led1 - x_outer_led2) ** 2 + (y_outer_led1 - y_outer_led2) ** 2) + 1e-8
        #
        #     if dist_led_outer_led1 < dist_outer_leds and dist_led_outer_led2 < dist_outer_leds:
        #         break
        #
        #     dists_led_arrays_search_areas[iled, idx_nearest_array] *= 2

        led_arrays[idx_nearest_array].append(iled)
    return led_arrays


def get_indices_of_ignored_leds(config):
    if config['analyse_positions']['ignore_indices'] != 'None':
        ignore_indices = np.array([int(i) for i in config['analyse_positions']['ignore_indices'].split(' ')])
    else:
        ignore_indices = np.array([])
    return ignore_indices
