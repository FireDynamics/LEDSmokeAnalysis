import os
from typing import List, Tuple

import numpy as np
from matplotlib import pyplot as plt

from ledsa.core.ConfigData import ConfigData


def match_leds_to_led_arrays(search_areas: np.ndarray, config: ConfigData) -> List[List]:
    """
    Matches LEDs to their corresponding LED arrays based on the given LED line edge indices.

    :param search_areas: A numpy array containing LED search areas.
    :type search_areas: numpy.ndarray
    :param config: An instance of ConfigData containing the configuration data.
    :type config: ConfigData
    :return: A List of Lists containing matched LED arrays.
    :rtype: list
    """
    edge_indices = _get_indices_of_outer_leds(config)
    if len(search_areas) <= np.max(edge_indices):
        exit("At least one of the chosen LED indices is larger that the number of found LEDS!")
    dists_led_arrays_search_areas = _calc_dists_between_led_arrays_and_search_areas(edge_indices, search_areas)
    led_arrays = _match_leds_to_arrays_with_min_dist(dists_led_arrays_search_areas, edge_indices, config, search_areas)
    return led_arrays


def generate_line_indices_files(line_indices: List[np.ndarray], filename_extension: str = '') -> None:
    """
    Generate files containing line indices.

    :param line_indices: List containing indices for each line.
    :type line_indices: list
    :param filename_extension: Optional extension for the generated filename, defaults to ''.
    :type filename_extension: str
    """
    for i in range(len(line_indices)):
        file_path = os.path.join('analysis', f'line_indices_{i:03}{filename_extension}.csv')
        out_file = open(file_path, 'w')
        for iled in line_indices[i]:
            out_file.write('{}\n'.format(iled))
        out_file.close()


def _get_indices_of_outer_leds(config: ConfigData) -> np.ndarray:
    """
    Retrieve the indices of outer LEDs based on the configuration data.

    :param config: An instance of ConfigData containing the configuration data.
    :type config: ConfigData
    :return: List containing indices of outer LEDs.
    :rtype: numpy.ndarray
    """
    if config['analyse_positions']['line_edge_indices'] == 'None':
        config.in_line_edge_indices()
        with open('config.ini', 'w') as configfile:
            config.write(configfile)
    line_edge_indices = config.get2dnparray('analyse_positions', 'line_edge_indices')
    # makes sure that line_edge_indices is a 2d list
    if len(line_edge_indices.shape) == 1:
        line_edge_indices = np.atleast_2d(line_edge_indices)
    return line_edge_indices


def _calc_dists_between_led_arrays_and_search_areas(line_edge_indices:np.ndarray, search_areas:np.ndarray) -> np.ndarray:
    """
    Calculate the distances between LED arrays and the search areas.

    :param line_edge_indices: List of indices indicating edges of the line.
    :type line_edge_indices: list
    :param search_areas: A numpy array containing LED search areas.
    :type search_areas: numpy.ndarray
    :return: distances_led_arrays_search_areas: A 2D numpy array containing distances between LED arrays and search areas.
    :rtype: numpy.ndarray
    """
    distances_led_arrays_search_areas = np.zeros((search_areas.shape[0], len(line_edge_indices)))

    for line_edge_idx in range(len(line_edge_indices)):
        i1 = line_edge_indices[line_edge_idx][0]
        i2 = line_edge_indices[line_edge_idx][1]

        corner1 = np.array(search_areas[i1, 1:])
        corner2 = np.array(search_areas[i2, 1:])

        d = _calc_dists_to_line_segment(search_areas[:, 1:], corner1, corner2)

        distances_led_arrays_search_areas[:, line_edge_idx] = d
    return distances_led_arrays_search_areas


def _calc_dists_to_line_segment(points: np.ndarray, c1: np.ndarray, c2: np.ndarray) -> np.ndarray:
    """
    Calculate the distance from points to a line segment defined by two corners.

    :param points: A numpy array containing coordinates of points.
    :type points: numpy.ndarray
    :param c1: Coordinates of the first corner of the line segment.
    :type c1: numpy.ndarray
    :param c2: Coordinates of the second corner of the line segment.
    :type c2: numpy.ndarray
    :return: A numpy array containing the distances from the points to the line segment.
    :rtype: numpy.ndarray
    """
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


def _match_leds_to_arrays_with_min_dist(dists_led_arrays_search_areas: np.ndarray, edge_indices: np.ndarray, config: ConfigData, search_areas: np.ndarray) -> np.ndarray:
    """
    Match LEDs to LED arrays based on minimum distances.

    :param dists_led_arrays_search_areas: A 2D numpy array containing distances between LED arrays and search areas.
    :type dists_led_arrays_search_areas: numpy.ndarray
    :param edge_indices: A numpy array containing the indices indicating edges of LED arrays.
    :type edge_indices: numpy.ndarray
    :param config: An instance of ConfigData containing the configuration data.
    :type config: ConfigData
    :param search_areas: A numpy array containing LED search areas.
    :type search_areas: numpy.ndarray
    :return: A 2D numpy array with matched LED arrays.
    :rtype: numpy.ndarray
    """
    ignore_indices = _get_indices_of_ignored_leds(config)
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


def _get_indices_of_ignored_leds(config: ConfigData) -> np.ndarray:
    """
    Retrieve the indices of ignored LEDs based on the configuration data.

    :param config: An instance of ConfigData containing the configuration data.
    :type config: ConfigData
    :return: A numpy array containing indices of ignored LEDs.
    :rtype: numpy.ndarray
    """
    if config['analyse_positions']['ignore_indices'] != 'None':
        ignore_indices = np.array([int(i) for i in config['analyse_positions']['ignore_indices'].split(' ')])
    else:
        ignore_indices = np.array([])
    return ignore_indices



