import numpy as np
from scipy.spatial import distance
from ledsa.core.ConfigData import ConfigData
from typing import List
import os


def match_leds_to_led_arrays(search_areas: np.array, config: ConfigData, max_n_neighbours=5,
                             distance_weighting_factor=0.987) -> List:
    """
    Assigns LEDs to LED Arrays. Starting with the first of the defined edge LEDs, the closest neighbour LEDs are
    first detected. After that, a cumulative distance from the current LED to each neighbour and to the last edge
    LED of the LED array is calculated. The next LED from the current LED is chosen by the minimum cumulative
    distance. To account for LEDs that are close to the current LED but in the wrong direction, the distance between
    the neighbour LEDs and the last edge LED is multiplied by a weighting factor.

    :param search_areas: Array containing LED search areas.
    :type search_areas: np.array
    :param config: Configuration data.
    :type config: ConfigData
    :param max_n_neighbours: Maximum number of neighbour LEDs to consider, defaults to 5.
    :type max_n_neighbours: int
    :param distance_weighting_factor: Weighting factor for distance calculations, defaults to 0.987.
    :type distance_weighting_factor: float
    :return: List of LED arrays with assigned led_ids.
    :rtype: List
    """
    edge_indices = _get_indices_of_outer_leds(config)

    search_areas_dict = {row[0]: row[1:] for row in search_areas}
    all_led_lines_dict = {}

    for led_line_id, ref_led_id_list in enumerate(edge_indices):
        led_line_list = []
        ref_led_id_iter = iter(ref_led_id_list)
        led_id = next(ref_led_id_iter)
        ref_led_id = next(ref_led_id_iter)
        led_loc = search_areas_dict[led_id]
        ref_led_loc = search_areas_dict[ref_led_id]

        while len(search_areas_dict) > 0:
            del search_areas_dict[led_id]
            led_line_list.append(led_id)
            led_id = find_next_led(search_areas_dict, max_n_neighbours, ref_led_loc, distance_weighting_factor, led_loc)
            led_loc = search_areas_dict[led_id]

            # If last led_id of led_line is reached go to next one
            if led_id == ref_led_id:
                led_line_list.append(led_id)
                del search_areas_dict[led_id]
                break
            elif led_id == ref_led_id:
                ref_led_id = next(ref_led_id_iter)
                ref_led_loc = search_areas_dict[ref_led_id]

        all_led_lines_dict[led_line_id] = led_line_list
    num_non_matched_leds = len(search_areas_dict)
    print(f"Number of not matched LEDs: {num_non_matched_leds}")
    if num_non_matched_leds > 0:
        print("LED IDs:")
        for led_id in search_areas_dict:
            print(led_id, end=" ")
        print("\n")
    all_led_lines_dict = _remove_ignored_leds(all_led_lines_dict, config)
    return list(all_led_lines_dict.values())

def find_next_led(search_areas_dict, max_n_neighbours, ref_led_loc, distance_weighting_factor, led_loc):

    total_dist_dict = {}
    neighbours_led_id_list, neighbours_dist_list = _find_neighbour_leds(led_loc, search_areas_dict, max_n_neighbours)

    # Compute total distance from current led to neighbour leds and from neighbour led to ref led
    # Distance from neighbour led to ref led is weighted less to prefer leds closer to current led
    for neighbour_led_id, dist_neighbour in zip(neighbours_led_id_list, neighbours_dist_list):
        neighbour_led_loc = search_areas_dict[neighbour_led_id]
        dist_ref = np.linalg.norm(np.array(ref_led_loc) - np.array(neighbour_led_loc))
        dist_total = distance_weighting_factor * dist_ref + dist_neighbour
        total_dist_dict[neighbour_led_id] = dist_total
    next_led_id = min(total_dist_dict, key=total_dist_dict.get)
    return next_led_id

def _find_neighbour_leds(led_loc, search_areas_dict, max_n_neighbours):
    search_areas_dict_temp = search_areas_dict.copy()
    neighbours_led_id_list = []
    neighbours_dist_list = []
    n_neighbours = min(max_n_neighbours, len(search_areas_dict))

    # Compute distances from current led to nearest neighbour leds
    for i in range(n_neighbours):
        neighbour_led_loc, neighbour_led_id, dist_neighbour = _find_closest_node(led_loc,
                                                                                 search_areas_dict_temp)
        del search_areas_dict_temp[neighbour_led_id]
        neighbours_led_id_list.append(neighbour_led_id)
        neighbours_dist_list.append(dist_neighbour)
    return neighbours_led_id_list, neighbours_dist_list
def _find_closest_node(node: tuple, nodes_dict: dict) -> 'to be defined':
    node = np.array(node)
    nodes = np.array(list(nodes_dict.values()))
    keys = np.array(list(nodes_dict.keys()))
    distance_array = distance.cdist([node], nodes)
    closest_index = distance_array.argmin()
    dist = distance_array[0][closest_index]
    return nodes[closest_index], keys[closest_index], dist


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


def _remove_ignored_leds(all_led_lines_dict: dict, config: ConfigData) -> dict:
    ignored_indices = _get_indices_of_ignored_leds(config)
    print('Removing ignored LEDs...\nLED IDs:')
    for led_line in all_led_lines_dict.values():
        for led_id in ignored_indices:
            try:
                led_line.remove(led_id)
                print(led_id, end=" ")
            except ValueError:
                pass
    print("\n")
    return all_led_lines_dict


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


def reorder_led_indices(line_indices: List[np.ndarray]) -> List[List[int]]:
    """
    Reorders LED indices to ensure continuous sequencing within each LED array.

    :param line_indices: A list of numpy arrays, each containing the indices of LEDs within a particular array.
    :type line_indices: List[np.ndarray]
    :return: A list of lists, where each inner list contains the reordered indices of LEDs for each array.
    :rtype: List[List[int]]
    """

    start_id = 0
    line_start_indices = []
    line_end_indices = []

    # Correctly populate line_start_indices and line_end_indices
    for line_id in line_indices:
        line_start_indices.append(start_id)
        end_id = start_id + len(line_id)  # Correct calculation of end_id
        line_end_indices.append(end_id - 1)
        start_id = end_id  # Update start_id for the next iteration

    # Correctly create ranges for each line
    # This comprehension iterates over pairs of start and end indices
    # and creates a list of integers for each pair.
    line_indices_reordered = [
        list(range(start_id, end_id + 1)) for start_id, end_id in zip(line_start_indices, line_end_indices)
    ]

    return line_indices_reordered


def reorder_search_areas(search_areas, line_indices_old) -> None:
    """
    Reorders search areas based on the new order of LED indices.

    :param search_areas: A numpy array containing the original search areas.
    :type search_areas: numpy.ndarray
    :param line_indices_old: A list of numpy arrays containing the old LED indices before reordering.
    :type line_indices_old: List[np.ndarray]
    :return: A numpy array of the reordered search areas.
    :rtype: np.ndarray
    """

    def flatten_list(nested_list):
        flat_list = []
        for sublist in nested_list:
            for item in sublist:
                flat_list.append(item)
        return flat_list

    old_led_indices = flatten_list(line_indices_old)
    # TODO: Check whats right here!
    search_areas_reordered = search_areas[flatten_list(line_indices_old)]
    search_areas_reordered[:, 0] = range(len(old_led_indices)-1, -1, -1)
    # search_areas_reordered[:, 0] = range(len(old_led_indices))
    return np.flip(search_areas_reordered, axis=0)
