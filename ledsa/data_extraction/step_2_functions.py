import numpy as np
from scipy.spatial import distance
from ledsa.core.ConfigData import ConfigData
from typing import List, Tuple, Dict, Optional
import os


def match_leds_to_led_arrays(search_areas: np.ndarray, 
                            config: ConfigData,
                            max_n_neighbours: int = 5,
                            distance_weighting_factor: float = 0.987,
                            ignore_leds: Optional[List[int]] = None) -> List[List[int]]:
    """
    Assigns LEDs to LED Arrays. Starting with the first of the defined edge LEDs, the closest neighbour LEDs are
    first detected. After that, a cumulative distance from the current LED to each neighbour and to the last edge
    LED of the LED array is calculated. The next LED from the current LED is chosen by the minimum cumulative
    distance. To account for LEDs that are close to the current LED but in the wrong direction, the distance between
    the neighbour LEDs and the last edge LED is multiplied by a weighting factor.

    :param search_areas: Array containing LED search areas.
    :param config: Configuration data.
    :param max_n_neighbours: Maximum number of neighbour LEDs to consider, defaults to 5.
    :param distance_weighting_factor: Weighting factor for distance calculations, defaults to 0.987.
    :param ignore_leds: List of LED IDs to ignore during processing, defaults to None.
    :return: List of LED arrays with assigned led_ids.
    """
    try:
        # Get edge indices from config
        edge_indices = _get_indices_of_outer_leds(config)
        ignored_indices = _get_indices_of_ignored_leds(config)

        # Convert search areas to dictionary for easier access
        search_areas_dict = {row[0]: row[1:] for row in search_areas}
        all_led_arrays_dict = {}

        # Remove ignored LEDs if specified
        if ignore_leds is not None:
            for led_id in ignore_leds:
                if led_id in search_areas_dict:
                    del search_areas_dict[led_id]

        # Process each LED array defined by edge indices
        for led_array_id, ref_led_id_list in enumerate(edge_indices):
            led_array_list = []

            # Initialize with the first two reference LEDs
            try:
                ref_led_id_iter = iter(ref_led_id_list)
                led_id = next(ref_led_id_iter)
                ref_led_id = next(ref_led_id_iter)
                led_loc = search_areas_dict.get(led_id)
                ref_led_loc = search_areas_dict.get(ref_led_id)

                if led_loc is None or ref_led_loc is None:
                    raise ValueError(f"Reference LED ID {led_id if led_loc is None else ref_led_id} not found in search areas. Maybe the Edge LEDs are not defined correctly in the config file?")

                # Process LEDs until we reach the end of the LED array or run out of LEDs
                while search_areas_dict:
                    # Remove current LED from dictionary and add to led_array_list
                    del search_areas_dict[led_id]
                    led_array_list.append(led_id)

                    # Find the next LED in the sequence
                    try:
                        led_id = _find_next_led(search_areas_dict, max_n_neighbours, 
                                              ref_led_loc, distance_weighting_factor, led_loc)
                        led_loc = search_areas_dict[led_id]

                        # If we've reached a reference LED, handle accordingly
                        if led_id == ref_led_id_list[-1]:
                            led_array_list.append(led_id)
                            del search_areas_dict[led_id]
                            break
                        elif led_id == ref_led_id:
                            try:
                                ref_led_id = next(ref_led_id_iter)
                                ref_led_loc = search_areas_dict[ref_led_id]
                            except StopIteration:
                                # No more reference LEDs in this LED array
                                break
                    except (KeyError, ValueError) as e:
                        print(f"Error processing LED array {led_array_id}: {str(e)}")
                        break

            except StopIteration:
                print(f"Warning: Not enough reference LEDs for LED array {led_array_id}")
                continue

            all_led_arrays_dict[led_array_id] = led_array_list

        # Report on unmatched LEDs
        num_non_matched_leds = len(search_areas_dict)
        print(f"Number of not matched LEDs: {num_non_matched_leds}")
        if num_non_matched_leds > 0:
            print("LED IDs:")
            for led_id in search_areas_dict:
                print(led_id, end=" ")
            print("\n")

        # Remove ignored LEDs from the results
        all_led_arrays_dict = _remove_ignored_leds(all_led_arrays_dict, ignored_indices)

        return list(all_led_arrays_dict.values())

    except Exception as e:
        print(f"Error in match_leds_to_led_arrays: {str(e)}")
        raise


def _find_next_led(search_areas_dict: Dict[int, np.ndarray], 
                  max_n_neighbours: int, 
                  ref_led_loc: np.ndarray,
                  distance_weighting_factor: float, 
                  led_loc: np.ndarray) -> int:
    """
    Find the next LED in the sequence based on distance calculations.

    :param search_areas_dict: Dictionary of LED IDs to their locations
    :param max_n_neighbours: Maximum number of neighbors to consider
    :param ref_led_loc: Location of the reference LED
    :param distance_weighting_factor: Factor to weight distances
    :param led_loc: Location of the current LED
    :return: ID of the next LED
    :raises ValueError: If no valid next LED can be found
    """
    if not search_areas_dict:
        raise ValueError("No LEDs left to process")

    total_dist_dict = {}
    neighbours_led_id_list, neighbours_dist_list = _find_neighbour_leds(
        led_loc, search_areas_dict, max_n_neighbours)

    # Compute total distance from current LED to neighbour LEDs and from neighbour LED to ref LED
    # Distance from neighbour LED to ref LED is weighted less to prefer LEDs closer to current LED
    for neighbour_led_id, dist_neighbour in zip(neighbours_led_id_list, neighbours_dist_list):
        neighbour_led_loc = search_areas_dict[neighbour_led_id]
        dist_ref = np.linalg.norm(np.array(ref_led_loc) - np.array(neighbour_led_loc))
        dist_total = distance_weighting_factor * dist_ref + dist_neighbour
        total_dist_dict[neighbour_led_id] = dist_total

    if not total_dist_dict:
        raise ValueError("No valid neighbors found")

    next_led_id = min(total_dist_dict, key=total_dist_dict.get)
    return next_led_id


def _find_neighbour_leds(led_loc: np.ndarray, 
                        search_areas_dict: Dict[int, np.ndarray], 
                        max_n_neighbours: int) -> Tuple[List[int], List[float]]:
    """
    Find the nearest neighboring LEDs to the current LED.

    :param led_loc: Location of the current LED
    :param search_areas_dict: Dictionary of LED IDs to their locations
    :param max_n_neighbours: Maximum number of neighbors to consider
    :return: Tuple of (list of neighbor LED IDs, list of distances)
    """
    search_areas_dict_temp = search_areas_dict.copy()
    neighbours_led_id_list = []
    neighbours_dist_list = []

    # Determine how many neighbors to find (limited by available LEDs)
    n_neighbours = min(max_n_neighbours, len(search_areas_dict))

    # Compute distances from current LED to nearest neighbour LEDs
    for _ in range(n_neighbours):
        try:
            neighbour_led_loc, neighbour_led_id, dist_neighbour = _find_closest_node(
                led_loc, search_areas_dict_temp)
            del search_areas_dict_temp[neighbour_led_id]
            neighbours_led_id_list.append(neighbour_led_id)
            neighbours_dist_list.append(dist_neighbour)
        except (KeyError, ValueError) as e:
            print(f"Warning: Error finding neighbor: {str(e)}")
            break

    return neighbours_led_id_list, neighbours_dist_list


def _find_closest_node(node: np.ndarray, 
                      nodes_dict: Dict[int, np.ndarray]) -> Tuple[np.ndarray, int, float]:
    """
    Find the closest node to the given node from a dictionary of nodes.

    :param node: The reference node
    :param nodes_dict: Dictionary of node IDs to their locations
    :return: Tuple of (closest node location, closest node ID, distance)
    :raises ValueError: If nodes_dict is empty
    """
    if not nodes_dict:
        raise ValueError("Empty nodes dictionary")

    node = np.array(node)
    nodes = np.array(list(nodes_dict.values()))
    keys = np.array(list(nodes_dict.keys()))

    # Calculate distances between the node and all other nodes
    distance_array = distance.cdist([node], nodes)
    closest_index = distance_array.argmin()
    dist = distance_array[0][closest_index]

    return nodes[closest_index], keys[closest_index], dist


def _get_indices_of_outer_leds(config: ConfigData) -> np.ndarray:
    """
    Retrieve the indices of outer LEDs based on the configuration data.

    :param config: Configuration data for LED processing
    :return: List containing indices of outer LEDs.
    """
    try:
        # Check if LED array edge indices are defined in config
        if config['analyse_positions']['led_array_edge_indices'] == 'None':
            config.in_led_array_edge_indices()
            with open('config.ini', 'w') as configfile:
                config.write(configfile)

        # Get the LED array edge indices from config
        led_array_edge_indices = config.get2dnparray('analyse_positions', 'led_array_edge_indices')

        # Ensure led_array_edge_indices is a 2D array
        if len(led_array_edge_indices.shape) == 1:
            led_array_edge_indices = np.atleast_2d(led_array_edge_indices)

        return led_array_edge_indices

    except Exception as e:
        print(f"Error getting indices of outer LEDs: {str(e)}")
        # Return empty array as fallback
        return np.array([])


def _get_indices_of_ignored_leds(config: ConfigData) -> np.ndarray:
    """
    Retrieve the indices of ignored LEDs based on the configuration data.

    :param config: Configuration data for LED processing
    :return: A numpy array containing indices of ignored LEDs.
    """
    try:
        # Check if ignore indices are defined in config
        if config['analyse_positions']['ignore_led_indices'] != 'None':
            # Parse the space-separated list of indices
            ignore_led_indices = np.array([int(i) for i in
                                      config['analyse_positions']['ignore_led_indices'].split(' ')])
        else:
            ignore_led_indices = np.array([])

        return ignore_led_indices

    except Exception as e:
        print(f"Error getting indices of ignored LEDs: {str(e)}")
        # Return empty array as fallback
        return np.array([])


def _remove_ignored_leds(all_led_arrays_dict: Dict[int, List[int]],
                         ignored_indices: np.ndarray) -> Dict[int, List[int]]:
    """
    Remove ignored LEDs from the LED arrays dictionary.

    :param all_led_arrays_dict: Dictionary mapping LED array indices to lists of LED indices
    :param ignored_indices: Array of LED indices to ignore
    :return: Updated dictionary with ignored LEDs removed
    """
    if len(ignored_indices) == 0:
        return all_led_arrays_dict

    print('Removing ignored LEDs...\nLED IDs:')

    # Create a copy to avoid modifying during iteration
    result_dict = {k: v.copy() for k, v in all_led_arrays_dict.items()}

    for led_array in result_dict.values():
        for led_id in ignored_indices:
            try:
                led_array.remove(led_id)
                print(led_id, end=" ")
            except ValueError:
                # LED ID not in this LED array, which is fine
                pass

    print("\n")
    return result_dict


def generate_led_array_indices_files(led_array_indices: List[List[int]],
                                     filename_extension: str = '') -> None:
    """
    Generate files containing LED array indices.

    :param led_array_indices: List containing indices for each LED array.
    :param filename_extension: Optional extension for the generated filename, defaults to ''.
    """
    try:
        # Create analysis directory if it doesn't exist
        os.makedirs('analysis', exist_ok=True)

        # Generate a file for each LED array
        for i, led_array in enumerate(led_array_indices):
            file_path = os.path.join('analysis', f'led_array_indices_{i:03}{filename_extension}.csv')

            with open(file_path, 'w') as out_file:
                for led_id in reversed(led_array):
                    out_file.write(f'{led_id}\n')

        print(f"Generated {len(led_array_indices)} LED array indices files in the 'analysis' directory")

    except Exception as e:
        print(f"Error generating LED array indices files: {str(e)}")
        raise


def reorder_led_indices(led_array_indices: List[List[int]]) -> List[List[int]]:
    """
    Reorders LED indices to ensure continuous sequencing within each LED array.

    :param led_array_indices: A list of lists, each containing the indices of LEDs within a particular array.
    :return: A list of lists, where each inner list contains the reordered indices of LEDs for each array.
    """
    try:
        start_id = 0
        led_array_start_indices = []
        led_array_end_indices = []

        # Populate led_array_start_indices and led_array_end_indices
        for led_array in led_array_indices:
            led_array_start_indices.append(start_id)
            end_id = start_id + len(led_array)  # Calculate end_id
            led_array_end_indices.append(end_id - 1)
            start_id = end_id  # Update start_id for the next iteration

        # Create ranges for each LED array
        # This comprehension iterates over pairs of start and end indices
        # and creates a list of integers for each pair.
        led_array_indices_reordered = [
            list(range(start_id, end_id + 1)) 
            for start_id, end_id in zip(led_array_start_indices, led_array_end_indices)
        ]

        return led_array_indices_reordered

    except Exception as e:
        print(f"Error reordering LED indices: {str(e)}")
        raise


def reorder_search_areas(search_areas: np.ndarray,
                         led_array_indices_old: List[List[int]]) -> np.ndarray:
    """
    Reorders search areas based on the new order of LED indices.

    :param search_areas: A numpy array containing the original search areas.
    :param led_array_indices_old: A list of lists containing the old LED indices before reordering.
    :return: A numpy array of the reordered search areas.
    """
    try:
        # Helper function to flatten nested lists
        def flatten_list(nested_list):
            flat_list = []
            for sublist in nested_list:
                for item in sublist:
                    flat_list.append(item)
            return flat_list

        # Flatten the list of LED indices
        old_led_indices = flatten_list(led_array_indices_old)

        # Reorder search areas based on flattened indices
        search_areas_reordered = search_areas[old_led_indices]

        # Update the LED IDs in the first column
        search_areas_reordered[:, 0] = range(len(old_led_indices))

        return search_areas_reordered

    except Exception as e:
        print(f"Error reordering search areas: {str(e)}")
        raise


def merge_indices_of_led_arrays(led_array_indices: List[List[int]], config: ConfigData) -> List[List[int]]:
    """
    Merges LED arrays according to the configuration specified in 'merge_led_arrays'.

    :param led_array_indices: A list of lists, each containing the indices of LEDs within a particular array.
    :param config: Configuration data containing merge information.
    :return: A list of lists with merged LED arrays.
    """
    try:
        # Check if merge_led_arrays is defined in config
        if config['analyse_positions']['merge_led_array_indices'] == 'None':
            # No merging needed
            return led_array_indices

        # Get the merge configuration
        merge_config = config.get2dnparray('analyse_positions', 'merge_led_array_indices', 'var')

        # Create a new list to hold the merged arrays
        merged_indices = []

        # Process each merge group
        for merge_group in merge_config:
            # Create a new array for this merge group
            merged_array = []

            # Add all LEDs from the specified arrays to the merged array
            for array_index in merge_group:
                if 0 <= array_index < len(led_array_indices):
                    merged_array.extend(led_array_indices[array_index])
                else:
                    print(f"Warning: Array index {array_index} out of range, skipping")

            # Add the merged array to the result if it's not empty
            if merged_array:
                merged_indices.append(merged_array)

        # Generate LED array indices files with the merged arrays
        generate_led_array_indices_files(merged_indices, '_merge')

        return merged_indices

    except Exception as e:
        print(f"Error merging LED array indices: {str(e)}")
        # Return the original indices as fallback
        return led_array_indices
