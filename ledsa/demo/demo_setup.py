import os
import requests
import shutil
import zipfile
from typing import Tuple
from tqdm import tqdm

from ledsa.analysis.ConfigDataAnalysis import ConfigDataAnalysis
from ledsa.core.ConfigData import ConfigData


def setup_demo(destination_path: str, image_data_url: str) -> None:
    """
    Set up the demo environment.

    :param destination_path: Path where the demo should be set up.
    :type destination_path: str
    :param image_data_url: URL to the image data.
    :type image_data_url: str
    """
    input_str = "Setting up the demo will download about 5 GB of data from the internet. Would you like to proceed? (yes/no): "
    exit_str = "Demo setup aborted by the user."
    proceed_demo_setup = _proceed_prompt(input_str, exit_str)
    if proceed_demo_setup:
        if os.path.isfile(os.path.join(destination_path, 'simulation', '.simulation_setup_successful')):
            input_str = "Looks like the simulation was already set up. Do you want to overwrite the existing data? (yes/no): "
            overwrite_demo = _proceed_prompt(input_str, exit_str)
            if overwrite_demo:
                # _cleanup_demo_directories(destination_path)
                pass
            else:
                print("No changes were made to the demo setup.")
                exit(0)
        image_dest_path, simulation_dest_path = _setup_directories(destination_path)
        _download_and_extract_images(image_data_url, image_dest_path, simulation_dest_path)
        _create_config_files(os.path.join(destination_path, 'simulation'))
        print("Demo setup successfully")
    else:
        exit(0)

def _create_config_files(path):
    owd = os.getcwd()
    os.chdir(path)
    config = ConfigData(
        load_config_file=False,
        img_directory="../image_data/",
        img_name_string="V001_Cam01_{}.CR2",
        num_img_overflow=9999,
        first_img_experiment_id=1,
        last_img_experiment_id=275,
        num_cores=4,
        date="27.11.2018",
        start_time="15:36:07",
        time_img_id=None,
        time_ref_img_time=None,
        time_diff_to_image_time=-1,
        ref_img_id=1,
        pixel_value_percentile=99.875,
        channel='all',
        max_num_leds=1000,
        search_area_radius=10,
        num_arrays=7,
        first_img_analysis_id=1,
        last_img_analysis_id=275,
        num_skip_imgs=0,
        num_skip_leds=0,
        merge_led_array_indices=None
    )
    config.set('analyse_positions', '   ignore_led_indices', '781 675 746')

    config.set('analyse_positions', '   led_array_edge_indices', '\n   457 246\n    12 347\n    578 671\n    31 535\n    825 838\n    198 698\n    965 782\n   ')
    config.set('analyse_positions', '   led_array_edge_coordinates','\n   6.86 3.13 1.14 5.98 2.83 3.32\n   '
                                                                '5.96 2.83 0.99 5.98 2.83 3.35\n   '
                                                                '5.96 2.83 1.17 5.05 2.62 3.32\n   '
                                                                '5.02 2.62 1.0 5.05 2.62 3.34\n   '
                                                                '4.09 2.38 1.19 5.05 2.62 3.32\n   '
                                                                '4.09 2.38 1.0 4.12 2.38 3.35\n   '
                                                                '3.17 2.25 1.17 4.12 2.38 3.32\n   ')

    config.save()

    config_analysis = ConfigDataAnalysis(
        load_config_file=False,
        num_layers=20,
        num_ref_images=10,
        num_cores=4,
        reference_property='sum_col_val',
        average_images=False,
        solver='linear',
        weighting_preference=-6e-3,
        weighting_curvature=1e-6,
        num_iterations=200
    )
    config_analysis.set('experiment_geometry', '   camera_position', '7.29 6.46 2.3')
    config_analysis.set('DEFAULT', '   camera_channels', '0 1 2')

    config_analysis.set('model_parameters', '   domain_bounds', '0.99 3.35')
    config_analysis.set('model_parameters', '   led_array_indices', '0 1 2 3 4 5 6')

    config_analysis.save()

    os.chdir(owd)


def _proceed_prompt(promt_str: str, exit_str: str) -> bool:
    """
    Prompts the user with a given message and waits for a 'yes' or 'no' response.

    :param promt_str: The prompt string to display to the user.
    :type promt_str: str
    :param exit_str: The message to display when the user responds with 'no'.
    :type exit_str: str
    :return: Returns True if user responds with 'yes', and False if 'no'.
    :rtype: bool
    """
    while True:
        proceed = input(promt_str).lower()
        if proceed == "yes":
            return True
        elif proceed == "no":
            print(exit_str)
            return False
        else:
            print("Invalid choice. Please enter 'yes' or 'no'.")
            continue



def _cleanup_demo_directories(path: str) -> None:
    """
    Removes the 'simulation' and 'image_data' directories from the given path.

    :param path: The base path where the directories are located.
    :type path: str
    """
    shutil.rmtree(os.path.join(path, 'simulation'))
    shutil.rmtree(os.path.join(path, 'image_data'))



def _setup_directories(destination_path: str) -> Tuple[str, str]:
    """
    Set up required directories.

    :param destination_path: Path where the directories should be set up.
    :type destination_path: str
    :return: Paths to the created directories (image_data, simulation).
    :rtype: tuple(str)
    """
    image_dest_path = os.path.join(destination_path, "image_data")
    simulation_dest_path = os.path.join(destination_path, "simulation")
    print("Setting up demo directories...")
    if not os.path.exists(image_dest_path):
        os.makedirs(image_dest_path)
    if not os.path.exists(simulation_dest_path):
        os.makedirs(simulation_dest_path)
    print("Done")
    return image_dest_path, simulation_dest_path



def _download_and_extract_images(image_data_url: str, image_data_dest_path: str, simulation_path: str) -> None:
    """
    Download and extract data from URLs to the target directories.

    :param image_data_url: URL to download the ZIP file from.
    :type image_data_url: str
    :param image_data_dest_path: Path to the destination image_data directory.
    :type image_data_dest_path: str
    :param simulation_path: Path to the simulation directory.
    :type simulation_path: str
    """
    # Helper function to download a file from a given URL with progress bar
    def download_file_from_url(url, destination):
        response = requests.get(url, stream=True)
        response.raise_for_status()
        total_size = int(response.headers.get('content-length', 0))
        block_size = 8192  # 8 Kibibytes
        progress_bar = tqdm(total=total_size, unit='iB', unit_scale=True)

        with open(destination, 'wb') as fd:
            for chunk in response.iter_content(chunk_size=block_size):
                progress_bar.update(len(chunk))
                fd.write(chunk)
        progress_bar.close()

    # Download ZIP file
    zip_name = os.path.basename(image_data_url)
    local_zip_path = os.path.join(image_data_dest_path, zip_name)
    download_file_from_url(image_data_url, local_zip_path)

    # Extract contents of the ZIP's directory directly into image_data_path with progress bar
    with zipfile.ZipFile(local_zip_path, 'r') as zip_ref:
        members = zip_ref.namelist()
        for member in tqdm(members, desc="Extracting", unit="files"):
            if member != members[0]:  # Assuming the first member is the root directory
                zip_ref.extract(member, image_data_dest_path)
                os.rename(os.path.join(image_data_dest_path, member),
                          os.path.join(image_data_dest_path, os.path.basename(member)))

    # Cleanup the extracted ZIP file and any unwanted directory
    os.remove(local_zip_path)
    shutil.rmtree(os.path.join(image_data_dest_path, members[0]), ignore_errors=True)
    with open(os.path.join(simulation_path, '.simulation_setup_successful'), 'w'):
        pass
    print("All Image files have been downloaded successfully!")


def _edit_config_files(simulation_path: str, num_cores=1) -> None:
    """
    Edit the configuration files based on the provided parameters.

    :param simulation_path: Path to the simulation directory containing config files.
    :type simulation_path: str
    :param num_cores: Number of cores to be set in the config.
    :type num_cores: int, optional
    """
    config_file = os.path.join(simulation_path, 'config.ini')
    config_analysis_file = os.path.join(simulation_path, 'config_analysis.ini')
    _replace_params_in_file(config_file, 'num_cores', num_cores)
    _replace_params_in_file(config_analysis_file, 'num_cores', num_cores)



def _replace_params_in_file(file_path: str, target_word: str, target_value: str) -> None:
    """
    For lines containing target_word, replace content after '=' with replacement_word.

    :param file_path: Name/path of the file.
    :type file_path: str
    :param target_word: Word to search for in each line.
    :type target_word: str
    :param target_value: Word that will replace content after '='.
    :type target_value: str
    """
    # Read the file lines into memory
    with open(file_path, 'r') as file:
        lines = file.readlines()

    # Process the lines
    new_lines = []
    for line in lines:
        if target_word in line and '=' in line:
            prefix = line.split('=')[0] + '='
            new_line = prefix + ' ' + str(target_value) + '\n'
            new_lines.append(new_line)
        else:
            new_lines.append(line)

    # Write the modified lines back to the file
    with open(file_path, 'w') as file:
        file.writelines(new_lines)
