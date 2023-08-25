import os
import requests
import shutil
import zipfile
from typing import Tuple


def setup_demo(destination_path: str, image_src_path: str, config_src_path: str) -> None:
    """
    Set up the demo environment.

    :param destination_path: Path where the demo should be set up.
    :type destination_path: str
    :param image_src_path: Path/URL to the image source.
    :type image_src_path: str
    :param config_src_path: Path/URL to the config source.
    :type config_src_path: str
    """
    input_str = "Setting up the demo will download about 5 GB of data from the internet. Would you like to proceed? (yes/no): "
    exit_str = "Demo setup aborted by the user."
    proceed_demo_setup = _proceed_prompt(input_str, exit_str)
    if proceed_demo_setup:
        if os.path.isfile(os.path.join(destination_path, 'simulation', '.simulation_setup_successful')):
            input_str = "Looks like the simulation was already set up. Do you want to overwrite the existing data? (yes/no): "
            overwrite_demo = _proceed_prompt(input_str, exit_str)
            if overwrite_demo:
                _cleanup_demo_directories(destination_path)
            else:
                print("No changes were made to the demo setup.")
                exit(0)
        image_dest_path, simulation_dest_path = _setup_directories(destination_path)
        _download_and_extract_images(image_dest_path, simulation_dest_path, image_src_path, config_src_path)
        _edit_config_files(simulation_dest_path, setup=True)
        print("Demo setup successfully")
    else:
        exit(0)


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




def _download_and_extract_images(image_data_path: str, simulation_path: str, local_zip_path: str, local_config_path: str) -> None: # TODO: Only working local right now
    """
    Move and extract data from local paths to the target directories.

    :param image_data_path: Path to the image_data directory.
    :type image_data_path: str
    :param simulation_path: Path to the simulation directory.
    :type simulation_path: str
    :param local_zip_path: Local path to the ZIP file.
    :type local_zip_path: str
    :param local_config_path: Local path to the config file.
    :type local_config_path: str
    """
    print("Downloading data...")
    # Move and extract ZIP file into the image_data directory
    shutil.copy(local_zip_path, image_data_path)
    zip_path = os.path.join(image_data_path, os.path.basename(local_zip_path))

    # Extract contents of the ZIP's directory directly into image_data_path
    with zipfile.ZipFile(local_zip_path, 'r') as zip_ref:
        root_dir = zip_ref.namelist()[0]
        for member in zip_ref.namelist():
            if member != root_dir:
                zip_ref.extract(member, image_data_path)
                os.rename(os.path.join(image_data_path, member),
                          os.path.join(image_data_path, os.path.basename(member)))

    # Cleanup the extracted ZIP file and any unwanted directory
    os.remove(zip_path)
    shutil.rmtree(os.path.join(image_data_path, root_dir))

    # Move config file to the simulation directory
    shutil.copy(os.path.join(local_config_path, 'config.ini'), simulation_path)
    shutil.copy(os.path.join(local_config_path, 'config_analysis.ini'), simulation_path)
    with open(os.path.join(simulation_path, '.simulation_setup_successful'), 'w'):
        pass
    print("Done")



# def download_and_extract(image_data_path: str, simulation_path: str, zip_url: str, config_dir_url: str) -> None:
#     """
#     Download and extract data from URLs to the target directories.
#
#     :param image_data_path: Path to the image_data directory.
#     :type image_data_path: str
#     :param simulation_path: Path to the simulation directory.
#     :type simulation_path: str
#     :param zip_url: URL to download the ZIP file from.
#     :type zip_url: str
#     :param config_dir_url: URL of directory to download the config and config_analysis file from.
#     :type config_dir_url: str
#     """
#     # Helper function to download a file from a given URL
#     def download_file_from_url(url, destination):
#         response = requests.get(url, stream=True)
#         response.raise_for_status()
#
#         with open(destination, 'wb') as fd:
#             for chunk in response.iter_content(chunk_size=8192):
#                 fd.write(chunk)
#
#     # Download ZIP file
#     zip_name = os.path.basename(zip_url)
#     local_zip_path = os.path.join(image_data_path, zip_name)
#     download_file_from_url(zip_url, local_zip_path)
#
#     # Extract contents of the ZIP's directory directly into image_data_path
#     with zipfile.ZipFile(local_zip_path, 'r') as zip_ref:
#         root_dir = zip_ref.namelist()[0]
#         for member in zip_ref.namelist():
#             if member != root_dir:
#                 zip_ref.extract(member, image_data_path)
#                 os.rename(os.path.join(image_data_path, member),
#                           os.path.join(image_data_path, os.path.basename(member)))
#
#     # Cleanup the extracted ZIP file and any unwanted directory
#     os.remove(local_zip_path)
#     shutil.rmtree(os.path.join(image_data_path, root_dir), ignore_errors=True)
#
#     # Download config files from the given URL and copy to the simulation directory
#
#     download_file_from_url(os.path.join(config_dir_url, 'config.ini'), simulation_path)
#     download_file_from_url(os.path.join(config_dir_url, 'config_analysis.ini'), simulation_path)
#     print("All Demo files have been downloaded successfully!")

def _edit_config_files(simulation_path: str, num_of_cores=1, setup=False) -> None:
    """
    Edit the configuration files based on the provided parameters.

    :param simulation_path: Path to the simulation directory containing config files.
    :type simulation_path: str
    :param num_of_cores: Number of cores to be set in the config.
    :type num_of_cores: int, optional
    :param setup: Indicator if setup is required or not.
    :type setup: bool, optional
    """
    config_file = os.path.join(simulation_path, 'config.ini')
    config_analysis_file = os.path.join(simulation_path, 'config_analysis.ini')
    if setup:
        _replace_params_in_file(config_file, 'img_directory', '../image_data/')
    _replace_params_in_file(config_file, 'num_of_cores', num_of_cores)
    _replace_params_in_file(config_analysis_file, 'num_of_cores', num_of_cores)



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


