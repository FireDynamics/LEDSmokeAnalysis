import os
import requests
import shutil
import zipfile

def setup_demo(destination_path, image_src_path, config_src_path):
    image_dest_path, simulation_dest_path = setup_directories(destination_path)
    download_and_extract(image_dest_path, simulation_dest_path, image_src_path, config_src_path)
    edit_config_files(simulation_dest_path)
    print("Demo setup successfully")

def setup_directories(destination_path):
    """
    Set up required directories.

    :param str destination_path: Path where the directories should be set up.
    :return: Paths to the created directories (image_data, simulation).
    :rtype: tuple
    """
    image_dest_path = os.path.join(destination_path, "image_data")
    simulation_dest_path = os.path.join(destination_path, "simulation")

    if not os.path.exists(image_dest_path):
        os.makedirs(image_dest_path)
    if not os.path.exists(simulation_dest_path):
        os.makedirs(simulation_dest_path)


    return image_dest_path, simulation_dest_path


# def download_and_extract(image_data_path, simulation_path, zip_url, config_url):
#     """
#     Download data from Zenodo and extract/place in appropriate directories.
#
#     :param str image_data_path: Path to the image_data directory.
#     :param str simulation_path: Path to the simulation directory.
#     :param str zip_url: URL to the ZIP file on Zenodo.
#     :param str config_url: URL to the config file on Zenodo.
#     """
#     zip_response = requests.get(zip_url, stream=True)
#     zip_path = os.path.join(image_data_path, "data.zip")
#
#     with open(zip_path, "wb") as f:
#         for chunk in zip_response.iter_content(chunk_size=8192):
#             f.write(chunk)
#
#     with zipfile.ZipFile(zip_path, 'r') as zip_ref:
#         zip_ref.extractall(image_data_path)
#
#     os.remove(zip_path)
#
#     config_response = requests.get(config_url)
#     config_path = os.path.join(simulation_path, "config.cfg")
#     with open(config_path, "wb") as f:
#         f.write(config_response.content)
#     print("All Demo files have been downloaded successfully!")


def download_and_extract(image_data_path, simulation_path, local_zip_path, local_config_path):
    """
    Move and extract data from local paths to the target directories.

    :param str image_data_path: Path to the image_data directory.
    :param str simulation_path: Path to the simulation directory.
    :param str local_zip_path: Local path to the ZIP file.
    :param str local_config_path: Local path to the config file.
    """
    # Move and extract ZIP file into the image_data directory
    shutil.copy(local_zip_path, image_data_path)
    zip_path = os.path.join(image_data_path, os.path.basename(local_zip_path))

    # Extract contents of the ZIP's directory directly into image_data_path
    with zipfile.ZipFile(local_zip_path, 'r') as zip_ref:
        root_dir = zip_ref.namelist()[0]
        # for member in zip_ref.namelist():
        #     if member != root_dir:
        #         zip_ref.extract(member, image_data_path)
        #         os.rename(os.path.join(image_data_path, member),
        #                   os.path.join(image_data_path, os.path.basename(member)))

    # # Move config file to the simulation directory
    # os.remove(zip_path)
    # shutil.rmtree(os.path.join(image_data_path, root_dir))

    # Move config file to the simulation directory
    shutil.copy(os.path.join(local_config_path, 'config.ini'), simulation_path)
    shutil.copy(os.path.join(local_config_path, 'config_analysis.ini'), simulation_path)

    print("All Demo files have been downloaded successfully!")

def edit_config_files(simulation_path, num_of_cores=1, setup=False):
    config_file = os.path.join(simulation_path, 'config.ini')
    config_analysis_file = os.path.join(simulation_path, 'config_analysis.ini')
    if setup:
        replace_params_in_file(config_file, 'img_directory', '../image_data/')
    replace_params_in_file(config_file, 'num_of_cores', num_of_cores)
    replace_params_in_file(config_analysis_file, 'num_of_cores', num_of_cores)



def replace_params_in_file(file_path, target_word, target_value):
    """
    For lines containing target_word, replace content after '=' with replacement_word.

    :param file_path: str - Name/path of the file
    :param target_word: str - Word to search for in each line
    :param target_value: str - Word that will replace content after '='

    :return: None
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


