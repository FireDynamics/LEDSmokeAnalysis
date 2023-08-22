import os
import requests
import shutil
import zipfile


def setup_directories(destination_path):
    """
    Set up required directories.

    :param str destination_path: Path where the directories should be set up.
    :return: Paths to the created directories (image_data, simulation).
    :rtype: tuple
    """
    image_data_path = os.path.join(destination_path, "image_data")
    simulation_path = os.path.join(destination_path, "simulation")

    if not os.path.exists(image_data_path):
        os.makedirs(image_data_path)
    if not os.path.exists(simulation_path):
        os.makedirs(simulation_path)

    return image_data_path, simulation_path


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

    with zipfile.ZipFile(zip_path, 'r') as zip_ref:
        zip_ref.extractall(image_data_path)

    os.remove(zip_path)

    # Move config file to the simulation directory
    shutil.copy(local_config_path, simulation_path)