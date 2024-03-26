import os

import exifread
import numpy as np
import rawpy
from matplotlib import pyplot as plt

def _read_img_file(filename: str, channel: int) -> np.ndarray:
    """
    Reads an image file and extracts a single color channel.

    :param filename: The file path of the image to be read.
    :type filename: str
    :param channel: The index of the color channel to extract (0 for Red, 1 for Green, 2 for Blue in RGB images).
    :type channel: int
    :return: A 2D numpy array containing the data of the specified color channel from the image.
    :rtype: np.ndarray
    """
    data = plt.imread(filename)
    return data[:, :, channel]

def _read_raw_file(filename: str, channel: int) -> np.ndarray:
    """
    Extracts a single color channel from a RAW image file.

    :param filename: The file path of the RAW image to be read.
    :type filename: str
    :param channel: The index of the color channel to extract. The Bayer array typically contains two green channels (1 and 3),
                    one red (0), and one blue (2).
    :type channel: int
    :return: A 2D numpy array representing the extracted channel, with all other channel values masked or set to zero.
    :rtype: np.ndarray
    """
    with rawpy.imread(filename) as raw:
        data = raw.raw_image_visible.copy()
        filter_array = raw.raw_colors_visible
        black_level = raw.black_level_per_channel[channel]
        white_level = raw.white_level
    channel_array = data.astype(np.int16) - black_level
    channel_array = (channel_array * (white_level / (white_level - black_level))).astype(np.int16)
    channel_array = np.clip(channel_array, 0, white_level)
    if channel == 0 or channel == 2:
        channel_array = np.where(filter_array == channel, channel_array, 0)
    elif channel == 1:
        channel_array = np.where((filter_array == 1) | (filter_array == 3), channel_array, 0)
    return channel_array


def read_channel_from_img(filename: str, channel: int) -> np.ndarray:
    """
    Returns a 2D array of the image for a single color channel.
    8bit is default range for JPG. For RAW files the Bayer array is returned as a 2D array where
    all channel values except the selected channel are masked.

    :param filename: The path of the image file to read.
    :type filename: str
    :param channel: The color channel to process.
    :type channel: int
    :return: A 2D array containing the processed image data for one channel.
    :rtype: np.ndarray
    """
    extension = os.path.splitext(filename)[-1]
    if extension in ['.JPG', '.JPEG', '.jpg', '.jpeg', '.PNG', '.png']:
        channel_array = _read_img_file(filename, channel)
    elif extension in ['.CR2']:
        channel_array = _read_raw_file(filename, channel)
    return channel_array

def get_exif_entry(filename: str, tag: str) -> str:
    """
    Retrieves the EXIF metadata entry from an image.

    :param filename: The path of the image file to read.
    :type filename: str
    :param tag: The EXIF metadata tag to fetch.
    :type tag: str
    :return: The value(s) associated with the given EXIF tag.
    :rtype: str
    :raises KeyError: If the EXIF tag is not found in the image metadata.
    """
    with open(filename, 'rb') as f:
        exif = exifread.process_file(f, details=False, stop_tag=tag)
    try:
        return exif[tag].values
    except KeyError:
        print("No EXIF metadata found")
        exit(1)
