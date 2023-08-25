import os

import exifread
import numpy as np
import rawpy
from matplotlib import pyplot as plt


def read_img(filename: str, channel: int, color_depth=14) -> np.ndarray:
    """
    Returns a 2D array of the image for a single color channel.
    8bit is default range for JPG. For RAW files the Bayer array is returned as a 2D array where
    all channel values except the selected channel are masked.

    :param filename: The path of the image file to read.
    :type filename: str
    :param channel: The color channel to process.
    :type channel: int
    :param color_depth: The bit depth of the image color. Default is 14 for RAW images.
    :type color_depth: int
    :return: A 2D array containing the processed image data for one channel.
    :rtype: np.ndarray
    """
    extension = os.path.splitext(filename)[-1]
    data = []
    if extension in ['.JPG', '.JPEG', '.jpg', '.jpeg', '.PNG', '.png']:
        data = plt.imread(filename)
    elif extension in ['.CR2']:
        with rawpy.imread(filename) as raw:
            data = raw.raw_image_visible.copy()
            filter_array = raw.raw_colors_visible
            black_level = raw.black_level_per_channel[channel]
            white_level = raw.white_level
        channel_range = 2 ** color_depth - 1
        channel_array = data.astype(np.int16) - black_level
        channel_array = (channel_array * (channel_range / (white_level - black_level))).astype(np.int16)
        channel_array = np.clip(channel_array, 0, channel_range)
        if channel == 0 or channel == 2:
            channel_array = np.where(filter_array == channel, channel_array, 0)
        elif channel == 1:
            channel_array = np.where((filter_array == 1) | (filter_array == 3), channel_array, 0)
        return channel_array
    return data[:, :, channel]


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
