import os
from typing import Callable

import exiv2
import numpy as np
import rawpy
from matplotlib import pyplot as plt

from ledsa.core.image_integrity import CorruptedImageError

IMG_FILE_EXTENSIONS = ['.JPG', '.JPEG', '.jpg', '.jpeg', '.PNG', '.png']
RAW_FILE_EXTENSIONS = ['.CR2', '.CR3']


def read_channel_data_from_img(filename: str, channel: int) -> np.ndarray:
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
    :raises CorruptedImageError: If the image cannot be read or has an unsupported file extension.
    """
    reader = _get_reader(filename, _read_channel_data_from_img_file, _read_channel_data_from_raw_file)
    return _read_or_raise(filename, reader, channel)

def read_img_array_from_img(filename: str, channel: int) -> np.ndarray:
    """
    Returns a 2D array of the image for all color channels.


    :param filename: The path of the image file to read.
    :type filename: str
    :param channel: The color channel used for calculating black level values.
    :type channel: int
    :return: A 2D array containing the processed image data for all color channels.
    :rtype: np.ndarray
    :raises CorruptedImageError: If the image cannot be read or has an unsupported file extension.
    """
    def read_raw_file(raw_filename, raw_channel):
        img_array, _ = _read_img_array_from_raw_file(raw_filename, raw_channel)
        return img_array

    reader = _get_reader(filename, lambda img_filename, _: _read_grayscale_img_array_from_img_file(img_filename),
                         read_raw_file)
    return _read_or_raise(filename, reader, channel)


def _get_reader(filename: str, img_file_reader: Callable, raw_file_reader: Callable) -> Callable:
    """
    Select the reader matching the file extension of an image.

    :param filename: The path of the image file to read.
    :type filename: str
    :param img_file_reader: Reader used for the JPG and PNG file extensions.
    :type img_file_reader: Callable
    :param raw_file_reader: Reader used for the RAW file extensions.
    :type raw_file_reader: Callable
    :return: The reader matching the file extension.
    :rtype: Callable
    :raises CorruptedImageError: If the file extension is not supported by LEDSA.
    """
    extension = os.path.splitext(filename)[-1]
    if extension in IMG_FILE_EXTENSIONS:
        return img_file_reader
    if extension in RAW_FILE_EXTENSIONS:
        return raw_file_reader
    supported = ', '.join(IMG_FILE_EXTENSIONS + RAW_FILE_EXTENSIONS)
    raise CorruptedImageError(filename, message=f"unsupported file extension '{extension}', "
                                                f"LEDSA can read {supported}")


def _read_or_raise(filename: str, reader: Callable, channel: int) -> np.ndarray:
    """
    Read an image and translate every error of the imaging library into a CorruptedImageError.

    The imaging libraries report a missing, empty, truncated or mistyped file through a wide range of error
    types, so they are caught collectively and reported with their original type and message, which keeps
    the cause of the problem visible in 'analysis/corrupted_images.csv'.

    :param filename: The path of the image file to read.
    :type filename: str
    :param reader: Reader matching the file extension of the image.
    :type reader: Callable
    :param channel: The color channel to process.
    :type channel: int
    :return: The image data returned by the reader.
    :rtype: np.ndarray
    :raises CorruptedImageError: If the reader fails on the image.
    """
    try:
        return reader(filename, channel)
    except Exception as error:
        raise CorruptedImageError(filename, error) from error


def _read_channel_data_from_img_file(filename: str, channel: int) -> np.ndarray:
    """
    Reads an image file and extracts a single color channel.

    :param filename: The file path of the image to be read.
    :type filename: str
    :param channel: The index of the color channel to extract (0 for Red, 1 for Green, 2 for Blue in RGB images).
    :type channel: int
    :return: A 2D numpy array containing the data of the specified color channel from the image.
    :rtype: np.ndarray
    """
    img_array = _read_img_array_from_img_file(filename)
    return img_array[:, :, channel]


def _read_channel_data_from_raw_file(filename: str, channel: int) -> np.ndarray:
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
    img_array, filter_array = _read_img_array_from_raw_file(filename, channel)
    if channel == 0 or channel == 2:
        channel_array = np.where(filter_array == channel, img_array, 0)
    elif channel == 1:
        channel_array = np.where((filter_array == 1) | (filter_array == 3), img_array, 0)
    return channel_array


def _read_img_array_from_raw_file(filename: str, channel: int) -> np.ndarray:
    # TODO: channel is only relevant for black level, consider individually!
    with rawpy.imread(filename) as raw:
        data = raw.raw_image_visible.copy()
        filter_array = raw.raw_colors_visible
        black_level = raw.black_level_per_channel[channel]
        white_level = raw.white_level
    img_array = data.astype(np.int16) - black_level
    img_array = (img_array * (white_level / (white_level - black_level))).astype(np.int16)
    img_array = np.clip(img_array, 0, white_level)
    return img_array, filter_array

def _read_img_array_from_img_file(filename: str) -> np.ndarray:
    img_array = plt.imread(filename)
    return img_array

def _read_grayscale_img_array_from_img_file(filename: str) -> np.ndarray:
    img_array = plt.imread(filename)
    weights = np.array([0.2989, 0.5870, 0.1140])
    gray = np.dot(img_array[..., :3], weights).astype(np.uint8)
    return gray


def get_exif_entry(filename: str, tag: str) -> str:
    """
    Retrieves the EXIF metadata entry from an image.

    :param filename: The path of the image file to read.
    :type filename: str
    :param tag: The EXIF metadata tag to fetch.
    :type tag: str
    :return: The value(s) associated with the given EXIF tag.
    :rtype: str
    :raises CorruptedImageError: If the metadata of the image cannot be read or does not contain the tag.
    """
    try:
        img = exiv2.ImageFactory.open(filename)
        img.readMetadata()
        exiv_data = img.exifData()
    except Exception as error:
        raise CorruptedImageError(filename, error) from error
    full_tag = f'Exif.Photo.{tag}'
    try:
        return exiv_data[full_tag].print()
    except KeyError as error:
        raise CorruptedImageError(filename, message=f'no EXIF entry {full_tag} found') from error
