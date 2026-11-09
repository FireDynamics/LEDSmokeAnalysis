import os
from datetime import datetime, timedelta
from typing import Callable, NamedTuple, Optional, Tuple

import exiv2
import numpy as np
import rawpy
from matplotlib import pyplot as plt

from ledsa.core.image_integrity import CorruptedImageError

# The extension of a file is matched lowercased, because cameras and operating systems differ in
# the case they write it in and Windows treats the two as the same file name anyway.
IMG_FILE_EXTENSIONS = ['.jpg', '.jpeg', '.png']
RAW_FILE_EXTENSIONS = ['.cr2', '.cr3', '.nef', '.arw', '.dng']

# The EXIF timestamps themselves only have second resolution. Cameras store the fraction of the second in a
# tag of its own, most of them as two digits, i.e. hundredths of a second.
SUBSEC_TAGS = {'DateTimeOriginal': 'SubSecTimeOriginal',
               'DateTimeDigitized': 'SubSecTimeDigitized'}
EXIF_DATETIME_FORMATS = ['%Y:%m:%d %H:%M:%S', '%d.%m.%Y %H:%M:%S']

# A camera turned on its side writes the pixels in the orientation of its sensor and notes how it was held in
# the metadata, so a vertical photo is stored as a horizontal array. The arrays are turned upright while they
# are read, which keeps the LED positions, the search areas and the plots of every image in one orientation.
AUTO_ROTATION = 'auto'
ORIENTATION_EXIF_KEY = 'Exif.Image.Orientation'
UPRIGHT_ORIENTATION = 1

# The transformation that makes the stored array upright, for every orientation EXIF defines. The numpy
# functions turn and mirror the first two axes, which leaves the color axis of an RGB array untouched.
ORIENTATION_TRANSFORMATIONS = {
    1: lambda img_array: img_array,                                 # upright
    2: lambda img_array: np.fliplr(img_array),                      # mirrored left to right
    3: lambda img_array: np.rot90(img_array, 2),                    # upside down
    4: lambda img_array: np.flipud(img_array),                      # mirrored top to bottom
    5: lambda img_array: np.swapaxes(img_array, 0, 1),              # mirrored along the main diagonal
    6: lambda img_array: np.rot90(img_array, -1),                   # to be turned clockwise
    7: lambda img_array: np.rot90(np.swapaxes(img_array, 0, 1), 2),  # mirrored along the second diagonal
    8: lambda img_array: np.rot90(img_array, 1),                    # to be turned counter-clockwise
}

# The orientation an image has if it must be turned by that many degrees clockwise to be upright, used for the
# img_rotation option of config.ini.
ROTATION_ORIENTATIONS = {0: 1, 90: 6, 180: 3, 270: 8}

# Orientation of the stored array as LibRaw reports it in sizes.flip of a RAW file. LibRaw derives it from the
# metadata of the camera maker, so it covers the RAW formats that do not store a plain EXIF orientation.
RAW_FLIP_ORIENTATIONS = {0: 1, 1: 2, 2: 4, 3: 3, 4: 5, 5: 8, 6: 6, 7: 7}


def read_channel_data_from_img(filename: str, channel: int, rotation: str = AUTO_ROTATION) -> np.ndarray:
    """
    Returns a 2D array of the image for a single color channel.
    8bit is default range for JPG. For RAW files the Bayer array is returned as a 2D array where
    all channel values except the selected channel are masked.

    :param filename: The path of the image file to read.
    :type filename: str
    :param channel: The color channel to process.
    :type channel: int
    :param rotation: Degrees the stored array is turned clockwise to be upright, 'auto' to take the rotation
                     from the metadata of the image.
    :type rotation: str or int
    :return: A 2D array containing the processed image data for one channel.
    :rtype: np.ndarray
    :raises CorruptedImageError: If the image cannot be read or has an unsupported file extension.
    :raises ValueError: If rotation is neither 'auto' nor one of 0, 90, 180 and 270.
    """
    orientation = _orientation_from_rotation(rotation)
    reader = _get_reader(filename, _read_channel_data_from_img_file, _read_channel_data_from_raw_file)
    return _read_or_raise(filename, reader, channel, orientation)


def read_channel_data_mask_and_noise_from_img(
        filename: str, channel: int,
        rotation: str = AUTO_ROTATION) -> Tuple[np.ndarray, np.ndarray, Optional[np.ndarray]]:
    """
    Returns the same 2D channel array as read_channel_data_from_img together with a boolean mask of the
    pixels that belong to the selected color channel and, for RAW images with an optical black margin, the
    read noise of every pixel.

    For RAW files the mask is taken from the Bayer filter pattern (raw_colors_visible) and not from the pixel
    values, because pixels of the selected channel can be 0 as well (noise clipped at 0 after the black level
    subtraction). For JPG/PNG files every pixel belongs to the channel.

    The noise carries 0 outside the channel, like the channel array itself, so that a sum over a search
    area and the noise of that sum need the mask no further.

    :param filename: The path of the image file to read.
    :type filename: str
    :param channel: The color channel to process.
    :type channel: int
    :param rotation: Degrees the stored array is turned clockwise to be upright, 'auto' to take the rotation
                     from the metadata of the image.
    :type rotation: str or int
    :return: The channel array, the boolean channel mask and the read noise of every pixel, all of the
        same shape. The noise is None for JPG and PNG images and for sensors without an optical black
        margin.
    :rtype: tuple[np.ndarray, np.ndarray, np.ndarray or None]
    :raises CorruptedImageError: If the image cannot be read or has an unsupported file extension.
    :raises ValueError: If rotation is neither 'auto' nor one of 0, 90, 180 and 270.
    """
    return ImageChannels(filename, rotation).channel_data_mask_and_noise(channel)


class ImageChannels:
    """
    An image read once for the analysis of several color channels.

    Decoding a RAW file takes most of the time of reading it, and nothing the decoding yields depends on the
    channel, because the black level and the read noise are taken per Bayer color. The file is therefore read
    when the object is created, and channel_data_mask_and_noise builds the arrays of a channel from it. They
    are built for one channel at a time, so that besides the decoded image only the channel in work is held in
    memory.

    The stored values of a RAW image are kept as they are, and the black level, the white level and the read
    noise are applied to the pixels that are asked for. Step 3 asks only for its search areas and side bands,
    a few percent of the sensor, which saves applying them to the whole image.
    """

    def __init__(self, filename: str, rotation: str = AUTO_ROTATION):
        """
        :param filename: The path of the image file to read.
        :type filename: str
        :param rotation: Degrees the stored array is turned clockwise to be upright, 'auto' to take the rotation
                         from the metadata of the image.
        :type rotation: str or int
        :raises CorruptedImageError: If the image cannot be read or has an unsupported file extension.
        :raises ValueError: If rotation is neither 'auto' nor one of 0, 90, 180 and 270.
        """
        def read_img_file(img_filename, _, img_orientation=None):
            return _read_img_array_from_img_file(img_filename, img_orientation)

        def read_raw_file(raw_filename, _, raw_orientation=None):
            raw = _read_raw_image(raw_filename)
            if raw_orientation is None:
                raw_orientation = raw.orientation
            # The Bayer pattern is turned with the pixels, so the filter array keeps naming the color of every pixel
            return raw._replace(data=_apply_orientation(raw.data, raw_orientation),
                                filter_array=_apply_orientation(raw.filter_array, raw_orientation),
                                orientation=UPRIGHT_ORIENTATION)

        self.filename = filename
        orientation = _orientation_from_rotation(rotation)
        reader = _get_reader(filename, read_img_file, read_raw_file)
        # no channel is passed, the read is the same for all of them
        img = _read_or_raise(filename, reader, None, orientation)
        self._raw = img if isinstance(img, RawImage) else None
        self._img_array = None if self._raw is not None else img
        # the calibrated pixels of the last selection, with the selection they belong to
        self._calibrated = None

    @property
    def shape(self) -> Tuple[int, int]:
        """The number of rows and columns of the upright image."""
        return (self._img_array if self._raw is None else self._raw.data).shape[:2]

    def channel_data_mask_and_noise(
            self, channel: int,
            pixels: Optional[Tuple[np.ndarray, np.ndarray]] = None
    ) -> Tuple[np.ndarray, np.ndarray, Optional[np.ndarray]]:
        """
        The channel array, the boolean channel mask and the read noise of every pixel of one color channel, as
        read_channel_data_mask_and_noise_from_img returns them, of the whole image or of selected pixels.

        The calibrated pixels of a RAW image do not depend on the channel. They are kept for the next call that
        passes the same selection, the same object, as the analysis of the next channel of an image does.

        :param channel: The color channel to process.
        :type channel: int
        :param pixels: Row and column indices of the selected pixels in the upright image, None for all pixels.
        :type pixels: tuple[np.ndarray, np.ndarray] or None
        :return: The channel array, the boolean channel mask and the read noise of every pixel, all of the
            shape of the image or of the index arrays. The noise is None for JPG and PNG images and for sensors
            without an optical black margin.
        :rtype: tuple[np.ndarray, np.ndarray, np.ndarray or None]
        :raises CorruptedImageError: If the image does not have the channel.
        """
        try:
            if self._raw is None:
                # every pixel of a JPG or PNG belongs to every channel, and there is no optical black to take the
                # read noise from
                channel_array = self._img_array[:, :, channel]
                if pixels is not None:
                    channel_array = channel_array[pixels]
                return channel_array, np.ones(channel_array.shape, dtype=bool), None
            img_array, filter_array, pixel_noise = self._calibrate(pixels)
            channel_mask = get_bayer_channel_mask(filter_array, channel)
            return (np.where(channel_mask, img_array, 0), channel_mask,
                    None if pixel_noise is None else np.where(channel_mask, pixel_noise, 0.0))
        except Exception as error:
            raise CorruptedImageError(self.filename, error) from error

    def _calibrate(self, pixels: Optional[Tuple[np.ndarray, np.ndarray]]
                   ) -> Tuple[np.ndarray, np.ndarray, Optional[np.ndarray]]:
        """
        The values with the black level subtracted and scaled to the white level, the Bayer colors and the read
        noise of the selected pixels of the RAW image, as _channel_data and _pixel_read_noise yield them for the
        whole image.
        """
        if self._calibrated is None or self._calibrated[0] is not pixels:
            raw = self._raw if pixels is None else self._raw._replace(data=self._raw.data[pixels],
                                                                      filter_array=self._raw.filter_array[pixels])
            # the selection is kept with the result, so that its identity cannot pass to another selection
            self._calibrated = (pixels, (_channel_data(raw), raw.filter_array, _pixel_read_noise(raw)))
        return self._calibrated[1]


def get_bayer_channel_mask(filter_array: np.ndarray, channel: int) -> np.ndarray:
    """
    Returns a boolean mask of the Bayer filter positions that belong to the given color channel.
    The Bayer array contains two green channels (1 and 3), one red (0) and one blue (2).

    :param filter_array: Bayer filter pattern of the raw image (rawpy raw_colors_visible).
    :type filter_array: np.ndarray
    :param channel: The color channel (0 red, 1 green, 2 blue).
    :type channel: int
    :return: Boolean mask, True where the pixel belongs to the channel.
    :rtype: np.ndarray
    """
    if channel == 1:
        return (filter_array == 1) | (filter_array == 3)
    return filter_array == channel

def read_img_array_from_img(filename: str, channel: int, rotation: str = AUTO_ROTATION) -> np.ndarray:
    """
    Returns a 2D array of the image for all color channels.


    :param filename: The path of the image file to read.
    :type filename: str
    :param channel: The color channel used for calculating black level values.
    :type channel: int
    :param rotation: Degrees the stored array is turned clockwise to be upright, 'auto' to take the rotation
                     from the metadata of the image.
    :type rotation: str or int
    :return: A 2D array containing the processed image data for all color channels.
    :rtype: np.ndarray
    :raises CorruptedImageError: If the image cannot be read or has an unsupported file extension.
    :raises ValueError: If rotation is neither 'auto' nor one of 0, 90, 180 and 270.
    """
    def read_img_file(img_filename, _, img_orientation=None):
        return _read_grayscale_img_array_from_img_file(img_filename, img_orientation)

    def read_raw_file(raw_filename, raw_channel, raw_orientation=None):
        img_array, _ = _read_img_array_from_raw_file(raw_filename, raw_channel, raw_orientation)
        return img_array

    orientation = _orientation_from_rotation(rotation)
    reader = _get_reader(filename, read_img_file, read_raw_file)
    return _read_or_raise(filename, reader, channel, orientation)


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
    if extension.lower() in IMG_FILE_EXTENSIONS:
        return img_file_reader
    if extension.lower() in RAW_FILE_EXTENSIONS:
        return raw_file_reader
    supported = ', '.join(IMG_FILE_EXTENSIONS + RAW_FILE_EXTENSIONS)
    raise CorruptedImageError(filename, message=f"unsupported file extension '{extension}', "
                                                f"LEDSA can read {supported} in any case")


def _read_or_raise(filename: str, reader: Callable, channel: int, orientation: Optional[int] = None) -> np.ndarray:
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
    :param orientation: EXIF orientation of the stored array, None to take it from the metadata of the image.
    :type orientation: int or None
    :return: The image data returned by the reader.
    :rtype: np.ndarray
    :raises CorruptedImageError: If the reader fails on the image.
    """
    try:
        return reader(filename, channel, orientation)
    except CorruptedImageError:
        raise  # the reader already named the image and the reason
    except Exception as error:
        raise CorruptedImageError(filename, error) from error


def _read_channel_data_from_img_file(filename: str, channel: int, orientation: Optional[int] = None) -> np.ndarray:
    """
    Reads an image file and extracts a single color channel.

    :param filename: The file path of the image to be read.
    :type filename: str
    :param channel: The index of the color channel to extract (0 for Red, 1 for Green, 2 for Blue in RGB images).
    :type channel: int
    :param orientation: EXIF orientation of the stored array, None to take it from the metadata of the image.
    :type orientation: int or None
    :return: A 2D numpy array containing the data of the specified color channel from the image.
    :rtype: np.ndarray
    """
    img_array = _read_img_array_from_img_file(filename, orientation)
    return img_array[:, :, channel]


def _read_channel_data_from_raw_file(filename: str, channel: int, orientation: Optional[int] = None) -> np.ndarray:
    """
    Extracts a single color channel from a RAW image file.

    :param filename: The file path of the RAW image to be read.
    :type filename: str
    :param channel: The index of the color channel to extract. The Bayer array typically contains two green channels (1 and 3),
                    one red (0), and one blue (2).
    :type channel: int
    :param orientation: EXIF orientation of the stored array, None to take it from the metadata of the image.
    :type orientation: int or None
    :return: A 2D numpy array representing the extracted channel, with all other channel values masked or set to zero.
    :rtype: np.ndarray
    """
    img_array, filter_array = _read_img_array_from_raw_file(filename, channel, orientation)
    if channel == 0 or channel == 2:
        channel_array = np.where(filter_array == channel, img_array, 0)
    elif channel == 1:
        channel_array = np.where((filter_array == 1) | (filter_array == 3), img_array, 0)
    return channel_array


# Minimum width of the optical black margin in pixels to estimate the black level from it
MIN_OPTICAL_BLACK_WIDTH = 8


def _optical_black_statistics(raw) -> Tuple[Optional[np.ndarray], Optional[np.ndarray]]:
    """
    Mean and standard deviation of the optical black pixels of a RAW image for every color of the Bayer
    pattern (index as in raw_colors). The optical black pixels are the masked pixels at the left margin of
    the sensor, which is read out with the image itself, so the mean is the black level of this image and
    the standard deviation is the read noise of a single pixel.

    :param raw: The opened RAW image (rawpy).
    :return: Mean and standard deviation per Bayer color, (None, None) if the sensor has no optical black
        margin of at least MIN_OPTICAL_BLACK_WIDTH pixels.
    :rtype: tuple[np.ndarray or None, np.ndarray or None]
    """
    sizes = raw.sizes
    if sizes.left_margin < MIN_OPTICAL_BLACK_WIDTH:
        return None, None
    rows = np.s_[sizes.top_margin:sizes.top_margin + sizes.height]
    margin = raw.raw_image[rows, :sizes.left_margin].astype(np.float64)
    colors = raw.raw_colors[rows, :sizes.left_margin]
    num_colors = int(raw.raw_colors_visible.max()) + 1
    per_color = [margin[colors == color] for color in range(num_colors)]
    return (np.array([values.mean() for values in per_color]),
            np.array([values.std() for values in per_color]))


class RawImage(NamedTuple):
    """What LEDSA reads from a RAW file, in one pass and before the orientation is applied."""

    data: np.ndarray                    # the visible sensor values as they are stored
    filter_array: np.ndarray            # the Bayer color of every pixel
    black_levels: np.ndarray            # per Bayer color
    white_level: float
    orientation: int                    # the one the camera recorded in its maker notes
    read_noise: Optional[np.ndarray]    # per Bayer color, None without an optical black margin

    @property
    def scales(self) -> np.ndarray:
        """Per Bayer color, the factor that maps the range above its black level onto the full range."""
        return self.white_level / (self.white_level - self.black_levels)


def _read_raw_image(filename: str) -> RawImage:
    """
    Read a RAW file once and keep the arrays and the scalars every version of its data is built from.

    Every color of the Bayer pattern gets its own black level, from its optical black pixels or, for
    sensors without an optical black margin, from the RAW metadata. Both sources hold one value per color.
    """
    with rawpy.imread(filename) as raw:
        optical_black_mean, read_noise = _optical_black_statistics(raw)
        black_levels = (np.asarray(raw.black_level_per_channel, dtype=np.float64)
                        if optical_black_mean is None else optical_black_mean)
        return RawImage(raw.raw_image_visible.copy(), raw.raw_colors_visible.copy(), black_levels,
                        float(raw.white_level),
                        RAW_FLIP_ORIENTATIONS.get(raw.sizes.flip, UPRIGHT_ORIENTATION), read_noise)


def _channel_data(raw: RawImage) -> np.ndarray:
    """
    The sensor values with the black level of their own Bayer color subtracted and scaled to the white
    level, with no lower limit.

    Clipping the values at zero would turn the negative half of the read noise into a positive offset of
    the integrated intensity of every LED. That offset stays while the LED dims in smoke, so it saturates
    the extinction coefficients in dense smoke, where it was 0.3 to 0.7 % of I0 in the Warsaw tests.
    Values below the black level therefore stay negative. The array is float because the black level is a
    fraction of a count, which is the size of the effect.
    """
    levels = raw.black_levels.astype(np.float32)[raw.filter_array]
    scales = raw.scales.astype(np.float32)[raw.filter_array]
    return (raw.data.astype(np.float32) - levels) * scales


def _pixel_read_noise(raw: RawImage) -> Optional[np.ndarray]:
    """
    The read noise of every single pixel, taken from the optical black pixels of its own Bayer color and
    scaled like the data. None for sensors without an optical black margin.
    """
    if raw.read_noise is None:
        return None
    return (raw.read_noise.astype(np.float32) * raw.scales.astype(np.float32))[raw.filter_array]


def _read_img_array_from_raw_file(filename: str, channel: int,
                                  orientation: Optional[int] = None) -> Tuple[np.ndarray, np.ndarray]:
    # channel is kept for the signature of the readers; the black level is taken per Bayer color
    raw = _read_raw_image(filename)
    if orientation is None:
        orientation = raw.orientation
    # The Bayer pattern is turned with the pixels, so the filter array keeps naming the color of every pixel
    return (_apply_orientation(_channel_data(raw), orientation),
            _apply_orientation(raw.filter_array, orientation))


def _read_img_array_from_img_file(filename: str, orientation: Optional[int] = None) -> np.ndarray:
    img_array = plt.imread(filename)
    if orientation is None:
        orientation = _read_orientation_from_exif(filename)
    return _apply_orientation(img_array, orientation)

def _read_grayscale_img_array_from_img_file(filename: str, orientation: Optional[int] = None) -> np.ndarray:
    img_array = _read_img_array_from_img_file(filename, orientation)
    weights = np.array([0.2989, 0.5870, 0.1140])
    gray = np.dot(img_array[..., :3], weights).astype(np.uint8)
    return gray


def _orientation_from_rotation(rotation) -> Optional[int]:
    """
    Translate the rotation an analysis is configured with into an EXIF orientation.

    :param rotation: Degrees the stored array is turned clockwise to be upright, 'auto' (or 'None', which an
                     older config file leaves the option at) to take the rotation from the metadata.
    :type rotation: str or int or None
    :return: The orientation of every image of the experiment, None if it is read from each image.
    :rtype: int or None
    :raises ValueError: If rotation is neither 'auto' nor one of 0, 90, 180 and 270.
    """
    text = str(rotation).strip().lower()
    if text in (AUTO_ROTATION, 'none', ''):
        return None
    try:
        degrees = float(text)
    except ValueError:
        degrees = None
    if degrees is None or degrees % 90 != 0 or int(degrees) % 360 not in ROTATION_ORIENTATIONS:
        raise ValueError(f"img_rotation must be '{AUTO_ROTATION}' or one of "
                         f"{', '.join(str(d) for d in ROTATION_ORIENTATIONS)}, got '{rotation}'")
    return ROTATION_ORIENTATIONS[int(degrees) % 360]


def _read_orientation_from_exif(filename: str) -> int:
    """
    Read the orientation of the stored pixel array from the metadata of a JPG or PNG file.

    An image without the entry is taken to be upright, which is how cameras that cannot tell how they were
    held write their photos and how every image was read before the orientation was taken into account.

    :param filename: The path of the image file to read.
    :type filename: str
    :return: The EXIF orientation of the image, 1 if it does not carry one.
    :rtype: int
    :raises CorruptedImageError: If the metadata of the image cannot be read.
    """
    img, exif_data = _read_exif_data(filename)  # img keeps exif_data alive
    entry = exif_data.findKey(exiv2.ExifKey(ORIENTATION_EXIF_KEY))
    if entry == exif_data.end():
        return UPRIGHT_ORIENTATION
    try:
        return int(entry.toInt64())
    except Exception:  # an entry of an unexpected type says nothing about the orientation
        return UPRIGHT_ORIENTATION


def _apply_orientation(img_array: np.ndarray, orientation: int) -> np.ndarray:
    """
    Turn the stored pixel array of an image upright.

    :param img_array: The pixel array as it is stored in the image file.
    :type img_array: np.ndarray
    :param orientation: The EXIF orientation of the stored array. The values 0 and 9, which cameras write if
                        they do not know how they were held, leave the array as it is.
    :type orientation: int
    :return: The upright array.
    :rtype: np.ndarray
    """
    transformation = ORIENTATION_TRANSFORMATIONS.get(int(orientation))
    if transformation is None or int(orientation) == UPRIGHT_ORIENTATION:
        return img_array
    # The transformations return a view with reordered axes, whose pixels the analysis would read across the
    # memory layout of the stored array, so the upright array is laid out as it is read.
    return np.ascontiguousarray(transformation(img_array))


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
    img, exif_data = _read_exif_data(filename)  # img keeps exif_data alive
    return _find_exif_entry(exif_data, filename, tag)


def get_exif_datetime(filename: str, tag: str = 'DateTimeOriginal') -> datetime:
    """
    Retrieves an EXIF timestamp of an image with sub-second resolution.

    EXIF stores the timestamps with second resolution only, so images taken less than a second apart would
    carry the same time. The fraction of the second is kept in a separate entry, which is added here if the
    camera wrote one.

    :param filename: The path of the image file to read.
    :type filename: str
    :param tag: The EXIF timestamp to fetch. Defaults to 'DateTimeOriginal'.
    :type tag: str
    :return: The timestamp, including the sub-second part if the camera recorded one.
    :rtype: datetime.datetime
    :raises CorruptedImageError: If the metadata of the image cannot be read, does not contain the timestamp
                                 or contains it in an unknown format.
    """
    img, exif_data = _read_exif_data(filename)  # img keeps exif_data alive
    entry = _find_exif_entry(exif_data, filename, tag)
    date_time = _parse_exif_datetime(entry, filename, tag)
    subsec_tag = SUBSEC_TAGS.get(tag)
    if subsec_tag is None:
        return date_time
    subsec_entry = _find_exif_entry(exif_data, filename, subsec_tag, required=False)
    return date_time + timedelta(seconds=_subsec_entry_to_seconds(subsec_entry))


def _read_exif_data(filename: str) -> Tuple['exiv2.Image', 'exiv2.ExifData']:
    """
    Open an image and return it together with its EXIF metadata.

    The image is returned as well because the metadata is only a view into it, which stays valid as long as
    the image object is referenced.

    :param filename: The path of the image file to read.
    :type filename: str
    :return: The opened image and its EXIF metadata.
    :rtype: Tuple[exiv2.Image, exiv2.ExifData]
    :raises CorruptedImageError: If the metadata of the image cannot be read.
    """
    try:
        img = exiv2.ImageFactory.open(filename)
        img.readMetadata()
        return img, img.exifData()
    except Exception as error:
        raise CorruptedImageError(filename, error) from error


def _find_exif_entry(exif_data: 'exiv2.ExifData', filename: str, tag: str, required: bool = True) -> str:
    """
    Look up a single entry in the EXIF metadata of an image.

    :param exif_data: The EXIF metadata to search.
    :type exif_data: exiv2.ExifData
    :param filename: The path of the image the metadata belongs to, used for error messages.
    :type filename: str
    :param tag: The EXIF metadata tag to fetch.
    :type tag: str
    :param required: Whether a missing entry is an error instead of an empty value.
    :type required: bool
    :return: The value(s) associated with the given EXIF tag, or an empty string if an optional tag is not set.
    :rtype: str
    :raises CorruptedImageError: If a required tag is not set.
    """
    full_tag = f'Exif.Photo.{tag}'
    entry = exif_data.findKey(exiv2.ExifKey(full_tag))
    if entry == exif_data.end():
        if required:
            raise CorruptedImageError(filename, message=f'no EXIF entry {full_tag} found')
        return ''
    return entry.print()


def _parse_exif_datetime(entry: str, filename: str, tag: str) -> datetime:
    """
    Convert an EXIF timestamp entry into a datetime object.

    :param entry: The value of the EXIF timestamp.
    :type entry: str
    :param filename: The path of the image the timestamp belongs to, used for error messages.
    :type filename: str
    :param tag: The EXIF metadata tag the timestamp was read from, used for error messages.
    :type tag: str
    :return: The corresponding datetime object.
    :rtype: datetime.datetime
    :raises CorruptedImageError: If the timestamp is not in one of the known formats.
    """
    for datetime_format in EXIF_DATETIME_FORMATS:
        try:
            return datetime.strptime(entry.strip(), datetime_format)
        except ValueError:
            continue
    raise CorruptedImageError(filename, message=f'EXIF entry Exif.Photo.{tag} is not a timestamp: {entry}')


def _subsec_entry_to_seconds(entry: str) -> float:
    """
    Convert a SubSecTime entry into a fraction of a second.

    The entry holds the digits behind the decimal point, so '07' means 0.07 s. How many digits a camera
    writes differs between models, and cameras which do not record the fraction leave the entry empty.

    :param entry: The value of the SubSecTime entry.
    :type entry: str
    :return: The fraction of the second in seconds, 0.0 if the entry holds no digits.
    :rtype: float
    """
    digits = entry.strip(' \t\r\n\x00')
    if not digits.isdigit():
        return 0.0
    return float(f'0.{digits}')
