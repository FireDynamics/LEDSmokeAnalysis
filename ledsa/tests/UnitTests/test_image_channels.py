"""Unit tests for reading an image once for the analysis of several color channels.

Decoding a RAW file takes most of the time of reading it, so step 3 reads every image once and builds
the arrays of all channels it analyses from that one read. The arrays have to be the same as those of
a read for each channel.
"""
import configparser

import matplotlib.pyplot as plt
import numpy as np
import pytest
from PIL import Image

from ledsa.core import image_handling, image_reading
from ledsa.core.image_integrity import CorruptedImageError
from ledsa.core.image_reading import ImageChannels, RawImage
from ledsa.data_extraction import step_3_functions
from ledsa.data_extraction.DataExtractor import DataExtractor

# RGGB pattern as in rawpy raw_colors_visible: 0 red, 1 and 3 green, 2 blue
FILTER_ARRAY = np.tile(np.array([[0, 1], [3, 2]]), (2, 3))
CHANNEL_COLORS = {0: [0], 1: [1, 3], 2: [2]}
# different per Bayer color, so that a test can tell the noise of one channel from that of another
READ_NOISE = np.array([1.0, 2.0, 3.0, 4.0])
SIDEWAYS = 6


@pytest.fixture
def stored():
    return np.arange(FILTER_ARRAY.size, dtype=np.float32).reshape(FILTER_ARRAY.shape)


@pytest.fixture
def raw_reads(monkeypatch, stored):
    """
    Let the one pass over a RAW file return a prepared image and count how often it is made. A black level
    of 0 and a white level of 255 leave the stored values as they are.
    """
    def install(orientation=image_reading.UPRIGHT_ORIENTATION):
        reads = []

        def read_raw_image(filename):
            reads.append(filename)
            return RawImage(stored, FILTER_ARRAY, np.zeros(4), 255.0, orientation, READ_NOISE)

        monkeypatch.setattr(image_reading, '_read_raw_image', read_raw_image)
        return reads
    return install


class TestImageChannels:
    @pytest.mark.parametrize('orientation', [image_reading.UPRIGHT_ORIENTATION, SIDEWAYS])
    def test_raw_channels_are_built_from_one_read(self, raw_reads, stored, orientation):
        reads = raw_reads(orientation)
        img_channels = ImageChannels('img.CR3')
        upright = image_reading.ORIENTATION_TRANSFORMATIONS[orientation]
        for channel, colors in CHANNEL_COLORS.items():
            channel_array, channel_mask, pixel_noise = img_channels.channel_data_mask_and_noise(channel)
            mask = upright(np.isin(FILTER_ARRAY, colors))
            np.testing.assert_array_equal(channel_mask, mask)
            np.testing.assert_array_equal(channel_array, np.where(mask, upright(stored), 0))
            np.testing.assert_array_equal(pixel_noise, np.where(mask, upright(READ_NOISE[FILTER_ARRAY]), 0))
        assert reads == ['img.CR3']

    def test_raw_channels_are_those_of_a_read_per_channel(self, raw_reads):
        raw_reads(SIDEWAYS)
        img_channels = ImageChannels('img.CR3')
        for channel in CHANNEL_COLORS:
            for once, per_channel in zip(img_channels.channel_data_mask_and_noise(channel),
                                         image_reading.read_channel_data_mask_and_noise_from_img('img.CR3', channel)):
                np.testing.assert_array_equal(once, per_channel)
            np.testing.assert_array_equal(img_channels.channel_data_mask_and_noise(channel)[0],
                                          image_reading.read_channel_data_from_img('img.CR3', channel))

    def test_rotation_option_overrides_the_raw_metadata(self, raw_reads, stored):
        raw_reads(SIDEWAYS)
        channel_array, _, _ = ImageChannels('img.CR3', rotation=0).channel_data_mask_and_noise(0)
        np.testing.assert_array_equal(channel_array, np.where(FILTER_ARRAY == 0, stored, 0))

    def test_png_channels_are_the_color_planes(self, tmp_path):
        path = str(tmp_path / 'img.png')
        plt.imsave(path, np.random.default_rng(0).random((6, 8, 3)))
        img_channels = ImageChannels(path)
        for channel in CHANNEL_COLORS:
            channel_array, channel_mask, pixel_noise = img_channels.channel_data_mask_and_noise(channel)
            np.testing.assert_array_equal(channel_array, image_reading.read_channel_data_from_img(path, channel))
            assert channel_mask.shape == channel_array.shape and channel_mask.all()
            assert pixel_noise is None  # a PNG has no optical black

    def test_missing_channel_is_reported_as_corrupted_image(self, tmp_path):
        path = str(tmp_path / 'gray.png')
        Image.fromarray(np.zeros((6, 8), dtype=np.uint8)).save(path)  # read as a 2D array without color planes
        img_channels = ImageChannels(path)
        with pytest.raises(CorruptedImageError, match='gray.png'):
            img_channels.channel_data_mask_and_noise(1)

    def test_unreadable_image_raises_when_it_is_read(self, tmp_path):
        with pytest.raises(CorruptedImageError, match='missing.CR3'):
            ImageChannels(str(tmp_path / 'missing.CR3'))

    def test_unsupported_format_raises(self):
        with pytest.raises(CorruptedImageError, match='unsupported file extension'):
            ImageChannels('img.xyz')


class TestSelectedPixels:
    @pytest.mark.parametrize('orientation', [image_reading.UPRIGHT_ORIENTATION, SIDEWAYS])
    def test_raw_pixels_are_those_of_the_whole_image(self, raw_reads, orientation):
        raw_reads(orientation)
        img_channels = ImageChannels('img.CR3')
        pixels = (np.array([0, 3, 1, 1]), np.array([2, 0, 3, 3]))  # in the upright image, one of them twice
        for channel in CHANNEL_COLORS:
            for selected, whole in zip(img_channels.channel_data_mask_and_noise(channel, pixels),
                                       img_channels.channel_data_mask_and_noise(channel)):
                np.testing.assert_array_equal(selected, whole[pixels])

    def test_png_pixels_are_those_of_the_color_planes(self, tmp_path):
        path = str(tmp_path / 'img.png')
        plt.imsave(path, np.random.default_rng(0).random((6, 8, 3)))
        img_channels = ImageChannels(path)
        pixels = (np.array([[0, 5], [2, 2]]), np.array([[7, 0], [3, 4]]))  # index arrays of any shape
        for channel in CHANNEL_COLORS:
            channel_array, channel_mask, pixel_noise = img_channels.channel_data_mask_and_noise(channel, pixels)
            np.testing.assert_array_equal(channel_array, img_channels.channel_data_mask_and_noise(channel)[0][pixels])
            assert channel_mask.shape == channel_array.shape == (2, 2) and channel_mask.all()
            assert pixel_noise is None

    def test_the_calibration_is_kept_for_the_same_selection_only(self, raw_reads, monkeypatch):
        raw_reads()
        calibrated = []
        channel_data = image_reading._channel_data
        monkeypatch.setattr(image_reading, '_channel_data',
                            lambda raw: calibrated.append(raw.data.shape) or channel_data(raw))
        img_channels = ImageChannels('img.CR3')
        pixels = (np.array([0, 1]), np.array([2, 3]))
        for channel in CHANNEL_COLORS:
            img_channels.channel_data_mask_and_noise(channel, pixels)
        assert calibrated == [(2,)]
        # another selection, even of the same pixels, and the whole image are calibrated anew
        img_channels.channel_data_mask_and_noise(0, (pixels[0].copy(), pixels[1].copy()))
        img_channels.channel_data_mask_and_noise(0)
        img_channels.channel_data_mask_and_noise(1)
        assert calibrated == [(2,), (2,), FILTER_ARRAY.shape]

    def test_shape_is_that_of_the_upright_image(self, raw_reads):
        raw_reads(SIDEWAYS)
        assert ImageChannels('img.CR3').shape == FILTER_ARRAY.shape[::-1]
        assert ImageChannels('img.CR3', rotation=0).shape == FILTER_ARRAY.shape


class TestStep3ReadsEveryImageOnce:
    @pytest.fixture
    def extractor(self, tmp_path, monkeypatch):
        """A DataExtractor for three channels, without the config file and the results of steps 1 and 2."""
        monkeypatch.chdir(tmp_path)
        monkeypatch.setattr(image_handling, 'get_img_id', lambda img_filename: '42')
        config = configparser.ConfigParser()
        config['DEFAULT'] = {'img_directory': str(tmp_path), 'img_rotation': 'auto'}
        extractor = DataExtractor.__new__(DataExtractor)
        extractor.channels = [0, 1, 2]
        extractor.config = config
        extractor.search_areas = np.zeros((1, 3))
        extractor.line_indices = [[0]]
        extractor.fit_leds = False
        return extractor

    def test_all_channels_are_analysed_from_one_read(self, extractor, raw_reads, monkeypatch):
        reads = raw_reads()
        analysed, written = [], []

        def generate_analysis_data(img_filename, channel, *args, img_channels=None, **kwargs):
            analysed.append(channel)
            return img_channels.channel_data_mask_and_noise(channel)

        monkeypatch.setattr(step_3_functions, 'generate_analysis_data', generate_analysis_data)
        monkeypatch.setattr(step_3_functions, 'create_fit_result_file',
                            lambda img_data, img_id, channel: written.append((img_id, channel)))
        assert extractor.process_img_file('V001_Cam01_0042.CR3') is None
        assert len(reads) == 1
        assert analysed == [0, 1, 2]
        assert written == [('42', 0), ('42', 1), ('42', 2)]

    def test_unreadable_image_is_skipped_for_all_channels(self, extractor, monkeypatch):
        written = []
        monkeypatch.setattr(step_3_functions, 'create_fit_result_file',
                            lambda img_data, img_id, channel: written.append(channel))
        error = extractor.process_img_file('V001_Cam01_0042.CR3')  # there is no such file
        assert isinstance(error, CorruptedImageError)
        assert written == []
