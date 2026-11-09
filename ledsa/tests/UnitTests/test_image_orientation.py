"""Unit tests for reading images that were taken with the camera on its side.

A camera writes the pixels in the orientation of its sensor and notes how it was held in the
metadata, so a vertical photo is stored as a horizontal array. LEDSA turns the arrays upright
while it reads them, because the LED positions of the reference image and the search areas of
every other image have to be in one orientation.
"""
import exiv2
import numpy as np
import pytest
from PIL import Image, ImageOps

from ledsa.core import image_reading
from ledsa.core.image_reading import (ORIENTATION_TRANSFORMATIONS, _apply_orientation, _orientation_from_rotation,
                                      get_bayer_channel_mask,
                                      read_channel_data_mask_and_noise_from_img,
                                      read_channel_data_from_img, read_img_array_from_img)
from ledsa.data_extraction.step_1_functions import find_search_areas

# Orientation of an image that has to be turned 90 degrees clockwise to be upright, i.e. the one a camera
# writes when it is held vertically.
SIDEWAYS = 6


def write_img(path, img_array, orientation=None) -> str:
    """Write an array as a PNG file, with the EXIF orientation of a camera held on its side.

    PNG keeps the pixel values as they are, so an image read back with and without the rotation
    can be compared pixel by pixel.

    :param orientation: EXIF orientation to store, None to write no orientation entry at all.
    :return: The path of the written image.
    """
    Image.fromarray(img_array).save(str(path))
    if orientation is not None:
        img = exiv2.ImageFactory.open(str(path))
        img.readMetadata()
        exif_data = img.exifData()
        exif_data['Exif.Image.Orientation'] = str(orientation)
        img.setExifData(exif_data)
        img.writeMetadata()
    return str(path)


def led_strip_img_array(num_leds=5, spacing=20, width=40) -> np.ndarray:
    """Return an upright image of a vertical LED strip, as the experiments photograph it."""
    img_array = np.zeros((num_leds * spacing + spacing, width, 3), dtype=np.uint8)
    for led in range(num_leds):
        row, col = (led + 1) * spacing, width // 2
        img_array[row - 2:row + 3, col - 2:col + 3] = 200
    return img_array


class TestOrientationTransformations:
    @pytest.mark.parametrize('orientation', sorted(ORIENTATION_TRANSFORMATIONS))
    def test_transformation_matches_the_exif_standard(self, orientation):
        """Pillow turns an image by the same entry, which fixes the meaning of all eight values."""
        stored = np.arange(2 * 3 * 3, dtype=np.uint8).reshape(2, 3, 3)
        img = Image.fromarray(stored)
        exif = img.getexif()
        exif[0x0112] = orientation
        img.info['exif'] = exif.tobytes()
        np.testing.assert_array_equal(_apply_orientation(stored, orientation),
                                      np.asarray(ImageOps.exif_transpose(img)))

    @pytest.mark.parametrize('orientation', [0, 9])
    def test_unknown_orientation_keeps_the_array(self, orientation):
        """Cameras that cannot tell how they were held write a value outside the standard."""
        stored = np.arange(6).reshape(2, 3)
        np.testing.assert_array_equal(_apply_orientation(stored, orientation), stored)

    def test_upright_array_is_not_copied(self):
        stored = np.arange(6).reshape(2, 3)
        assert _apply_orientation(stored, 1) is stored

    def test_turned_array_is_contiguous(self):
        stored = np.arange(6).reshape(2, 3)
        assert _apply_orientation(stored, SIDEWAYS).flags['C_CONTIGUOUS']


class TestRotationOption:
    @pytest.mark.parametrize('rotation, orientation', [(0, 1), (90, 6), (180, 3), (270, 8),
                                                        ('90', 6), ('180.0', 3), (360, 1)])
    def test_rotation_is_translated_into_an_orientation(self, rotation, orientation):
        assert _orientation_from_rotation(rotation) == orientation

    @pytest.mark.parametrize('rotation', ['auto', 'AUTO', 'None', None, '', '  '])
    def test_auto_reads_the_orientation_from_the_image(self, rotation):
        """An older config file does not hold the option, which the parser reports as 'None'."""
        assert _orientation_from_rotation(rotation) is None

    @pytest.mark.parametrize('rotation', [45, '-1', 'upright', '90 180'])
    def test_unusable_rotation_is_rejected(self, rotation):
        with pytest.raises(ValueError, match='img_rotation'):
            _orientation_from_rotation(rotation)


class TestImgFileOrientation:
    def test_sideways_img_is_read_upright(self, tmp_path):
        upright = led_strip_img_array()
        stored = np.rot90(upright)  # as the camera wrote it, to be turned back clockwise
        path = write_img(tmp_path / 'sideways.png', stored, orientation=SIDEWAYS)
        channel_array = read_channel_data_from_img(path, channel=0)
        assert channel_array.shape == upright.shape[:2]
        np.testing.assert_allclose(channel_array, upright[:, :, 0] / 255, atol=1 / 255)

    def test_img_without_orientation_is_read_as_stored(self, tmp_path):
        stored = led_strip_img_array()
        path = write_img(tmp_path / 'upright.png', stored)
        np.testing.assert_allclose(read_channel_data_from_img(path, channel=0), stored[:, :, 0] / 255,
                                   atol=1 / 255)

    def test_all_channels_are_read_upright(self, tmp_path):
        stored = np.rot90(led_strip_img_array())
        path = write_img(tmp_path / 'sideways.png', stored, orientation=SIDEWAYS)
        upright_shape = read_img_array_from_img(path, channel=0).shape
        assert upright_shape == stored.shape[1::-1]

    def test_channel_mask_fits_the_upright_array(self, tmp_path):
        stored = np.rot90(led_strip_img_array())
        path = write_img(tmp_path / 'sideways.png', stored, orientation=SIDEWAYS)
        channel_array, channel_mask, _ = read_channel_data_mask_and_noise_from_img(path, channel=0)
        assert channel_mask.shape == channel_array.shape and channel_mask.all()

    def test_rotation_option_overrides_the_metadata(self, tmp_path):
        stored = np.rot90(led_strip_img_array())
        path = write_img(tmp_path / 'sideways.png', stored, orientation=SIDEWAYS)
        np.testing.assert_allclose(read_channel_data_from_img(path, channel=0, rotation=0), stored[:, :, 0] / 255,
                                   atol=1 / 255)

    def test_rotation_option_turns_an_image_without_metadata(self, tmp_path):
        """Images exported without their EXIF entries need the rotation from the config."""
        upright = led_strip_img_array()
        path = write_img(tmp_path / 'sideways.png', np.rot90(upright))
        channel_array = read_channel_data_from_img(path, channel=0, rotation=90)
        np.testing.assert_allclose(channel_array, upright[:, :, 0] / 255, atol=1 / 255)

    def test_unusable_rotation_is_not_reported_as_a_corrupted_image(self, tmp_path):
        path = write_img(tmp_path / 'upright.png', led_strip_img_array())
        with pytest.raises(ValueError, match='img_rotation'):
            read_channel_data_from_img(path, channel=0, rotation=45)

    def test_leds_are_found_at_the_same_positions_as_in_the_upright_image(self, tmp_path):
        """The symptom of a photo read sideways: the LED positions do not fit the LED arrays."""
        upright = led_strip_img_array()
        upright_path = write_img(tmp_path / 'upright.png', upright)
        sideways_path = write_img(tmp_path / 'sideways.png', np.rot90(upright), orientation=SIDEWAYS)
        # The LEDs cover a larger share of this image than of a photo, so they lie above a lower percentile
        search_areas, _ = find_search_areas(read_channel_data_from_img(upright_path, channel=0),
                                            search_area_radius=5, pixel_value_percentile=95, max_n_leds=10)
        turned_search_areas, _ = find_search_areas(read_channel_data_from_img(sideways_path, channel=0),
                                                    search_area_radius=5, pixel_value_percentile=95, max_n_leds=10)
        assert len(search_areas) == 5
        np.testing.assert_array_equal(turned_search_areas, search_areas)


class TestRawFileOrientation:
    # RGGB pattern as in rawpy raw_colors_visible: 0 red, 1 and 3 green, 2 blue
    FILTER_ARRAY = np.tile(np.array([[0, 1], [3, 2]]), (3, 2))

    class FakeRaw:
        """The part of a rawpy RawPy object that LEDSA reads, with a Bayer array of its own."""

        def __init__(self, raw_image_visible, filter_array, flip):
            self.raw_image_visible = raw_image_visible
            self.raw_colors_visible = filter_array
            # a sensor without an optical black margin, so the black level comes from the metadata
            self.sizes = type('Sizes', (), {'flip': flip, 'left_margin': 0, 'top_margin': 0,
                                            'height': raw_image_visible.shape[0]})
            self.black_level_per_channel = [0, 0, 0, 0]
            self.white_level = 255

        def __enter__(self):
            return self

        def __exit__(self, *args):
            return False

    @pytest.fixture
    def stored(self):
        return (np.arange(self.FILTER_ARRAY.size).reshape(self.FILTER_ARRAY.shape) % 256).astype(np.uint16)

    def fake_imread(self, monkeypatch, stored, flip):
        monkeypatch.setattr(image_reading.rawpy, 'imread',
                            lambda filename: self.FakeRaw(stored, self.FILTER_ARRAY, flip))

    def test_sideways_raw_is_read_upright(self, monkeypatch, stored):
        """LibRaw reports the orientation of the maker notes of the camera in sizes.flip."""
        self.fake_imread(monkeypatch, stored, flip=6)
        channel_array = read_channel_data_from_img('img.CR3', channel=0)
        upright = np.rot90(stored, -1)
        np.testing.assert_array_equal(channel_array, np.where(np.rot90(self.FILTER_ARRAY, -1) == 0, upright, 0))

    def test_bayer_pattern_is_turned_with_the_pixels(self, monkeypatch, stored):
        """The mask has to keep naming the color of every pixel, or the sums mix the channels."""
        self.fake_imread(monkeypatch, stored, flip=6)
        channel_array, channel_mask, _ = \
            read_channel_data_mask_and_noise_from_img('img.CR3', channel=1)
        np.testing.assert_array_equal(channel_mask, np.rot90(get_bayer_channel_mask(self.FILTER_ARRAY, 1), -1))
        np.testing.assert_array_equal(channel_array, np.where(channel_mask, np.rot90(stored, -1), 0))

    def test_upright_raw_is_read_as_stored(self, monkeypatch, stored):
        self.fake_imread(monkeypatch, stored, flip=0)
        channel_array = read_channel_data_from_img('img.CR3', channel=0)
        np.testing.assert_array_equal(channel_array, np.where(self.FILTER_ARRAY == 0, stored, 0))

    def test_rotation_option_overrides_the_raw_metadata(self, monkeypatch, stored):
        self.fake_imread(monkeypatch, stored, flip=6)
        channel_array = read_channel_data_from_img('img.CR3', channel=0, rotation=0)
        np.testing.assert_array_equal(channel_array, np.where(self.FILTER_ARRAY == 0, stored, 0))

    def test_all_channels_of_a_sideways_raw_are_read_upright(self, monkeypatch, stored):
        self.fake_imread(monkeypatch, stored, flip=6)
        np.testing.assert_array_equal(read_img_array_from_img('img.CR3', channel=0), np.rot90(stored, -1))
