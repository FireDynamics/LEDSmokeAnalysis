from types import SimpleNamespace

import numpy as np
import pytest

import ledsa.core.image_reading as image_reading
from ledsa.core.image_reading import (MIN_OPTICAL_BLACK_WIDTH, UPRIGHT_ORIENTATION, _black_level,
                                      _optical_black_statistics,
                                      _read_img_array_and_optical_black_from_raw_file,
                                      _read_img_array_from_raw_file)

# RGGB pattern as in rawpy raw_colors: 0 red, 1 and 3 green, 2 blue
BAYER_TILE = np.array([[0, 1], [3, 2]])
# black level of every Bayer color in the optical black margin, far from the nominal value below
OPTICAL_BLACK_LEVELS = [100.0, 101.0, 102.0, 103.0]
METADATA_BLACK_LEVELS = [2047, 2047, 2047, 2047]
WHITE_LEVEL = 16383


def make_raw(left_margin: int, noise: float = 0.0, visible: float = None) -> SimpleNamespace:
    """
    The part of a rawpy RawPy object that LEDSA reads: 4 x 4 visible pixels behind a top and a left margin,
    with a known black level per Bayer color in the margin. With noise the two halves of the margin columns
    are offset by -noise and +noise, so the standard deviation of every color is noise. With visible the
    visible pixels are set to that value instead of their black level.
    """
    top_margin, height, width = 2, 4, 4
    shape = (top_margin + height, left_margin + width)
    colors = np.tile(BAYER_TILE, (shape[0] // 2 + 1, shape[1] // 2 + 1))[:shape[0], :shape[1]]
    raw_image = np.zeros(shape, dtype=np.float64)
    for color, level in enumerate(OPTICAL_BLACK_LEVELS):
        raw_image[colors == color] = level
    if noise and left_margin:
        raw_image[:, :left_margin] += np.where(np.arange(left_margin) < left_margin // 2, -noise, noise)
    if visible is not None:
        raw_image[top_margin:, left_margin:] = visible
    raw = SimpleNamespace(
        sizes=SimpleNamespace(left_margin=left_margin, top_margin=top_margin, height=height, width=width,
                              flip=0),
        raw_image=raw_image,
        raw_image_visible=raw_image[top_margin:, left_margin:],
        raw_colors=colors,
        raw_colors_visible=colors[top_margin:, left_margin:],
        black_level_per_channel=METADATA_BLACK_LEVELS,
        white_level=WHITE_LEVEL,
    )
    raw.__enter__ = lambda: raw
    raw.__exit__ = lambda *args: False
    return raw


@pytest.fixture
def fake_imread(monkeypatch):
    """Let rawpy.imread return a prepared fake RAW image."""
    def install(raw):
        class Opened:
            def __enter__(self):
                return raw

            def __exit__(self, *args):
                return False

        monkeypatch.setattr(image_reading.rawpy, 'imread', lambda filename: Opened())
    return install


class TestOpticalBlackStatistics:
    def test_mean_and_standard_deviation_per_bayer_color(self):
        mean, std = _optical_black_statistics(make_raw(MIN_OPTICAL_BLACK_WIDTH, noise=3.0))
        assert mean == pytest.approx(OPTICAL_BLACK_LEVELS)
        assert std == pytest.approx([3.0] * len(OPTICAL_BLACK_LEVELS))

    def test_only_the_margin_enters_the_statistics(self):
        # the visible pixels of a bright image must not raise the black level or the noise
        mean, std = _optical_black_statistics(make_raw(MIN_OPTICAL_BLACK_WIDTH, visible=9000.0))
        assert mean == pytest.approx(OPTICAL_BLACK_LEVELS)
        assert std == pytest.approx([0.0] * len(OPTICAL_BLACK_LEVELS))

    def test_margin_narrower_than_the_minimum_is_not_used(self):
        assert _optical_black_statistics(make_raw(MIN_OPTICAL_BLACK_WIDTH - 2)) == (None, None)

    def test_sensor_without_a_margin(self):
        assert _optical_black_statistics(make_raw(0)) == (None, None)


class TestBlackLevel:
    @pytest.mark.parametrize('channel', [0, 1, 2])
    def test_optical_black_is_preferred_over_the_metadata(self, channel):
        # the metadata hold a nominal value of the camera model, the optical black pixels are read out with
        # the image itself
        assert _black_level(make_raw(MIN_OPTICAL_BLACK_WIDTH), channel) == \
               pytest.approx(OPTICAL_BLACK_LEVELS[channel])

    @pytest.mark.parametrize('channel', [0, 1, 2])
    def test_metadata_is_the_fallback_without_an_optical_black_margin(self, channel):
        assert _black_level(make_raw(MIN_OPTICAL_BLACK_WIDTH - 1), channel) == \
               pytest.approx(float(METADATA_BLACK_LEVELS[channel]))

    def test_the_black_level_is_a_float_in_both_cases(self):
        # it is subtracted from the image and enters the scale factor, where an int would round the
        # fraction of a count away that the estimate is about
        assert isinstance(_black_level(make_raw(MIN_OPTICAL_BLACK_WIDTH), 0), float)
        assert isinstance(_black_level(make_raw(0), 0), float)


class TestUnclippedData:
    CHANNEL = 0

    def test_values_below_the_black_level_stay_negative(self, fake_imread):
        # the visible pixels read 2 counts below the black level of their color. Only the pixels of the
        # selected channel are compared, because the clipped version subtracts one black level from the
        # whole Bayer array (see the TODO on _read_raw_image).
        raw = make_raw(MIN_OPTICAL_BLACK_WIDTH)
        raw.raw_image[raw.sizes.top_margin:, MIN_OPTICAL_BLACK_WIDTH:] -= 2.0
        fake_imread(raw)
        clipped, filter_array, unclipped, _ = \
            _read_img_array_and_optical_black_from_raw_file('img.CR3', self.CHANNEL)
        mask = filter_array == self.CHANNEL
        scale = WHITE_LEVEL / (WHITE_LEVEL - OPTICAL_BLACK_LEVELS[self.CHANNEL])
        assert (clipped[mask] == 0).all()  # the whole signal is gone, and this is the offset it leaves
        assert unclipped[mask] == pytest.approx(np.full(mask.sum(), -2.0 * scale), rel=1e-5)

    def test_every_bayer_color_gets_its_own_black_level(self, fake_imread):
        # the visible pixels sit exactly at the black level of their own color, so all of them are 0
        fake_imread(make_raw(MIN_OPTICAL_BLACK_WIDTH))
        _, _, unclipped, _ = _read_img_array_and_optical_black_from_raw_file('img.CR3', self.CHANNEL)
        assert unclipped == pytest.approx(np.zeros(unclipped.shape))

    def test_the_noise_of_a_pixel_is_the_one_of_its_color(self, fake_imread):
        fake_imread(make_raw(MIN_OPTICAL_BLACK_WIDTH, noise=4.0))
        _, filter_array, _, noise = _read_img_array_and_optical_black_from_raw_file('img.CR3', self.CHANNEL)
        scale = WHITE_LEVEL / (WHITE_LEVEL - OPTICAL_BLACK_LEVELS[self.CHANNEL])
        assert noise.shape == filter_array.shape
        assert noise == pytest.approx(np.full(noise.shape, 4.0 * scale), rel=1e-5)

    def test_the_clipped_version_is_the_one_every_other_reader_gets(self, fake_imread):
        fake_imread(make_raw(MIN_OPTICAL_BLACK_WIDTH, visible=5000.0))
        clipped, filter_array, _, _ = _read_img_array_and_optical_black_from_raw_file('img.CR3', self.CHANNEL)
        expected, expected_filter = _read_img_array_from_raw_file('img.CR3', self.CHANNEL)
        np.testing.assert_array_equal(clipped, expected)
        np.testing.assert_array_equal(filter_array, expected_filter)

    def test_without_an_optical_black_margin_there_is_nothing_to_return(self, fake_imread):
        fake_imread(make_raw(MIN_OPTICAL_BLACK_WIDTH - 1, visible=5000.0))
        clipped, filter_array, unclipped, noise = \
            _read_img_array_and_optical_black_from_raw_file('img.CR3', self.CHANNEL)
        assert unclipped is None and noise is None
        assert clipped.shape == filter_array.shape

    def test_the_orientation_reaches_all_four_arrays(self, fake_imread):
        raw = make_raw(MIN_OPTICAL_BLACK_WIDTH, noise=4.0, visible=5000.0)
        fake_imread(raw)
        upright = _read_img_array_and_optical_black_from_raw_file('img.CR3', self.CHANNEL,
                                                                 UPRIGHT_ORIENTATION)
        turned = _read_img_array_and_optical_black_from_raw_file('img.CR3', self.CHANNEL, 6)
        for upright_array, turned_array in zip(upright, turned):
            np.testing.assert_array_equal(turned_array, np.rot90(upright_array, -1))


class TestOpticalBlackSumsOfAnLed:
    """The two columns step 3 writes from the unclipped data."""

    SEARCH_AREA_RADIUS = 6
    CENTER = (20, 20)

    def _extract(self, data, ob_data=None, ob_noise=None):
        from ledsa.data_extraction.step_3_functions import _generate_led_analysis_data
        search_areas = np.array([[0, self.CENTER[0], self.CENTER[1]]])
        return _generate_led_analysis_data(None, 0, data, False, 0, 'img', 0, search_areas,
                                           self.SEARCH_AREA_RADIUS, fit_leds=False,
                                           ob_data=ob_data, ob_noise=ob_noise)

    @property
    def _box(self):
        radius = self.SEARCH_AREA_RADIUS
        return np.index_exp[self.CENTER[0] - radius:self.CENTER[0] + radius,
                            self.CENTER[1] - radius:self.CENTER[1] + radius]

    def test_without_optical_black_the_columns_stay_nan(self):
        led_data = self._extract(np.full((40, 40), 5.0))
        assert np.isnan(led_data.ob_sum_color_value)
        assert np.isnan(led_data.ob_noise_sum_color_value)

    def test_the_sum_runs_over_the_search_area(self):
        data = np.full((40, 40), 5.0)
        ob_data = np.zeros((40, 40), dtype=np.float32)
        ob_data[self._box] = 3.0
        led_data = self._extract(data, ob_data, np.zeros((40, 40), dtype=np.float32))
        assert led_data.ob_sum_color_value == pytest.approx(3.0 * ob_data[self._box].size)

    def test_negative_values_are_kept_in_the_sum(self):
        # this is the whole point: the clipped sum cannot go below the pedestal it carries
        data = np.zeros((40, 40))
        ob_data = np.full((40, 40), -2.0, dtype=np.float32)
        led_data = self._extract(data, ob_data, np.zeros((40, 40), dtype=np.float32))
        assert led_data.ob_sum_color_value == pytest.approx(-2.0 * ob_data[self._box].size)

    def test_the_noise_adds_up_quadratically(self):
        data = np.zeros((40, 40))
        noise = np.zeros((40, 40), dtype=np.float32)
        noise[self._box] = 2.0
        led_data = self._extract(data, np.zeros((40, 40), dtype=np.float32), noise)
        num_pixels = noise[self._box].size
        assert led_data.ob_noise_sum_color_value == pytest.approx(2.0 * np.sqrt(num_pixels))

    def test_pixels_outside_the_channel_carry_no_noise(self):
        # the reader zeroes both arrays outside the channel, so a Bayer image sums only its own pixels
        data = np.zeros((40, 40))
        noise = np.zeros((40, 40), dtype=np.float32)
        noise[::2, ::2] = 2.0
        led_data = self._extract(data, np.zeros((40, 40), dtype=np.float32), noise)
        num_channel_pixels = np.count_nonzero(noise[self._box])
        assert led_data.ob_noise_sum_color_value == pytest.approx(2.0 * np.sqrt(num_channel_pixels))
