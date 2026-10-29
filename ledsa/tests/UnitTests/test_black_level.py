from types import SimpleNamespace

import numpy as np
import pytest

import ledsa.core.image_reading as image_reading
from ledsa.core.image_reading import (MIN_OPTICAL_BLACK_WIDTH, UPRIGHT_ORIENTATION, _channel_data,
                                      _optical_black_statistics, _pixel_read_noise,
                                      _read_img_array_and_noise_from_raw_file, _read_raw_image)

# RGGB pattern as in rawpy raw_colors: 0 red, 1 and 3 green, 2 blue
BAYER_TILE = np.array([[0, 1], [3, 2]])
# black level of every Bayer color in the optical black margin, far from the nominal values below
OPTICAL_BLACK_LEVELS = [100.0, 101.0, 102.0, 103.0]
# different per color, so that a test can tell a per-color fallback from a per-channel one
METADATA_BLACK_LEVELS = [2040, 2041, 2042, 2043]
WHITE_LEVEL = 16383


def make_raw(left_margin: int, noise: float = 0.0, visible: float = None) -> SimpleNamespace:
    """
    The part of a rawpy RawPy object that LEDSA reads: 4 x 4 visible pixels behind a top and a left margin,
    with a known black level per Bayer color in the margin. With noise the two halves of the margin columns
    are offset by -noise and +noise, so the standard deviation of every color is noise. With visible the
    visible pixels are set to that value instead of the black level of their color.
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
    return SimpleNamespace(
        sizes=SimpleNamespace(left_margin=left_margin, top_margin=top_margin, height=height, width=width,
                              flip=0),
        raw_image=raw_image,
        raw_image_visible=raw_image[top_margin:, left_margin:],
        raw_colors=colors,
        raw_colors_visible=colors[top_margin:, left_margin:],
        black_level_per_channel=METADATA_BLACK_LEVELS,
        white_level=WHITE_LEVEL,
    )


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
        return raw
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


class TestBlackLevelsOfARawImage:
    def test_the_optical_black_is_preferred_over_the_metadata(self, fake_imread):
        # the metadata hold nominal values of the camera model, the optical black pixels are read out with
        # the image itself and follow the drift with the sensor temperature
        fake_imread(make_raw(MIN_OPTICAL_BLACK_WIDTH, noise=2.0))
        raw = _read_raw_image('img.CR3')
        assert raw.black_levels == pytest.approx(OPTICAL_BLACK_LEVELS)
        assert raw.read_noise == pytest.approx([2.0] * len(OPTICAL_BLACK_LEVELS))

    def test_the_metadata_are_the_fallback_and_are_per_color_too(self, fake_imread):
        fake_imread(make_raw(MIN_OPTICAL_BLACK_WIDTH - 1))
        raw = _read_raw_image('img.CR3')
        assert raw.black_levels == pytest.approx([float(level) for level in METADATA_BLACK_LEVELS])
        assert raw.read_noise is None

    def test_every_color_is_scaled_by_its_own_black_level(self, fake_imread):
        fake_imread(make_raw(MIN_OPTICAL_BLACK_WIDTH))
        raw = _read_raw_image('img.CR3')
        assert raw.scales == pytest.approx([WHITE_LEVEL / (WHITE_LEVEL - level)
                                            for level in OPTICAL_BLACK_LEVELS])


class TestChannelDataWithoutClipping:
    def test_values_below_the_black_level_stay_negative(self, fake_imread):
        # this is the whole point: clipping them at zero leaves a positive offset in every LED sum
        raw = fake_imread(make_raw(MIN_OPTICAL_BLACK_WIDTH))
        raw.raw_image[raw.sizes.top_margin:, MIN_OPTICAL_BLACK_WIDTH:] -= 2.0
        data = _channel_data(_read_raw_image('img.CR3'))
        scales = np.array([WHITE_LEVEL / (WHITE_LEVEL - level) for level in OPTICAL_BLACK_LEVELS])
        colors = raw.raw_colors_visible
        assert data == pytest.approx(-2.0 * scales[colors], rel=1e-5)

    def test_every_bayer_color_gets_its_own_black_level(self, fake_imread):
        # the visible pixels sit exactly at the black level of their own color, so all of them are 0
        fake_imread(make_raw(MIN_OPTICAL_BLACK_WIDTH))
        assert _channel_data(_read_raw_image('img.CR3')) == pytest.approx(np.zeros((4, 4)))

    def test_the_data_are_float_because_the_black_level_is_a_fraction_of_a_count(self, fake_imread):
        raw = fake_imread(make_raw(MIN_OPTICAL_BLACK_WIDTH))
        raw.raw_image[:, :MIN_OPTICAL_BLACK_WIDTH] += 0.5  # half a count, which an int would lose
        data = _channel_data(_read_raw_image('img.CR3'))
        assert data.dtype == np.float32
        assert (data < 0).all() and (data > -1.0).all()


class TestPixelReadNoise:
    def test_a_pixel_carries_the_noise_of_its_own_color(self, fake_imread):
        fake_imread(make_raw(MIN_OPTICAL_BLACK_WIDTH, noise=4.0))
        raw = _read_raw_image('img.CR3')
        scales = np.array([WHITE_LEVEL / (WHITE_LEVEL - level) for level in OPTICAL_BLACK_LEVELS])
        assert _pixel_read_noise(raw) == pytest.approx(4.0 * scales[raw.filter_array], rel=1e-5)

    def test_there_is_no_noise_without_an_optical_black_margin(self, fake_imread):
        fake_imread(make_raw(MIN_OPTICAL_BLACK_WIDTH - 1))
        assert _pixel_read_noise(_read_raw_image('img.CR3')) is None


class TestTheReaderStepThreeUses:
    def test_the_orientation_reaches_the_data_the_filter_and_the_noise(self, fake_imread):
        fake_imread(make_raw(MIN_OPTICAL_BLACK_WIDTH, noise=4.0, visible=5000.0))
        upright = _read_img_array_and_noise_from_raw_file('img.CR3', 0, UPRIGHT_ORIENTATION)
        turned = _read_img_array_and_noise_from_raw_file('img.CR3', 0, 6)
        for upright_array, turned_array in zip(upright, turned):
            np.testing.assert_array_equal(turned_array, np.rot90(upright_array, -1))

    def test_without_a_margin_only_the_noise_is_missing(self, fake_imread):
        fake_imread(make_raw(MIN_OPTICAL_BLACK_WIDTH - 1, visible=5000.0))
        data, filter_array, noise = _read_img_array_and_noise_from_raw_file('img.CR3', 0)
        assert noise is None
        assert data.shape == filter_array.shape and np.isfinite(data).all()


class TestSumsOfAnLed:
    SEARCH_AREA_RADIUS = 6
    CENTER = (20, 20)

    def _extract(self, data, pixel_noise=None):
        from ledsa.data_extraction.step_3_functions import _generate_led_analysis_data
        search_areas = np.array([[0, self.CENTER[0], self.CENTER[1]]])
        return _generate_led_analysis_data(None, 0, data, False, 0, 'img', 0, search_areas,
                                           self.SEARCH_AREA_RADIUS, fit_leds=False,
                                           pixel_noise=pixel_noise)

    @property
    def _box(self):
        radius = self.SEARCH_AREA_RADIUS
        return np.index_exp[self.CENTER[0] - radius:self.CENTER[0] + radius,
                            self.CENTER[1] - radius:self.CENTER[1] + radius]

    def test_a_sum_below_the_black_level_stays_negative(self):
        # the clipped version could not go below the pedestal it carried
        data = np.full((40, 40), -2.0)
        led_data = self._extract(data)
        assert led_data.sum_color_value == pytest.approx(-2.0 * data[self._box].size)

    def test_without_an_optical_black_margin_the_noise_stays_nan(self):
        assert np.isnan(self._extract(np.full((40, 40), 5.0)).noise_sum_color_value)

    def test_the_noise_of_the_sum_adds_up_quadratically(self):
        noise = np.zeros((40, 40), dtype=np.float32)
        noise[self._box] = 2.0
        led_data = self._extract(np.zeros((40, 40)), noise)
        assert led_data.noise_sum_color_value == pytest.approx(2.0 * np.sqrt(noise[self._box].size))

    def test_pixels_outside_the_channel_carry_no_noise(self):
        # the reader zeroes the noise outside the channel, so a Bayer image sums only its own pixels
        noise = np.zeros((40, 40), dtype=np.float32)
        noise[::2, ::2] = 2.0
        led_data = self._extract(np.zeros((40, 40)), noise)
        assert led_data.noise_sum_color_value == pytest.approx(
            2.0 * np.sqrt(np.count_nonzero(noise[self._box])))
