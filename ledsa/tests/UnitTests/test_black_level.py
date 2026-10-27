from types import SimpleNamespace

import numpy as np
import pytest

from ledsa.core.image_reading import MIN_OPTICAL_BLACK_WIDTH, _black_level_and_noise, _optical_black_statistics

# RGGB pattern as in rawpy raw_colors: 0 red, 1 and 3 green, 2 blue
BAYER_TILE = np.array([[0, 1], [3, 2]])
# black level of every Bayer color in the optical black margin, far from the nominal value below
OPTICAL_BLACK_LEVELS = [100.0, 101.0, 102.0, 103.0]
METADATA_BLACK_LEVELS = [2047, 2047, 2047, 2047]


def make_raw(left_margin: int, noise: float = 0.0) -> SimpleNamespace:
    """
    A raw image of 4 x 4 visible pixels behind a top and a left margin, with a known black level per Bayer color
    in the margin. With noise the two halves of the margin rows are offset by +-noise, so the standard deviation
    of every color is noise.
    """
    top_margin, height, width = 2, 4, 4
    shape = (top_margin + height, left_margin + width)
    colors = np.tile(BAYER_TILE, (shape[0] // 2 + 1, shape[1] // 2 + 1))[:shape[0], :shape[1]]
    raw_image = np.zeros(shape, dtype=np.float64)
    for color, level in enumerate(OPTICAL_BLACK_LEVELS):
        raw_image[colors == color] = level
    if noise and left_margin:
        offsets = np.where(np.arange(left_margin) < left_margin // 2, -noise, noise)
        raw_image[:, :left_margin] += offsets
    return SimpleNamespace(
        sizes=SimpleNamespace(left_margin=left_margin, top_margin=top_margin, height=height, width=width),
        raw_image=raw_image,
        raw_colors=colors,
        raw_colors_visible=colors[top_margin:, left_margin:],
        black_level_per_channel=METADATA_BLACK_LEVELS,
        white_level=16383,
    )


class TestOpticalBlackStatistics:
    def test_mean_and_std_per_bayer_color(self):
        mean, std = _optical_black_statistics(make_raw(MIN_OPTICAL_BLACK_WIDTH, noise=3.0))
        assert mean == pytest.approx(OPTICAL_BLACK_LEVELS)
        assert std == pytest.approx([3.0] * len(OPTICAL_BLACK_LEVELS))

    def test_margin_narrower_than_the_minimum_is_not_used(self):
        mean, std = _optical_black_statistics(make_raw(MIN_OPTICAL_BLACK_WIDTH - 2))
        assert mean is None and std is None

    def test_sensor_without_a_margin(self):
        mean, std = _optical_black_statistics(make_raw(0))
        assert mean is None and std is None


class TestBlackLevelAndNoise:
    @pytest.mark.parametrize('channel', [0, 1, 2])
    def test_optical_black_is_preferred_over_the_metadata(self, channel):
        # the black level of the metadata is a nominal value of the camera model, the optical black pixels are read
        # out with the image itself
        black_level, mean, std = _black_level_and_noise(make_raw(MIN_OPTICAL_BLACK_WIDTH, noise=1.5), channel)
        assert black_level == pytest.approx(OPTICAL_BLACK_LEVELS[channel])
        assert mean == pytest.approx(OPTICAL_BLACK_LEVELS)
        assert std == pytest.approx([1.5] * len(OPTICAL_BLACK_LEVELS))

    @pytest.mark.parametrize('channel', [0, 1, 2])
    def test_metadata_is_the_fallback_without_an_optical_black_margin(self, channel):
        black_level, mean, std = _black_level_and_noise(make_raw(MIN_OPTICAL_BLACK_WIDTH - 1), channel)
        assert black_level == pytest.approx(float(METADATA_BLACK_LEVELS[channel]))
        assert mean is None and std is None

    def test_the_black_level_is_a_float_in_both_cases(self):
        # it is subtracted from the image and enters the scale factor, where an int would round the offset away
        assert isinstance(_black_level_and_noise(make_raw(MIN_OPTICAL_BLACK_WIDTH), 0)[0], float)
        assert isinstance(_black_level_and_noise(make_raw(0), 0)[0], float)
