import numpy as np
import pytest

from ledsa.data_extraction.step_3_functions import (_estimate_local_background, _estimate_local_background_mean,
                                                    _generate_led_analysis_data)

SEARCH_AREA_RADIUS = 8
CENTER = (40, 40)


def make_image(transmission=1.0, pedestal=0.0, noise_sigma=0.0, clip=False, bayer_mask=False, seed=0,
               amplitude=4000.0, sigma=1.5):
    """Single Gaussian LED plus constant pedestal and noise, optionally clipped at 0 and with an RGGB green mask."""
    rng = np.random.default_rng(seed)
    rows, cols = np.mgrid[0:80, 0:80]
    led = amplitude * transmission * np.exp(-((rows - CENTER[0]) ** 2 + (cols - CENTER[1]) ** 2) / (2 * sigma ** 2))
    img = led + pedestal + rng.normal(0.0, noise_sigma, led.shape)
    if clip:
        img = np.clip(img, 0.0, None)
    mask = (rows + cols) % 2 == 1 if bayer_mask else np.ones(img.shape, dtype=bool)
    return np.where(mask, img, 0.0), mask


def analyse(img, mask):
    search_areas = np.array([[0, CENTER[0], CENTER[1]]])
    return _generate_led_analysis_data(None, 1, img, False, 0, 'test_img', 0, search_areas, SEARCH_AREA_RADIUS,
                                       fit_leds=False, channel_mask=mask)


def box(array):
    r = SEARCH_AREA_RADIUS
    return array[CENTER[0] - r:CENTER[0] + r, CENTER[1] - r:CENTER[1] + r]


class TestRingBackgroundMean:
    def test_constant_image(self):
        assert _estimate_local_background_mean(np.full((16, 16), 7.5)) == pytest.approx(7.5)

    def test_foreign_channel_pixels_are_excluded_by_mask(self):
        img, mask = make_image(pedestal=7.5, bayer_mask=True, amplitude=0.0)
        assert _estimate_local_background_mean(box(img), box(mask)) == pytest.approx(7.5)

    def test_channel_pixels_with_value_zero_are_included(self):
        data = np.full((16, 16), 4.0)
        data[::2, :] = 0.0  # channel pixels clipped at 0
        ring = np.ones((16, 16), dtype=bool)
        ring[2:-2, 2:-2] = False
        assert _estimate_local_background_mean(data) == pytest.approx(data[ring].mean())
        assert _estimate_local_background(data) == pytest.approx(4.0)  # old estimator ignores the zeros

    def test_too_small_or_empty_gives_nan(self):
        assert np.isnan(_estimate_local_background_mean(np.ones((4, 4))))
        assert np.isnan(_estimate_local_background_mean(np.ones((16, 16)), np.zeros((16, 16), dtype=bool)))


class TestBgsub2Sum:
    @pytest.mark.parametrize('bayer_mask', [False, True])
    def test_subtraction_is_linear(self, bayer_mask):
        img, mask = make_image(pedestal=3.0, noise_sigma=2.0, clip=True, bayer_mask=bayer_mask)
        led_data = analyse(img, mask)
        num_channel_pixels = np.count_nonzero(box(mask))
        expected = box(img).sum() - num_channel_pixels * _estimate_local_background_mean(box(img), box(mask))
        assert led_data.bgsub2_sum_color_value == pytest.approx(expected)

    def test_floor_limits_the_corrected_sum(self):
        # LED shifted to the corner of the box: its light falls into the ring and N * b exceeds the box sum
        img, mask = make_image(amplitude=0.0)
        row, col = CENTER[0] - SEARCH_AREA_RADIUS + 1, CENTER[1] - SEARCH_AREA_RADIUS + 1
        img[row - 1:row + 2, col - 1:col + 2] = 1000.0
        led_data = analyse(img, mask)
        num_channel_pixels = np.count_nonzero(box(mask))
        assert box(img).sum() - num_channel_pixels * _estimate_local_background_mean(box(img), box(mask)) < 0
        assert led_data.bgsub2_sum_color_value == pytest.approx(0.05 * box(img).sum())

    def test_existing_values_are_unchanged(self):
        img, mask = make_image(pedestal=3.0, noise_sigma=2.0, clip=True, bayer_mask=True)
        led_data = analyse(img, mask)
        reference = _generate_led_analysis_data(None, 1, img, False, 0, 'test_img', 0, np.array([[0, *CENTER]]),
                                                SEARCH_AREA_RADIUS, fit_leds=False)
        assert led_data.sum_color_value == reference.sum_color_value
        assert led_data.bgsub_sum_color_value == reference.bgsub_sum_color_value
        fields = led_data.get_main_data_string().split(',')
        assert ','.join(fields[:6]) == ','.join(reference.get_main_data_string().split(',')[:6])
        assert float(fields[8]) == pytest.approx(led_data.bgsub2_sum_color_value, rel=1e-4)

    def test_optical_depth_bias_with_clipped_noise(self):
        # LED without halo in the ring: the mean estimator removes the pedestal including the clipped noise,
        # the median of the pixels > 0 with pixel-wise clipping (bgsub) leaves a bias
        transmission = 0.01
        args = dict(pedestal=2.0, noise_sigma=2.0, clip=True, bayer_mask=True)
        true_depth = -np.log(transmission)
        depth = {'sum': [], 'bgsub': [], 'bgsub2': []}
        for seed in range(20):
            ref = analyse(*make_image(transmission=1.0, seed=2 * seed, **args))
            img = analyse(*make_image(transmission=transmission, seed=2 * seed + 1, **args))
            depth['sum'].append(-np.log(img.sum_color_value / ref.sum_color_value))
            depth['bgsub'].append(-np.log(img.bgsub_sum_color_value / ref.bgsub_sum_color_value))
            depth['bgsub2'].append(-np.log(img.bgsub2_sum_color_value / ref.bgsub2_sum_color_value))
        bias = {key: np.mean(values) / true_depth - 1 for key, values in depth.items()}
        assert abs(bias['bgsub2']) < 0.02
        assert bias['bgsub'] < -0.02
        assert bias['sum'] < -0.1
