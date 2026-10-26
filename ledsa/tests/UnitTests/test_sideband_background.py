import numpy as np
import pytest

from ledsa.data_extraction.sideband_background import (build_sideband_pixels, estimate_sideband_background,
                                                       estimate_strip_directions, get_sideband_pixels,
                                                       subtract_pedestal)
from ledsa.data_extraction.step_3_functions import _generate_led_analysis_data, get_sideband_parameters

SEARCH_AREA_RADIUS = 8
LED_SPACING = 20
LED_AMPLITUDE = 400.0
LED_SIGMA = 1.5
IMG_SIZE = 240
SIDEBAND_DISTANCES = (25, 60)


def make_strip(angle_deg=0.0, num_leds=7, center=(120, 120)):
    """LED positions (row, col) along a straight strip; angle 0 is a vertical strip (along the rows)."""
    along = np.array([np.cos(np.radians(angle_deg)), np.sin(np.radians(angle_deg))])
    offsets = (np.arange(num_leds) - num_leds // 2) * LED_SPACING
    positions = np.rint(np.array(center) + offsets[:, None] * along).astype(int)
    return np.column_stack([np.arange(num_leds), positions])


def make_image(search_areas, transmission=1.0, pedestal=0.0, noise_sigma=0.0, bayer_mask=False, clip=False,
               seed=0, amplitude=LED_AMPLITUDE):
    """Gaussian LEDs scaled by the transmission plus a constant pedestal and Gaussian noise.

    With clip=True the image is clipped at 0 like LEDSA does after the black level subtraction of raw files.
    With bayer_mask=True only the green pixels of an RGGB pattern belong to the channel, all other pixels are 0.
    Returns the channel array, the channel mask and the pure LED image (without pedestal and noise).
    """
    rng = np.random.default_rng(seed)
    rows, cols = np.mgrid[0:IMG_SIZE, 0:IMG_SIZE]
    led_img = np.zeros((IMG_SIZE, IMG_SIZE))
    for _, row, col in search_areas:
        led_img += amplitude * transmission * np.exp(-((rows - row) ** 2 + (cols - col) ** 2) / (2 * LED_SIGMA ** 2))
    img = led_img + pedestal + rng.normal(0.0, noise_sigma, led_img.shape)
    if clip:
        img = np.clip(img, 0.0, None)
    if bayer_mask:
        mask = (rows + cols) % 2 == 1
    else:
        mask = np.ones(img.shape, dtype=bool)
    img = np.where(mask, img, 0.0)
    led_img = np.where(mask, led_img, 0.0)
    return img, mask, led_img


def box_sum(img, row, col, radius=SEARCH_AREA_RADIUS):
    return float(np.sum(img[row - radius:row + radius, col - radius:col + radius]))


def led_data_for(img, mask, search_areas, led_id, floor=0.05):
    sideband_pixels = get_sideband_pixels(search_areas, [search_areas[:, 0]], img.shape, SEARCH_AREA_RADIUS,
                                          SIDEBAND_DISTANCES)
    return _generate_led_analysis_data(None, 1, img, False, led_id, 'test_img', 0, search_areas, SEARCH_AREA_RADIUS,
                                       fit_leds=False, channel_mask=mask, sideband_pixels=sideband_pixels[led_id],
                                       sideband_floor=floor)


class TestStripDirection:
    @pytest.mark.parametrize('angle', [0.0, 90.0, 30.0, -45.0])
    def test_direction_follows_strip(self, angle):
        search_areas = make_strip(angle)
        directions = estimate_strip_directions(search_areas, [search_areas[:, 0]])
        expected = np.array([np.cos(np.radians(angle)), np.sin(np.radians(angle))])
        for direction in directions:
            # direction is defined up to the sign; rounding of the LED positions tilts it slightly
            assert abs(np.dot(direction, expected)) == pytest.approx(1.0, abs=2e-3)

    def test_led_without_array_has_no_direction(self):
        search_areas = make_strip(0.0)
        directions = estimate_strip_directions(search_areas, [search_areas[:3, 0]])
        assert np.all(np.isfinite(directions[:3]))
        assert np.all(np.isnan(directions[3:]))


class TestSidebandPixels:
    def test_vertical_strip_matches_axis_aligned_bands(self):
        # same pixels as the axis-aligned bands of the H2 evaluation: rows r-R..r+R-1, columns c+25..c+59 and
        # c-60..c-26
        search_areas = make_strip(0.0)
        directions = np.tile([1.0, 0.0], (len(search_areas), 1))
        pixels = build_sideband_pixels(search_areas, directions, (IMG_SIZE, IMG_SIZE), SEARCH_AREA_RADIUS,
                                       SIDEBAND_DISTANCES)
        led_id, row, col = search_areas[3]
        expected = np.zeros((IMG_SIZE, IMG_SIZE), dtype=bool)
        expected[row - SEARCH_AREA_RADIUS:row + SEARCH_AREA_RADIUS, col + 25:col + 60] = True
        expected[row - SEARCH_AREA_RADIUS:row + SEARCH_AREA_RADIUS, col - 60:col - 25] = True
        selected = np.zeros_like(expected)
        selected[pixels[led_id]] = True
        np.testing.assert_array_equal(selected, expected)

    @pytest.mark.parametrize('angle', [0.0, 90.0, 30.0])
    def test_bands_do_not_touch_search_areas(self, angle):
        search_areas = make_strip(angle)
        pixels = get_sideband_pixels(search_areas, [search_areas[:, 0]], (IMG_SIZE, IMG_SIZE), SEARCH_AREA_RADIUS,
                                     SIDEBAND_DISTANCES)
        for _, row, col in search_areas:
            box = np.zeros((IMG_SIZE, IMG_SIZE), dtype=bool)
            box[row - SEARCH_AREA_RADIUS:row + SEARCH_AREA_RADIUS, col - SEARCH_AREA_RADIUS:col + SEARCH_AREA_RADIUS] = True
            for led_pixels in pixels:
                assert led_pixels[0].size > 0
                assert not box[led_pixels].any()

    def test_neighbouring_strip_is_excluded(self):
        strip = make_strip(0.0, center=(120, 80))
        neighbour = make_strip(0.0, center=(120, 120))  # 40 px to the right, inside the right side band
        neighbour[:, 0] += len(strip)
        search_areas = np.vstack([strip, neighbour])
        exclusion = 2 * SEARCH_AREA_RADIUS
        pixels = get_sideband_pixels(search_areas, [strip[:, 0], neighbour[:, 0]], (IMG_SIZE, IMG_SIZE),
                                     SEARCH_AREA_RADIUS, SIDEBAND_DISTANCES)
        rows, cols = pixels[3]
        assert np.all((cols < 120 - exclusion) | (cols >= 120 + exclusion))
        assert np.any(cols >= 120 + exclusion)

    def test_bands_are_clipped_at_the_image_border(self):
        search_areas = make_strip(0.0, center=(120, 30))
        pixels = get_sideband_pixels(search_areas, [search_areas[:, 0]], (IMG_SIZE, IMG_SIZE), SEARCH_AREA_RADIUS,
                                     SIDEBAND_DISTANCES)
        rows, cols = pixels[3]
        assert cols.min() == 0
        assert cols.size > 0


class TestSidebandBackground:
    @pytest.mark.parametrize('bayer_mask', [False, True])
    def test_constant_pedestal_is_recovered(self, bayer_mask):
        search_areas = make_strip(30.0)
        img, mask, _ = make_image(search_areas, pedestal=7.0, bayer_mask=bayer_mask)
        pixels = get_sideband_pixels(search_areas, [search_areas[:, 0]], img.shape, SEARCH_AREA_RADIUS,
                                     SIDEBAND_DISTANCES)
        for led_pixels in pixels:
            assert estimate_sideband_background(img, mask, led_pixels) == pytest.approx(7.0)

    def test_clipped_zeros_are_included(self):
        # pedestal of 1 count with noise of 2 counts: clipping at 0 raises the mean in the box and in the side
        # bands alike, so the side band mean (including zeros) is the matching estimator; the mean of the
        # pixels > 0 is too high
        from scipy.stats import norm
        search_areas = make_strip(0.0)
        img, mask, _ = make_image(search_areas, pedestal=1.0, noise_sigma=2.0, clip=True, bayer_mask=True)
        clipped_mean = 1.0 * norm.cdf(0.5) + 2.0 * norm.pdf(0.5)  # expectation of max(0, N(1, 2))
        pixels = get_sideband_pixels(search_areas, [search_areas[:, 0]], img.shape, SEARCH_AREA_RADIUS,
                                     SIDEBAND_DISTANCES)
        estimate = estimate_sideband_background(img, mask, pixels[3])
        positive_mean = np.mean(img[pixels[3]][img[pixels[3]] > 0])
        assert estimate == pytest.approx(clipped_mean, rel=0.1)
        assert positive_mean > 1.3 * estimate

    def test_no_channel_pixels_gives_nan(self):
        img = np.zeros((10, 10))
        assert np.isnan(estimate_sideband_background(img, None, None))
        assert np.isnan(estimate_sideband_background(img, None, (np.array([], dtype=int), np.array([], dtype=int))))
        mask = np.zeros((10, 10), dtype=bool)
        assert np.isnan(estimate_sideband_background(img, mask, (np.array([1]), np.array([1]))))


class TestSidebandSum:
    @pytest.mark.parametrize('bayer_mask', [False, True])
    @pytest.mark.parametrize('angle', [0.0, 90.0, 30.0])
    def test_sidebg_sum_is_independent_of_pedestal(self, bayer_mask, angle):
        search_areas = make_strip(angle)
        for pedestal in [0.0, 3.0, 20.0]:
            img, mask, led_img = make_image(search_areas, pedestal=pedestal, bayer_mask=bayer_mask)
            for led_id, row, col in search_areas:
                led_data = led_data_for(img, mask, search_areas, led_id)
                assert led_data.sidebg_sum_color_value == pytest.approx(box_sum(led_img, row, col), rel=1e-6)
                num_channel_pixels = (2 * SEARCH_AREA_RADIUS) ** 2 // (2 if bayer_mask else 1)
                assert led_data.sidebg_pedestal == pytest.approx(num_channel_pixels * pedestal, abs=1e-9)

    def test_optical_depth_bias_with_clipped_noise(self):
        # optical depth -ln(I/I0) of a dimmed LED with pedestal, clipped noise and Bayer mask: the side band
        # correction removes the bias of the plain box sum
        search_areas = make_strip(0.0)
        transmission = 0.01
        image_args = dict(pedestal=2.0, noise_sigma=2.0, clip=True, bayer_mask=True, amplitude=4000.0)
        clean, mask, _ = make_image(search_areas, transmission=1.0, seed=1, **image_args)
        smoke, _, _ = make_image(search_areas, transmission=transmission, seed=2, **image_args)
        true_depth = -np.log(transmission)
        depth_sum, depth_sidebg = [], []
        for led_id in search_areas[:, 0]:
            ref = led_data_for(clean, mask, search_areas, led_id)
            img = led_data_for(smoke, mask, search_areas, led_id)
            depth_sum.append(-np.log(img.sum_color_value / ref.sum_color_value))
            depth_sidebg.append(-np.log(img.sidebg_sum_color_value / ref.sidebg_sum_color_value))
        assert abs(np.mean(depth_sidebg) - true_depth) / true_depth < 0.02
        assert (true_depth - np.mean(depth_sum)) / true_depth > 0.1

    def test_floor_limits_the_corrected_sum(self):
        assert subtract_pedestal(100.0, 99.0, 0.05) == pytest.approx(5.0)
        assert subtract_pedestal(100.0, 20.0, 0.05) == pytest.approx(80.0)
        assert np.isnan(subtract_pedestal(100.0, np.nan, 0.05))

    def test_old_values_are_unchanged(self):
        # the existing columns keep their values; the new ones are appended to the CSV line
        search_areas = make_strip(0.0)
        img, mask, _ = make_image(search_areas, pedestal=5.0, noise_sigma=1.0, bayer_mask=True)
        led_data = led_data_for(img, mask, search_areas, 3)
        reference = _generate_led_analysis_data(None, 1, img, False, 3, 'test_img', 0, search_areas,
                                                SEARCH_AREA_RADIUS, fit_leds=False)
        assert led_data.sum_color_value == reference.sum_color_value
        assert led_data.mean_color_value == reference.mean_color_value
        assert led_data.max_color_value == reference.max_color_value
        assert led_data.bgsub_sum_color_value == reference.bgsub_sum_color_value
        fields = led_data.get_main_data_string().split(',')
        assert ','.join(fields[:6]) == ','.join(reference.get_main_data_string().split(',')[:6])
        assert float(fields[6]) == pytest.approx(led_data.sidebg_sum_color_value, rel=1e-4)
        assert float(fields[7]) == pytest.approx(led_data.sidebg_pedestal, rel=1e-4)


class TestSidebandParameters:
    def test_defaults_without_config(self):
        assert get_sideband_parameters(None) == ((25, 60), 0.05, None)

    def test_defaults_for_old_config(self, tmp_path, monkeypatch):
        from ledsa.core.ConfigData import ConfigData
        monkeypatch.chdir(tmp_path)
        (tmp_path / 'config.ini').write_text('[DEFAULT]\nimg_directory = x\n[analyse_photo]\nnum_skip_leds = 0\n', encoding='utf-8')
        assert get_sideband_parameters(ConfigData()) == ((25, 60), 0.05, None)

    def test_values_from_config(self, tmp_path, monkeypatch):
        from ledsa.core.ConfigData import ConfigData
        monkeypatch.chdir(tmp_path)
        ConfigData(load_config_file=False, sideband_distances='12 20', sideband_floor=0.1,
                   sideband_exclusion_margin=3)
        assert get_sideband_parameters(ConfigData()) == ((12.0, 20.0), 0.1, 3)
