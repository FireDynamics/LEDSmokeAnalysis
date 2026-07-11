import numpy as np
import pytest

from ledsa.data_extraction.step_3_functions import _estimate_local_background, _generate_led_analysis_data

SEARCH_AREA_RADIUS = 10
LED_CENTER = (30, 30)
LED_AMPLITUDE = 200.0
LED_SIGMA = 2.0


def make_synthetic_image(background=0.0, bayer_mask=False):
    """Create a 60x60 image with a Gaussian LED spot at LED_CENTER plus a constant background.

    With bayer_mask=True, three out of four pixels are set to 0 like in the Bayer
    array of a raw image where all pixels of foreign color channels are masked with 0.
    """
    x, y = np.meshgrid(np.arange(60), np.arange(60), indexing='ij')
    dist_sq = (x - LED_CENTER[0]) ** 2 + (y - LED_CENTER[1]) ** 2
    led = LED_AMPLITUDE * np.exp(-dist_sq / (2 * LED_SIGMA ** 2))
    img = led + background
    if bayer_mask:
        mask = (x % 2 == 0) & (y % 2 == 0)
        img = np.where(mask, img, 0.0)
    return img


def pure_led_integral(img_with_zero_background):
    radius = SEARCH_AREA_RADIUS
    return np.sum(img_with_zero_background[LED_CENTER[0] - radius:LED_CENTER[0] + radius,
                                           LED_CENTER[1] - radius:LED_CENTER[1] + radius])


def extract_led_data(img):
    search_areas = np.array([[0, LED_CENTER[0], LED_CENTER[1]]])
    return _generate_led_analysis_data(None, 0, img, False, 0, 'test_img', 0, search_areas,
                                       SEARCH_AREA_RADIUS, fit_leds=False)


class TestEstimateLocalBackground:
    def test_constant_image_returns_background(self):
        img = np.full((20, 20), 7.5)
        assert _estimate_local_background(img) == pytest.approx(7.5)

    def test_masked_zero_pixels_are_ignored(self):
        img = np.full((20, 20), 7.5)
        img[::2, :] = 0.0
        img[:, ::2] = 0.0
        assert _estimate_local_background(img) == pytest.approx(7.5)

    def test_all_zero_border_returns_zero(self):
        img = np.zeros((20, 20))
        img[10, 10] = 100.0
        assert _estimate_local_background(img) == 0.0

    def test_too_small_search_area_returns_zero(self):
        img = np.full((4, 4), 7.5)
        assert _estimate_local_background(img) == 0.0


class TestBackgroundSubtractedSum:
    def test_bgsub_sum_is_independent_of_background(self):
        led_integral = pure_led_integral(make_synthetic_image(background=0.0))
        for background in [0.0, 5.0, 20.0, 50.0]:
            led_data = extract_led_data(make_synthetic_image(background=background))
            assert led_data.bgsub_sum_color_value == pytest.approx(led_integral, rel=1e-3), \
                f'bgsub_sum_color_value deviates from the pure LED integral for background {background}'

    def test_sum_col_value_scales_with_background(self):
        led_integral = pure_led_integral(make_synthetic_image(background=0.0))
        num_pixels = (2 * SEARCH_AREA_RADIUS) ** 2
        for background in [5.0, 20.0]:
            led_data = extract_led_data(make_synthetic_image(background=background))
            assert led_data.sum_color_value == pytest.approx(led_integral + background * num_pixels, rel=1e-6)

    def test_bgsub_sum_with_bayer_masked_pixels(self):
        led_integral = pure_led_integral(make_synthetic_image(background=0.0, bayer_mask=True))
        for background in [5.0, 20.0]:
            led_data = extract_led_data(make_synthetic_image(background=background, bayer_mask=True))
            assert led_data.bgsub_sum_color_value == pytest.approx(led_integral, rel=1e-3)

    def test_sum_col_value_is_unchanged_by_new_quantity(self):
        img = make_synthetic_image(background=10.0)
        led_data = extract_led_data(img)
        radius = SEARCH_AREA_RADIUS
        expected = np.sum(img[LED_CENTER[0] - radius:LED_CENTER[0] + radius,
                              LED_CENTER[1] - radius:LED_CENTER[1] + radius])
        assert led_data.sum_color_value == pytest.approx(expected)

    def test_csv_serialization_contains_bgsub_value(self):
        led_data = extract_led_data(make_synthetic_image(background=10.0))
        main_data_fields = led_data.get_main_data_string().split(',')
        assert len(main_data_fields) == 6
        assert float(main_data_fields[5]) == pytest.approx(led_data.bgsub_sum_color_value, rel=1e-4)


class TestColumnNames:
    def _write_led_positions_csv(self, tmp_path, num_columns):
        channel_dir = tmp_path / 'analysis' / 'channel0'
        channel_dir.mkdir(parents=True)
        row = ','.join(str(float(i)) for i in range(num_columns))
        (channel_dir / '1_led_positions.csv').write_text(f'# header\n{row}\n{row}\n')

    def test_new_csv_with_bgsub_column(self, tmp_path, monkeypatch):
        from ledsa.core.file_handling import _get_column_names
        self._write_led_positions_csv(tmp_path, 6)
        monkeypatch.chdir(tmp_path)
        columns = _get_column_names(0)
        assert columns == ["img_id", "led_id", "led_array_id", "sum_col_val", "mean_col_val",
                           "max_col_val", "bgsub_sum_col_val", "width", "height"]

    def test_old_csv_without_bgsub_column(self, tmp_path, monkeypatch):
        from ledsa.core.file_handling import _get_column_names
        self._write_led_positions_csv(tmp_path, 5)
        monkeypatch.chdir(tmp_path)
        columns = _get_column_names(0)
        assert columns == ["img_id", "led_id", "led_array_id", "sum_col_val", "mean_col_val",
                           "max_col_val", "width", "height"]
