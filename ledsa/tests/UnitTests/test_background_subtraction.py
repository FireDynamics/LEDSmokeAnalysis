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
        assert len(main_data_fields) >= 6  # further background columns are appended after bgsub
        assert float(main_data_fields[5]) == pytest.approx(led_data.bgsub_sum_color_value, rel=1e-4)


class TestColumnNames:
    def _write_led_positions_csv(self, tmp_path, num_columns, header='# header'):
        channel_dir = tmp_path / 'analysis' / 'channel0'
        channel_dir.mkdir(parents=True)
        row = ','.join(str(float(i)) for i in range(num_columns))
        (channel_dir / '1_led_positions.csv').write_text(f'{header}\n{row}\n{row}\n', encoding='utf-8')

    def test_csv_written_by_step_3_without_fit(self, tmp_path, monkeypatch):
        from ledsa.core.file_handling import _get_column_names
        from ledsa.data_extraction.step_3_functions import _create_header
        img_infos = np.array([['1', 'img.jpg', '12:00:00', '0']])
        header = _create_header(0, '1', 'img.jpg', img_infos, 'root', False).strip()
        num_columns = len(header.split('\n')[1].split(','))
        self._write_led_positions_csv(tmp_path, num_columns, header)
        monkeypatch.chdir(tmp_path)
        columns = _get_column_names(0)
        assert columns[:7] == ["img_id", "led_id", "led_array_id", "sum_col_val", "mean_col_val",
                               "max_col_val", "bgsub_sum_col_val"]
        assert "sidebg_sum_col_val" in columns and "sidebg_pedestal" in columns
        assert columns[-2:] == ["width", "height"]
        assert len(columns) == num_columns + 3

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

    def test_fitted_csv_written_by_step_3(self, tmp_path, monkeypatch):
        # a row as written by step 3 with fits: the columns must parse and match the names
        from types import SimpleNamespace
        from ledsa.core.file_handling import _get_column_names
        from ledsa.data_extraction.LEDAnalysisData import LEDAnalysisData
        led_data = LEDAnalysisData(3, 0, True)
        led_data.sum_color_value, led_data.mean_color_value, led_data.max_color_value = 41210.0, 103.0, 232
        led_data.bgsub_sum_color_value = 40000.0
        led_data.sidebg_sum_color_value, led_data.sidebg_pedestal = 39000.0, 2210.0
        led_data.bgsub2_sum_color_value = 39500.0
        led_data.led_center_x, led_data.led_center_y = 50.521, 60.25
        led_data.fit_results = SimpleNamespace(x=np.arange(1.5, 9.5), success=True, fun=0.5, nfev=120)
        led_data.fit_time = 0.25
        channel_dir = tmp_path / 'analysis' / 'channel0'
        channel_dir.mkdir(parents=True)
        (channel_dir / '1_led_positions.csv').write_text(f'# header\n{led_data}{led_data}', encoding='utf-8')
        monkeypatch.chdir(tmp_path)
        columns = _get_column_names(0)
        values = np.loadtxt(channel_dir / '1_led_positions.csv', delimiter=',')[0]
        assert len(columns) - 2 == len(values) + 1  # width/height appended, img_id prepended on load
        named = dict(zip(columns[1:], values))
        assert named['max_col_val'] == 232
        assert named['bgsub_sum_col_val'] == pytest.approx(40000.0)
        assert named['sidebg_pedestal'] == pytest.approx(2210.0)
        assert named['bgsub2_sum_col_val'] == pytest.approx(39500.0)
        assert named['led_center_x'] == pytest.approx(50.521)
        assert named['A'] == pytest.approx(5.5)
        assert named['fit_nfev'] == 120
        assert named['fit_time'] == pytest.approx(0.25)

    @pytest.mark.parametrize('num_columns, last_main', [(6, 'bgsub_sum_col_val'), (9, 'bgsub2_sum_col_val'),
                                                         (20, 'bgsub_sum_col_val'), (23, 'bgsub2_sum_col_val')])
    def test_layouts(self, tmp_path, monkeypatch, num_columns, last_main):
        from ledsa.core.file_handling import _get_column_names
        self._write_led_positions_csv(tmp_path, num_columns)
        monkeypatch.chdir(tmp_path)
        columns = _get_column_names(0)
        assert len(columns) == num_columns + 1 + 2
        main = columns[:columns.index('led_center_x')] if 'led_center_x' in columns else columns[:-2]
        assert main[-1] == last_main
        if num_columns > 9:
            assert columns[-3:] == ['fit_time', 'width', 'height']

    def test_fitted_csv_without_delimiter_raises_with_hint(self, tmp_path, monkeypatch):
        # fitted files of this branch written before the delimiter fix: bgsub_sum_col_value and led_center_x glued
        from ledsa.core.file_handling import _get_column_names
        fields = ['3', '0', '4.1210e+04', '1.0300e+02', '232', '4.1210e+04', '5.0521e+01'] + ['1.0'] * 13
        row = ','.join(fields[:5]) + ',' + fields[5] + ','.join(fields[6:])
        channel_dir = tmp_path / 'analysis' / 'channel0'
        channel_dir.mkdir(parents=True)
        (channel_dir / '1_led_positions.csv').write_text(f'# header\n{row}\n{row}\n', encoding='utf-8')
        monkeypatch.chdir(tmp_path)
        with pytest.raises(ValueError, match='glued into one field. Run step 3 again'):
            _get_column_names(0)

    def test_legacy_fitted_csv(self, tmp_path, monkeypatch, capsys):
        # fitted files written before the delimiter fix have 18 columns; read as before, with a warning
        from ledsa.core.file_handling import _get_column_names
        self._write_led_positions_csv(tmp_path, 18)
        monkeypatch.chdir(tmp_path)
        columns = _get_column_names(0)
        assert columns[:8] == ["img_id", "led_id", "led_array_id", "sum_col_val", "mean_col_val", "max_col_val",
                               "led_center_x", "led_center_y"]
        assert columns[-3:] == ["fit_nfev", "width", "height"]
        assert len(columns) == 18 + 1 + 2
        assert 'max_col_val and all fit columns are not' in capsys.readouterr().out

    @pytest.mark.parametrize('num_columns, first_names', [
        (15, ["img_id", "led_id", "led_array_id", "x"]), (4, ["img_id", "led_id", "led_array_id", "sum_col_val"])])
    def test_oldest_layouts(self, tmp_path, monkeypatch, num_columns, first_names):
        from ledsa.core.file_handling import _get_column_names
        self._write_led_positions_csv(tmp_path, num_columns)
        monkeypatch.chdir(tmp_path)
        columns = _get_column_names(0)
        assert columns[:4] == first_names
        assert len(columns) == num_columns + 1 + 2

    @pytest.mark.parametrize('num_columns', [7, 19, 21, 22])
    def test_unknown_layout_raises(self, tmp_path, monkeypatch, num_columns):
        from ledsa.core.file_handling import _get_column_names
        self._write_led_positions_csv(tmp_path, num_columns)
        monkeypatch.chdir(tmp_path)
        with pytest.raises(ValueError, match=f'has {num_columns} columns'):
            _get_column_names(0)
