"""Unit tests for the pedestal of the search areas from the far field of the LED strips (farbg_sum_col_val).

The pedestal is the level of the color channel in two bands parallel to the strip, far from every LED, smoothed
along the strip, and of the two sides the lower one. Step 3 subtracts N times it from the sum of the search area
when farfield_distances of config.ini is set, and writes the result files as before when it is not.
"""
import configparser

import numpy as np
import pytest

from ledsa.core import image_reading
from ledsa.core.ConfigData import ConfigData
from ledsa.core.file_handling import (BASE_COLUMNS, COLUMN_LAYOUTS, FARBG_COLUMNS, FIT_COLUMNS,
                                      LED_CENTER_COLUMNS, _get_column_names)
from ledsa.core.image_reading import ImageChannels, RawImage
from ledsa.data_extraction import farfield_background as farfield
from ledsa.data_extraction.step_3_functions import _create_header, generate_analysis_data

RADIUS = 4
DISTANCES = (10.0, 20.0)
IMG_SHAPE = (120, 200)
# a vertical strip of ten LEDs at column 100
VERTICAL = np.array([[i, 15 + 10 * i, 100] for i in range(10)])


def band_set(pixels):
    return set(zip(pixels[0].tolist(), pixels[1].tolist()))


class TestDistances:
    @pytest.mark.parametrize('value', [None, 'None', 'none', '', '  '])
    def test_none_estimates_no_pedestal(self, value):
        assert farfield.parse_farfield_distances(value, RADIUS) is None

    def test_two_distances(self):
        assert farfield.parse_farfield_distances('150 300', 9) == (150.0, 300.0)

    @pytest.mark.parametrize('value', ['150', '300 150', '150 150', 'far away', '150 300 400', '5 300'])
    def test_anything_else_names_the_option(self, value):
        # the last one puts the band into the search area
        with pytest.raises(ValueError, match='farfield_distances of config.ini'):
            farfield.parse_farfield_distances(value, 9)


class TestStripDirections:
    def test_vertical_horizontal_and_oblique_strips(self):
        horizontal = np.array([[i, 50, 20 + 10 * i] for i in range(6)])
        oblique = np.array([[i, 20 + 7 * i, 20 + 7 * i] for i in range(6)])
        for strip, expected in ((VERTICAL, (1.0, 0.0)), (horizontal, (0.0, 1.0)),
                                (oblique, (np.sqrt(0.5), np.sqrt(0.5)))):
            directions = farfield.estimate_strip_directions(strip, [strip[:, 0]])
            np.testing.assert_allclose(directions, np.tile(expected, (len(strip), 1)), atol=1e-12)

    def test_an_led_alone_in_its_array_has_no_direction(self):
        search_areas = np.vstack([VERTICAL, [[10, 60, 30]]])
        directions = farfield.estimate_strip_directions(search_areas, [VERTICAL[:, 0], [10]])
        assert np.isnan(directions[10]).all()
        assert np.isfinite(directions[:10]).all()


class TestBandPixels:
    def test_bands_of_a_vertical_strip(self):
        geometry = farfield.build_farfield_geometry(VERTICAL, [VERTICAL[:, 0]], IMG_SHAPE, RADIUS, DISTANCES)
        row, col = VERTICAL[3, 1:]
        rows = range(row - RADIUS, row + RADIUS)
        right, left = geometry.pixels[3]
        assert band_set(right) == {(r, c) for r in rows for c in range(col + 10, col + 20)}
        assert band_set(left) == {(r, c) for r in rows for c in range(col - 20, col - 10)}

    def test_bands_of_a_horizontal_strip(self):
        horizontal = np.array([[i, 60, 30 + 10 * i] for i in range(10)])
        geometry = farfield.build_farfield_geometry(horizontal, [horizontal[:, 0]], IMG_SHAPE, RADIUS, DISTANCES)
        row, col = horizontal[4, 1:]
        cols = range(col - RADIUS, col + RADIUS)
        # the across coordinate of a strip along the columns points towards smaller rows
        above, below = geometry.pixels[4]
        assert band_set(above) == {(r, c) for c in cols for r in range(row - 19, row - 9)}
        assert band_set(below) == {(r, c) for c in cols for r in range(row + 11, row + 21)}

    def test_pixels_outside_the_image_are_left_out(self):
        near_border = VERTICAL.copy()
        near_border[:, 2] = IMG_SHAPE[1] - 15
        geometry = farfield.build_farfield_geometry(near_border, [near_border[:, 0]], IMG_SHAPE, RADIUS, DISTANCES)
        right, left = geometry.pixels[3]
        assert right[1].min() == IMG_SHAPE[1] - 5 and right[1].max() == IMG_SHAPE[1] - 1
        assert left[0].size == 2 * RADIUS * 10

    def test_pixels_close_to_another_strip_are_left_out(self):
        # a second strip 15 px to the right: its LEDs lie in the right bands of the first strip
        second = VERTICAL.copy()
        second[:, 0] += 10
        second[:, 2] += 15
        search_areas = np.vstack([VERTICAL, second])
        geometry = farfield.build_farfield_geometry(search_areas, [VERTICAL[:, 0], second[:, 0]], IMG_SHAPE, RADIUS,
                                                    DISTANCES)
        right, left = geometry.pixels[3]
        centers = search_areas[:, 1:3]
        for rows, cols in (right, left):
            distances = np.hypot(rows[:, None] - centers[:, 0], cols[:, None] - centers[:, 1])
            assert (distances >= DISTANCES[0]).all()
        assert right[0].size < 2 * RADIUS * 10
        assert left[0].size == 2 * RADIUS * 10

    def test_place_on_the_strip(self):
        search_areas = np.vstack([VERTICAL, [[10, 60, 30]]])
        geometry = farfield.build_farfield_geometry(search_areas, [VERTICAL[:, 0], [10]], IMG_SHAPE, RADIUS,
                                                    DISTANCES)
        assert geometry.strip_of_led.tolist() == [0] * 10 + [-1]
        assert np.all(np.diff(geometry.position_on_strip[:10]) > 0)
        assert geometry.pixels[10] is None

    def test_the_geometry_is_built_once_for_all_images(self):
        first = farfield.get_farfield_geometry(VERTICAL, [VERTICAL[:, 0]], IMG_SHAPE, RADIUS, DISTANCES)
        assert farfield.get_farfield_geometry(VERTICAL.copy(), [VERTICAL[:, 0]], IMG_SHAPE, RADIUS,
                                              DISTANCES) is first
        assert farfield.get_farfield_geometry(VERTICAL, [VERTICAL[:, 0]], IMG_SHAPE, RADIUS, (10.0, 30.0)) \
            is not first


class TestBandLevels:
    def test_mean_over_the_channel_pixels(self):
        values = np.array([1.0, 3.0, 100.0, 5.0, 7.0, 9.0])
        mask = np.array([True, True, False, True, True, True])
        np.testing.assert_allclose(farfield.mean_of_channel_pixels(values, mask, [3, 3]), [2.0, 7.0])

    def test_a_band_without_a_pixel_of_the_channel_has_no_level(self):
        values = np.array([1.0, 3.0, 5.0, 7.0])
        mask = np.array([True, True, False, False])
        levels = farfield.mean_of_channel_pixels(values, mask, [2, 0, 2, 0])
        assert levels[0] == 2.0
        assert np.isnan(levels[1:]).all()

    def test_no_band_at_all(self):
        assert np.isnan(farfield.mean_of_channel_pixels(np.zeros(0), np.zeros(0, dtype=bool), [0, 0])).all()


class TestSmoothing:
    def test_a_constant_level_stays(self):
        np.testing.assert_allclose(farfield.smooth_along_strip(np.full(40, 0.6)), 0.6)

    def test_the_outlier_of_a_single_led_is_dropped(self):
        levels = np.full(40, 0.6)
        levels[17] = 50.0
        np.testing.assert_allclose(farfield.smooth_along_strip(levels), 0.6)

    def test_a_slope_along_the_strip_is_kept(self):
        levels = 0.5 + 0.01 * np.arange(60)
        inner = slice(12, -12)
        np.testing.assert_allclose(farfield.smooth_along_strip(levels)[inner], levels[inner])

    def test_an_led_without_a_level_takes_that_of_its_neighbours(self):
        levels = np.full(40, 0.6)
        levels[[0, 20, 21, 39]] = np.nan
        np.testing.assert_allclose(farfield.smooth_along_strip(levels), 0.6)

    def test_a_side_without_any_level_has_none(self):
        assert np.isnan(farfield.smooth_along_strip(np.full(40, np.nan))).all()

    def test_short_strips(self):
        # the windows are cut short at the ends of the strip, here they hold the whole strip
        np.testing.assert_allclose(farfield.smooth_along_strip(np.array([0.5, 0.7, 0.9])), 0.7)
        assert farfield.smooth_along_strip(np.zeros(0)).size == 0

    def test_the_noise_of_the_estimate_is_reduced(self):
        levels = 0.6 + 0.2 * np.random.default_rng(1).standard_normal(126)
        assert np.std(farfield.smooth_along_strip(levels)) < 0.4 * np.std(levels)


class TestPedestals:
    def test_the_lower_side_is_the_pedestal(self):
        levels = np.column_stack((np.full(30, 0.9), np.full(30, 0.6)))
        pedestals = farfield.farfield_pedestals(levels, np.zeros(30, dtype=int), np.arange(30.0))
        np.testing.assert_allclose(pedestals, 0.6)

    def test_the_lower_side_is_chosen_led_by_led(self):
        levels = np.column_stack((np.linspace(0.2, 1.0, 60), np.full(60, 0.6)))
        pedestals = farfield.farfield_pedestals(levels, np.zeros(60, dtype=int), np.arange(60.0))
        assert pedestals[12] == pytest.approx(levels[12, 0], abs=1e-12)
        assert levels[12, 0] < 0.6 < levels[50, 0]
        assert pedestals[50] == pytest.approx(0.6)

    def test_a_side_without_a_level_leaves_the_other(self):
        levels = np.column_stack((np.full(30, np.nan), np.full(30, 0.6)))
        pedestals = farfield.farfield_pedestals(levels, np.zeros(30, dtype=int), np.arange(30.0))
        np.testing.assert_allclose(pedestals, 0.6)

    def test_strips_are_smoothed_separately_and_in_their_order_along_the_strip(self):
        # two strips with different levels, their LEDs interleaved and the second strip given backwards
        strip_of_led = np.tile([0, 1], 30)
        position = np.where(strip_of_led == 0, np.arange(60.0), -np.arange(60.0))
        level = np.where(strip_of_led == 0, 0.5, 2.0 + 0.01 * position)
        pedestals = farfield.farfield_pedestals(np.column_stack((level, level)), strip_of_led, position)
        np.testing.assert_allclose(pedestals[strip_of_led == 0], 0.5)
        inner = np.flatnonzero(strip_of_led == 1)[10:-10]
        np.testing.assert_allclose(pedestals[inner], level[inner])

    def test_an_led_on_no_strip_or_without_any_level_has_no_pedestal(self):
        levels = np.column_stack((np.full(5, 0.6), np.full(5, 0.6)))
        assert np.isnan(farfield.farfield_pedestals(levels, np.full(5, -1), np.arange(5.0))).all()
        assert np.isnan(farfield.farfield_pedestals(np.full((5, 2), np.nan), np.zeros(5, dtype=int),
                                                    np.arange(5.0))).all()


# ------------------------------------------------------------------------------------------------------- step 3
STEP3_SHAPE = (100, 120)
# RGGB pattern as in rawpy raw_colors_visible: 0 red, 1 and 3 green, 2 blue
FILTER_ARRAY = np.tile(np.array([[0, 1], [3, 2]]), (STEP3_SHAPE[0] // 2, STEP3_SHAPE[1] // 2))
BLACK_LEVELS = np.array([2040.3, 2041.7, 2042.1, 2043.9])
STRIP = np.array([[i, 14 + 10 * i, 60] for i in range(8)])
LINE_INDICES = [STRIP[:, 0]]


def make_config(farfield_distances=None):
    config = configparser.ConfigParser()
    config['DEFAULT'] = {'img_directory': '.', 'img_rotation': 'auto'}
    config['find_search_areas'] = {'search_area_radius': str(RADIUS)}
    config['analyse_photo'] = {'num_skip_leds': '0'}
    if farfield_distances is not None:
        config['analyse_photo']['farfield_distances'] = farfield_distances
    return config


def make_image(monkeypatch, background):
    """A RAW image of the strip: an LED in every search area on a background given per pixel in stored counts."""
    stored = np.asarray(background, dtype=np.float64) + BLACK_LEVELS[FILTER_ARRAY]
    for _, row, col in STRIP:
        stored[row - 2:row + 2, col - 2:col + 2] += 3000.0
    raw = RawImage(np.round(stored).astype(np.uint16), FILTER_ARRAY, BLACK_LEVELS, 16383.0,
                   image_reading.UPRIGHT_ORIENTATION)
    monkeypatch.setattr(image_reading, '_read_raw_image', lambda filename: raw)
    return ImageChannels('img.CR3')


def pedestal_by_definition(img_channels, channel, row, col, side):
    """Level of the band on one side of an LED of the vertical strip, from the arrays of the whole image."""
    data, mask = img_channels.channel_data_and_mask(channel)
    cols = slice(col + 10, col + 20) if side == 'right' else slice(col - 20, col - 10)
    band = np.s_[row - RADIUS:row + RADIUS, cols]
    return data[band][mask[band]].mean()


class TestStep3:
    @pytest.mark.parametrize('channel', [0, 1, 2])
    def test_no_pedestal_without_the_option(self, monkeypatch, channel):
        img_channels = make_image(monkeypatch, np.full(STEP3_SHAPE, 40.0))
        led_data = generate_analysis_data('img.CR3', channel, STRIP, LINE_INDICES, make_config(), fit_leds=False,
                                          img_channels=img_channels)
        assert all(data.farbg_sum_color_value is None and data.farbg_pedestal is None for data in led_data)
        assert len(str(led_data[0]).split(',')) == 5

    @pytest.mark.parametrize('channel', [0, 1, 2])
    def test_a_flat_background_is_removed(self, monkeypatch, channel):
        flat = make_image(monkeypatch, np.full(STEP3_SHAPE, 40.0))
        with_pedestal = generate_analysis_data('img.CR3', channel, STRIP, LINE_INDICES, make_config('10 20'),
                                               fit_leds=False, img_channels=flat)
        dark = make_image(monkeypatch, np.zeros(STEP3_SHAPE))
        without = generate_analysis_data('img.CR3', channel, STRIP, LINE_INDICES, make_config('10 20'),
                                         fit_leds=False, img_channels=dark)
        for led, led_dark in zip(with_pedestal, without):
            # the corrected sum of the image with the background is the one of the image without it
            assert led.farbg_sum_color_value == pytest.approx(led_dark.farbg_sum_color_value, abs=0.2)
            assert led.farbg_sum_color_value == pytest.approx(led.sum_color_value - led.farbg_pedestal, rel=1e-9)
            assert led.farbg_pedestal > 0.02 * led.sum_color_value
            assert led.sum_color_value > led_dark.sum_color_value

    @pytest.mark.parametrize('channel', [0, 1, 2])
    def test_the_pedestal_is_the_lower_side_times_the_channel_pixels(self, monkeypatch, channel):
        # brighter towards larger columns: the left band is the lower one
        background = 20.0 + 0.5 * np.arange(STEP3_SHAPE[1]) * np.ones(STEP3_SHAPE)
        img_channels = make_image(monkeypatch, background)
        led_data = generate_analysis_data('img.CR3', channel, STRIP, LINE_INDICES, make_config('10 20'),
                                          fit_leds=False, img_channels=img_channels)
        _, mask = img_channels.channel_data_and_mask(channel)
        for data, (_, row, col) in zip(led_data, STRIP):
            left = pedestal_by_definition(img_channels, channel, row, col, 'left')
            assert left < pedestal_by_definition(img_channels, channel, row, col, 'right')
            num_channel_pixels = mask[row - RADIUS:row + RADIUS, col - RADIUS:col + RADIUS].sum()
            # the background does not change along the strip, so the smoothing leaves the level as it is
            assert data.farbg_pedestal == pytest.approx(num_channel_pixels * left, rel=1e-6)

    def test_an_led_without_a_strip_keeps_its_plain_sum(self, monkeypatch):
        img_channels = make_image(monkeypatch, np.full(STEP3_SHAPE, 40.0))
        search_areas = np.vstack([STRIP, [[8, 50, 20]]])
        led_data = generate_analysis_data('img.CR3', 0, search_areas, [STRIP[:, 0], np.array([8])],
                                          make_config('10 20'), fit_leds=False, img_channels=img_channels)
        assert led_data[-1].farbg_pedestal == 0.0
        assert led_data[-1].farbg_sum_color_value == pytest.approx(float(led_data[-1].sum_color_value))
        assert led_data[0].farbg_pedestal > 0.0

    def test_the_pixels_are_calibrated_once_for_all_channels(self, monkeypatch):
        img_channels = make_image(monkeypatch, np.full(STEP3_SHAPE, 40.0))
        calibrated = []
        channel_data = image_reading._channel_data
        monkeypatch.setattr(image_reading, '_channel_data',
                            lambda raw: calibrated.append(raw.data.size) or channel_data(raw))
        for channel in [0, 1, 2]:
            generate_analysis_data('img.CR3', channel, STRIP, LINE_INDICES, make_config('10 20'), fit_leds=False,
                                   img_channels=img_channels)
        assert len(calibrated) == 1

    def test_the_plain_quantities_do_not_change(self, monkeypatch):
        img_channels = make_image(monkeypatch, np.full(STEP3_SHAPE, 40.0))
        plain = generate_analysis_data('img.CR3', 1, STRIP, LINE_INDICES, make_config(), fit_leds=False,
                                       img_channels=img_channels)
        corrected = generate_analysis_data('img.CR3', 1, STRIP, LINE_INDICES, make_config('10 20'), fit_leds=False,
                                           img_channels=img_channels)
        for a, b in zip(plain, corrected):
            assert (a.sum_color_value, a.mean_color_value, a.max_color_value) == \
                (b.sum_color_value, b.mean_color_value, b.max_color_value)


class TestResultFiles:
    IMG_INFOS = np.array([['1', 'img.jpg', '12:00:00', '0']])

    def test_layouts_with_the_pedestal(self):
        assert COLUMN_LAYOUTS[len(BASE_COLUMNS + FARBG_COLUMNS)] == BASE_COLUMNS + FARBG_COLUMNS
        assert COLUMN_LAYOUTS[len(BASE_COLUMNS + FARBG_COLUMNS + LED_CENTER_COLUMNS + FIT_COLUMNS)] == \
            BASE_COLUMNS + FARBG_COLUMNS + LED_CENTER_COLUMNS + FIT_COLUMNS

    @pytest.mark.parametrize('fit_leds', [False, True])
    def test_header_names_the_columns_of_the_rows(self, fit_leds):
        header = _create_header(0, '1', 'img.jpg', self.IMG_INFOS, 'root', fit_leds, True)
        names = [name.strip() for name in header.split('\n')[1].lstrip('#').split('//')[0].split(',')]
        assert names[2:7] == ['sum_col_value', 'average_col_value', 'max_col_value', 'farbg_sum_col_value',
                              'farbg_pedestal']
        assert ('fit_success' in names) == fit_leds

    def test_result_file_written_by_step_3_is_read_with_its_names(self, monkeypatch, tmp_path):
        img_channels = make_image(monkeypatch, np.full(STEP3_SHAPE, 40.0))
        led_data = generate_analysis_data('img.CR3', 0, STRIP, LINE_INDICES, make_config('10 20'), fit_leds=False,
                                          img_channels=img_channels)
        channel_dir = tmp_path / 'analysis' / 'channel0'
        channel_dir.mkdir(parents=True)
        header = _create_header(0, '1', 'img.jpg', self.IMG_INFOS, 'root', False, True)
        content = header + ''.join(str(data) for data in led_data)
        (channel_dir / '1_led_positions.csv').write_text(content, encoding='utf-8')
        monkeypatch.chdir(tmp_path)
        columns = _get_column_names(0)
        assert columns == BASE_COLUMNS + FARBG_COLUMNS + ['width', 'height']
        values = np.loadtxt(channel_dir / '1_led_positions.csv', delimiter=',')
        named = dict(zip(columns[1:], values[0]))
        assert named['farbg_sum_col_val'] == pytest.approx(led_data[0].farbg_sum_color_value, rel=1e-4)
        assert named['farbg_pedestal'] == pytest.approx(led_data[0].farbg_pedestal, rel=1e-4)
        assert named['sum_col_val'] - named['farbg_pedestal'] == pytest.approx(named['farbg_sum_col_val'], rel=1e-3)


class TestConfig:
    def test_template_holds_the_option_without_a_pedestal(self, tmp_path, monkeypatch):
        monkeypatch.chdir(tmp_path)
        ConfigData(load_config_file=False)
        config = ConfigData()
        assert config.get('analyse_photo', 'farfield_distances') == 'None'
        assert farfield.parse_farfield_distances(config.get('analyse_photo', 'farfield_distances'), 10) is None

    def test_template_with_distances(self, tmp_path, monkeypatch):
        monkeypatch.chdir(tmp_path)
        ConfigData(load_config_file=False, farfield_distances='150 300')
        assert farfield.parse_farfield_distances(ConfigData().get('analyse_photo', 'farfield_distances'), 10) == \
            (150.0, 300.0)

    def test_a_config_written_before_the_option_estimates_no_pedestal(self, tmp_path, monkeypatch):
        monkeypatch.chdir(tmp_path)
        (tmp_path / 'config.ini').write_text('[analyse_photo]\nnum_skip_leds = 0\n', encoding='utf-8')
        assert farfield.parse_farfield_distances(ConfigData().get('analyse_photo', 'farfield_distances'), 10) is None
