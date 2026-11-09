"""Unit tests for the analysis of all search areas of an image at once.

Step 3 reads only the pixels of the search areas and side bands from an image, calibrates them once for all
channels and takes every quantity along the axes of a stack of search areas. The results have to be those of
the definitions applied LED by LED to the arrays of the whole image.
"""
import configparser

import numpy as np
import pytest

from ledsa.core import image_reading
from ledsa.core.image_reading import ImageChannels, RawImage
from ledsa.data_extraction.sideband_background import get_sideband_pixels
from ledsa.data_extraction.step_3_functions import (_get_analysis_pixels, generate_analysis_data,
                                                    get_sideband_parameters)

SEARCH_AREA_RADIUS = 4
IMG_SHAPE = (40, 48)
# RGGB pattern as in rawpy raw_colors_visible: 0 red, 1 and 3 green, 2 blue
FILTER_ARRAY = np.tile(np.array([[0, 1], [3, 2]]), (IMG_SHAPE[0] // 2, IMG_SHAPE[1] // 2))
SIDEWAYS = 6
# A vertical strip and an LED alone in its array, whose strip direction and therefore side bands are unknown.
# The search area of LED 4 is cut short by the lower border of the upright image, that of LED 5 by the right
# border of the image turned sideways.
SEARCH_AREAS = np.array([[0, 8, 20], [1, 16, 20], [2, 24, 20], [3, 32, 20], [4, 38, 20], [5, 20, 38]])
LINE_INDICES = [np.array([0, 1, 2, 3, 4]), np.array([5])]
QUANTITIES = ['sum_color_value', 'mean_color_value', 'max_color_value', 'bgsub_sum_color_value',
              'sidebg_sum_color_value', 'sidebg_pedestal', 'bgsub2_sum_color_value', 'noise_sum_color_value']


def make_config():
    config = configparser.ConfigParser()
    config['DEFAULT'] = {'img_directory': '.', 'img_rotation': 'auto'}
    config['find_search_areas'] = {'search_area_radius': str(SEARCH_AREA_RADIUS)}
    config['analyse_photo'] = {'num_skip_leds': '0', 'sideband_distances': '6 12', 'background_floor': 'None',
                               'sideband_exclusion_margin': 'None'}
    config['analyse_positions'] = {'merge_led_array_indices': 'None'}
    return config


def sideband_pixels_of(img_channels, config):
    distances, _, margin = get_sideband_parameters(config)
    return get_sideband_pixels(SEARCH_AREAS, LINE_INDICES, img_channels.shape, SEARCH_AREA_RADIUS, distances, margin)


@pytest.fixture(params=[image_reading.UPRIGHT_ORIENTATION, SIDEWAYS])
def img_channels(request, monkeypatch):
    """A RAW image with a black level and a read noise of its own for every Bayer color."""
    stored = np.random.default_rng(0).integers(1900, 4000, IMG_SHAPE).astype(np.uint16)
    raw = RawImage(stored, FILTER_ARRAY, np.array([2040.3, 2041.7, 2042.1, 2043.9]), 16383.0, request.param,
                   np.array([1.5, 2.5, 3.5, 4.5]))
    monkeypatch.setattr(image_reading, '_read_raw_image', lambda filename: raw)
    return ImageChannels('img.CR3')


def quantities_by_definition(img_channels, channel, row, col, sideband_pixels):
    """The quantities of one LED, from their definitions on the arrays of the whole image."""
    data, mask, noise = img_channels.channel_data_mask_and_noise(channel)
    box = np.s_[row - SEARCH_AREA_RADIUS:row + SEARCH_AREA_RADIUS, col - SEARCH_AREA_RADIUS:col + SEARCH_AREA_RADIUS]
    values, in_channel = data[box], mask[box]
    ring = np.ones(values.shape, dtype=bool)
    ring[2:-2, 2:-2] = False
    ring_pixels = values[ring & in_channel]
    num_channel_pixels = np.count_nonzero(in_channel)
    sideband_values = np.zeros(0) if sideband_pixels is None else data[sideband_pixels][mask[sideband_pixels]]
    sideband_background = np.mean(sideband_values, dtype=np.float64) if sideband_values.size else np.nan
    box_sum = float(np.sum(values))
    return {'sum_color_value': np.sum(values), 'mean_color_value': np.mean(values),
            'max_color_value': np.amax(values),
            'bgsub_sum_color_value': np.sum(values.astype(np.float64) - np.median(ring_pixels) * in_channel),
            'sidebg_sum_color_value': box_sum - num_channel_pixels * sideband_background,
            'sidebg_pedestal': num_channel_pixels * sideband_background,
            'bgsub2_sum_color_value': box_sum - num_channel_pixels * np.mean(ring_pixels, dtype=np.float64),
            'noise_sum_color_value': np.sqrt(np.sum(noise[box].astype(np.float64) ** 2))}


class TestStackedAnalysis:
    @pytest.mark.parametrize('channel', [0, 1, 2])
    def test_quantities_are_those_of_their_definitions(self, img_channels, channel):
        config = make_config()
        led_data = generate_analysis_data('img.CR3', channel, SEARCH_AREAS, LINE_INDICES, config, fit_leds=False,
                                          img_channels=img_channels)
        sideband_pixels = sideband_pixels_of(img_channels, config)
        assert [(data.led_id, data.led_array) for data in led_data] == [(0, 0), (1, 0), (2, 0), (3, 0), (4, 0),
                                                                        (5, 1)]
        for data, (led_id, row, col) in zip(led_data, SEARCH_AREAS):
            expected = quantities_by_definition(img_channels, channel, row, col, sideband_pixels[led_id])
            for quantity in QUANTITIES:
                assert getattr(data, quantity) == pytest.approx(expected[quantity], rel=1e-12, nan_ok=True), \
                    f'{quantity} of LED {led_id}'

    def test_a_search_area_at_the_border_is_cut_short(self, img_channels):
        # as data[row - R:row + R, col - R:col + R] cuts it; the search areas of the other shape are a stack of
        # their own, so test_quantities_are_those_of_their_definitions covers both
        analysis_pixels = _get_analysis_pixels(SEARCH_AREAS, SEARCH_AREAS[:, 0], sideband_pixels_of(img_channels,
                                               make_config()), img_channels.shape, SEARCH_AREA_RADIUS)
        stacks = {shape: positions.tolist() for positions, shape, _ in analysis_pixels.box_groups}
        if img_channels.shape == IMG_SHAPE:
            assert stacks == {(8, 8): [0, 1, 2, 3, 5], (6, 8): [4]}
        else:
            assert stacks == {(8, 8): [0, 1, 2, 3, 4], (8, 6): [5]}

    def test_the_led_without_side_bands_has_no_side_band_background(self, img_channels):
        led_data = generate_analysis_data('img.CR3', 0, SEARCH_AREAS, LINE_INDICES, make_config(), fit_leds=False,
                                          img_channels=img_channels)
        assert np.isnan(led_data[5].sidebg_pedestal)
        assert np.isfinite([data.sidebg_pedestal for data in led_data[:5]]).all()

    def test_only_the_analysed_pixels_are_calibrated_once_for_all_channels(self, img_channels, monkeypatch):
        calibrated = []
        channel_data = image_reading._channel_data
        monkeypatch.setattr(image_reading, '_channel_data',
                            lambda raw: calibrated.append(raw.data.size) or channel_data(raw))
        for channel in [0, 1, 2]:
            generate_analysis_data('img.CR3', channel, SEARCH_AREAS, LINE_INDICES, make_config(), fit_leds=False,
                                   img_channels=img_channels)
        assert len(calibrated) == 1
        assert calibrated[0] < np.prod(IMG_SHAPE)

    def test_skipped_leds_are_not_analysed(self, img_channels):
        config = make_config()
        config['analyse_photo']['num_skip_leds'] = '1'
        led_data = generate_analysis_data('img.CR3', 0, SEARCH_AREAS, LINE_INDICES, config, fit_leds=False,
                                          img_channels=img_channels)
        assert [data.led_id for data in led_data] == [0, 2, 4]
