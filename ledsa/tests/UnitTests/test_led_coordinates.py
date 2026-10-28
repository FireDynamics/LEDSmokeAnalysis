"""Unit tests for the physical coordinates computed from the edge LEDs of a LED array.

led_array_edge_coordinates holds the coordinates of the two LEDs of led_array_edge_indices in
the same order, while step 2 writes the LED array indices file in the reverse order of the
array: the coordinates that come first in the config belong to the *last* LED of that file.
Pairing them the other way round mirrors the height of every LED of the array without leaving a
trace in the result, because the transformation between pixel and room coordinates is solved
from the two edge LEDs alone and fits them exactly either way. The LED array indices file is
therefore written here by step 2 itself instead of by hand.
"""
import numpy as np
import pytest

from ledsa.core.ConfigData import ConfigData
from ledsa.data_extraction.step_2_functions import generate_led_array_indices_files
from ledsa.ledpositions.coordinates import _calculate_3d_coordinates

TOP_EDGE_LED = 0  # named first in led_array_edge_indices, so its coordinates come first
MIDDLE_LED = 1
BOT_EDGE_LED = 2
LED_ARRAY = [TOP_EDGE_LED, MIDDLE_LED, BOT_EDGE_LED]
PIXEL_POSITIONS = {TOP_EDGE_LED: (50, 100), MIDDLE_LED: (150, 100), BOT_EDGE_LED: (250, 100)}
TOP_EDGE_COORDINATES = (0.0, 4.0, 3.0)
BOT_EDGE_COORDINATES = (0.0, 4.0, 1.0)


def create_analysis_files(tmp_path, monkeypatch) -> None:
    """Write the config and the step 1 and step 2 results of a single vertical LED array."""
    monkeypatch.chdir(tmp_path)
    edge_coordinates = ' '.join(str(value) for value in TOP_EDGE_COORDINATES + BOT_EDGE_COORDINATES)
    ConfigData(load_config_file=False, img_directory='imgs/', img_name_string='img{:04d}.jpg',
               num_arrays=1, led_array_edge_indices=f'{TOP_EDGE_LED} {BOT_EDGE_LED}',
               led_array_edge_coordinates=edge_coordinates)

    (tmp_path / 'analysis').mkdir(exist_ok=True)
    search_areas = '\n'.join(f'{led_id},{row},{column}' for led_id, (row, column) in
                             PIXEL_POSITIONS.items())
    search_areas_path = tmp_path / 'analysis' / 'led_search_areas.csv'
    search_areas_path.write_text(search_areas + '\n', encoding='utf-8')
    generate_led_array_indices_files([LED_ARRAY])


def fail_on_input(monkeypatch) -> None:
    """Let a prompt fail the test, so an option already in the config cannot be asked for again."""
    def fail(prompt=''):
        raise AssertionError(f'asked for an option that is set in the config: {prompt}')

    monkeypatch.setattr('builtins.input', fail)


def coordinates_of_leds(tmp_path, monkeypatch) -> dict:
    """Return the computed physical coordinates of every LED by its ID."""
    create_analysis_files(tmp_path, monkeypatch)
    fail_on_input(monkeypatch)
    computed = _calculate_3d_coordinates()
    return {int(row[0]): row[-3:] for row in computed}


def test_the_edge_leds_get_the_coordinates_they_are_configured_with(tmp_path, monkeypatch):
    coordinates = coordinates_of_leds(tmp_path, monkeypatch)

    assert coordinates[TOP_EDGE_LED] == pytest.approx(TOP_EDGE_COORDINATES)
    assert coordinates[BOT_EDGE_LED] == pytest.approx(BOT_EDGE_COORDINATES)


def test_the_led_between_the_edges_is_interpolated(tmp_path, monkeypatch):
    coordinates = coordinates_of_leds(tmp_path, monkeypatch)

    assert coordinates[MIDDLE_LED] == pytest.approx(np.mean([TOP_EDGE_COORDINATES,
                                                             BOT_EDGE_COORDINATES], axis=0))
