"""Unit tests for the LED search areas found on the reference image.

Every LED found is blanked out, so that the next brightest pixel of the image belongs to
another LED. The blanked box used to be cut with a negative lower bound for an LED within
search_area_radius of the top or the left border of the image, which makes the slice empty:
that LED stayed the brightest pixel and was found again for every one of the max_num_leds
slots, while the LEDs behind it were never found at all.
"""
import numpy as np

from ledsa.data_extraction.step_1_functions import find_search_areas

SEARCH_AREA_RADIUS = 10
MAX_NUM_LEDS = 40
# every pixel that is not an LED is darker than the threshold, so the LEDs are the only finds
PIXEL_VALUE_PERCENTILE = 50


def image_with_leds(*positions, shape=(200, 200)) -> np.ndarray:
    """Return an image holding one bright pixel per given (row, column) position."""
    image = np.zeros(shape, dtype=np.uint8)
    for row, column in positions:
        image[row, column] = 255
    return image


def found_positions(image: np.ndarray) -> list:
    """Return the (row, column) positions of the LEDs found on an image, sorted."""
    search_areas, _ = find_search_areas(image, SEARCH_AREA_RADIUS,
                                        pixel_value_percentile=PIXEL_VALUE_PERCENTILE,
                                        max_n_leds=MAX_NUM_LEDS)
    return sorted((int(row), int(column)) for _, row, column in search_areas)


def test_every_led_away_from_the_borders_is_found_once():
    positions = [(50, 50), (100, 100), (150, 50)]

    assert found_positions(image_with_leds(*positions)) == positions


def test_an_led_at_the_top_border_is_found_once():
    positions = [(SEARCH_AREA_RADIUS - 7, 100), (100, 100), (150, 50)]

    assert found_positions(image_with_leds(*positions)) == positions


def test_an_led_at_the_left_border_is_found_once():
    positions = [(100, SEARCH_AREA_RADIUS - 7), (100, 100), (150, 50)]

    assert found_positions(image_with_leds(*positions)) == sorted(positions)


def test_an_led_in_the_top_left_corner_is_found_once():
    positions = [(0, 0), (100, 100), (150, 50)]

    assert found_positions(image_with_leds(*positions)) == positions
