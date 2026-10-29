"""Unit tests for the light path of an LED through the layers.

The traversed distances are accumulated per layer, so an LED or a camera outside the layers
loses the part of its light path that lies outside. The inconsistency was detected and then
turned into a row of NaN in the distance array: the nonlinear solver returned zero extinction
coefficients for every image of the run without a warning, while the linear solver stopped
with a numpy error naming neither the LED nor the domain bounds.
"""
import numpy as np
import pytest

from ledsa.analysis.Experiment import Camera, Experiment, InconsistentDistanceError, LED, Layers
from ledsa.analysis.ExtinctionCoefficientsNonLinear import ExtinctionCoefficientsNonLinear

NUM_LAYERS = 10
BOTTOM_BORDER = 0.0
TOP_BORDER = 3.0
CAMERA = Camera(0, 0, 1.5)
LED_DISTANCE = 4  # horizontal distance between the LED array and the camera


class ExperimentStub(Experiment):
    """Experiment whose LEDs are given by their heights instead of read from the analysis files."""

    def __init__(self, heights, camera=CAMERA):
        self.heights = heights
        super().__init__(layers=Layers(NUM_LAYERS, BOTTOM_BORDER, TOP_BORDER), led_array=0,
                         camera=camera)

    def set_leds(self) -> None:
        self.leds = [LED(led_id, 0, LED_DISTANCE, height) for led_id, height in
                     enumerate(self.heights)]
        self.num_leds = len(self.leds)


def distance_to_camera(height, camera=CAMERA):
    return np.sqrt(LED_DISTANCE ** 2 + (height - camera.pos_z) ** 2)


def test_the_light_path_of_an_led_inside_the_layers_adds_up():
    experiment = ExperimentStub([2.5])

    distances = experiment.calc_traversed_dist_per_layer(experiment.leds[0])

    assert np.sum(distances) == pytest.approx(distance_to_camera(2.5))


def test_an_led_above_the_layers_is_reported():
    experiment = ExperimentStub([TOP_BORDER + 0.5])

    with pytest.raises(InconsistentDistanceError, match='domain_bounds'):
        experiment.calc_traversed_dist_per_layer(experiment.leds[0])


def test_an_led_below_the_layers_is_reported():
    experiment = ExperimentStub([BOTTOM_BORDER - 0.5])

    with pytest.raises(InconsistentDistanceError, match='domain_bounds'):
        experiment.calc_traversed_dist_per_layer(experiment.leds[0])


def test_a_camera_outside_the_layers_is_reported():
    experiment = ExperimentStub([2.5], camera=Camera(0, 0, TOP_BORDER + 0.5))

    with pytest.raises(InconsistentDistanceError, match='domain_bounds'):
        experiment.calc_traversed_dist_per_layer(experiment.leds[0])


def test_the_distance_array_of_leds_inside_the_layers_holds_their_light_paths():
    heights = [0.5, 1.5, 2.5]
    solver = ExtinctionCoefficientsNonLinear(ExperimentStub(heights))

    distances = solver.calc_distance_array()

    assert not np.isnan(distances).any()
    # the distance array is built from the top LED down, the reverse of the LED order
    assert np.sum(distances, axis=1) == pytest.approx([distance_to_camera(height) for height in
                                                       reversed(heights)])


def test_the_distance_array_names_the_led_outside_the_layers_instead_of_holding_nan():
    solver = ExtinctionCoefficientsNonLinear(ExperimentStub([0.5, 1.5, TOP_BORDER + 0.5]))

    with pytest.raises(InconsistentDistanceError, match='LED 2'):
        solver.calc_distance_array()
