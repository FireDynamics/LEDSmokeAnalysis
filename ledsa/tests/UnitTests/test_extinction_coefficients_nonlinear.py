"""Unit tests for the nonlinear solver: its gradient, and the solution it reaches.

The solver used to be handed no gradient, so the minimizer estimated one from finite
differences of intensities that fall off exponentially. It stopped far from the minimum, at a
point that depended on where it started - and it starts from the coefficients of the previous
image. The same image therefore gave different extinction coefficients depending on the image
before it, with errors of the order of the coefficients themselves.
"""
import numpy as np
import pytest

from ledsa.analysis.Experiment import Experiment, Layers, Camera, LED
from ledsa.analysis.ExtinctionCoefficientsNonLinear import ExtinctionCoefficientsNonLinear, \
    second_difference_operator

NUM_LAYERS = 20
NUM_LEDS = 100
BOTTOM_BORDER = 0.0
TOP_BORDER = 3.0
CAMERA = Camera(0, 0, 2)
LED_DISTANCE = 4  # horizontal distance between the LED array and the camera
LAYER_HEIGHTS = np.linspace(BOTTOM_BORDER, TOP_BORDER, NUM_LAYERS)


class ExperimentStub(Experiment):
    """Experiment whose LEDs are given by their heights instead of read from the analysis files."""

    def set_leds(self) -> None:
        self.leds = [LED(led_id, 0, LED_DISTANCE, height) for led_id, height in
                     enumerate(np.linspace(BOTTOM_BORDER + 0.05, TOP_BORDER - 0.05, NUM_LEDS))]
        self.num_leds = len(self.leds)


def make_solver(previous_coefficients=None, **kwargs):
    """Create a solver on the LED array of the acceptance tests.

    :param previous_coefficients: Coefficients of a preceding image, which the solver takes as the
        starting point of its minimization. None leaves it starting from zeros.
    """
    solver = ExtinctionCoefficientsNonLinear(
        ExperimentStub(layers=Layers(NUM_LAYERS, BOTTOM_BORDER, TOP_BORDER), led_array=0, camera=CAMERA),
        **kwargs)
    solver.distances_per_led_and_layer = solver.calc_distance_array()
    if previous_coefficients is not None:
        solver.coefficients_per_image_and_layer.append(previous_coefficients)
    return solver


def linear_profile_with_noise(seed):
    """Extinction coefficients rising linearly with the height, as the test images hold them."""
    rng = np.random.default_rng(seed)
    return np.clip(0.15 * LAYER_HEIGHTS + rng.normal(0, 0.05, NUM_LAYERS), a_min=0.0, a_max=None)


def rmse(computed, expected):
    return np.sqrt(np.mean((computed - expected) ** 2))


def test_intensities_follow_the_beer_lambert_law():
    solver = make_solver()
    sigmas = linear_profile_with_noise(0)

    intensities = solver.calc_intensities(sigmas)

    for led, distances in enumerate(solver.distances_per_led_and_layer):
        transmission = np.prod([np.exp(-sigma * distance) for sigma, distance in zip(sigmas, distances)])
        assert intensities[led] == pytest.approx(transmission)


def test_cost_gradient_matches_the_finite_differences_of_the_cost():
    solver = make_solver()
    sigmas = linear_profile_with_noise(1)
    target = solver.calc_intensities(np.full(NUM_LAYERS, 0.2))
    step = 1e-7

    gradient = solver.cost_gradient(sigmas, target)

    for layer, direction in enumerate(np.eye(NUM_LAYERS)):
        central_difference = (solver.cost_function(sigmas + step * direction, target) -
                              solver.cost_function(sigmas - step * direction, target)) / (2 * step)
        assert gradient[layer] == pytest.approx(central_difference, abs=1e-7)


@pytest.mark.parametrize('seed', range(4))
def test_profile_is_reconstructed_from_its_intensities(seed):
    sigmas = linear_profile_with_noise(seed)
    target = make_solver().calc_intensities(sigmas)

    computed = make_solver().calc_coefficients_of_img(target)

    assert rmse(computed, sigmas) < 1e-4


@pytest.mark.parametrize('seed', range(4))
def test_coefficients_do_not_depend_on_the_coefficients_of_the_previous_image(seed):
    sigmas = linear_profile_with_noise(seed)
    # The intensities of an image carry the error of the LED fit, so the solver has to reach the
    # minimum of its cost function rather than stop at the first point that looks flat
    rng = np.random.default_rng(100 + seed)
    target = np.clip(make_solver().calc_intensities(sigmas) + rng.normal(0, 3e-3, NUM_LEDS), 1e-6, 1.0)
    previous_images = [None,                               # first image of the run
                       np.full(NUM_LAYERS, 0.5),           # a smokier image before this one
                       sigmas[::-1].copy()]                # smoke layered the other way round

    results = [make_solver(previous_coefficients=previous).calc_coefficients_of_img(target)
               for previous in previous_images]

    for computed in results:
        assert np.allclose(computed, results[0], atol=1e-3)
        assert rmse(computed, sigmas) < 0.03


def test_an_undisturbed_image_gives_no_extinction_without_a_warning(recwarn):
    """The reference image matches its target exactly, where the rmse term has no gradient."""
    solver = make_solver()

    computed = solver.calc_coefficients_of_img(np.ones(NUM_LEDS))

    assert computed == pytest.approx(np.zeros(NUM_LAYERS))
    assert [str(warning.message) for warning in recwarn] == []


def test_an_exhausted_iteration_budget_is_reported():
    solver = make_solver(num_iterations=5)
    target = solver.calc_intensities(linear_profile_with_noise(0))

    with pytest.warns(UserWarning, match='num_iterations'):
        solver.calc_coefficients_of_img(target)


def test_second_difference_operator_holds_one_row_per_second_difference():
    operator = second_difference_operator(4)

    assert operator.tolist() == [[1, -2, 1, 0],
                                 [0, 1, -2, 1]]


@pytest.mark.parametrize('num_layers', [1, 2])
def test_too_few_layers_for_a_second_difference_leave_out_the_curvature(num_layers):
    """A domain of one or two layers has no second difference, which must not end up as NaN."""
    solver = ExtinctionCoefficientsNonLinear(
        ExperimentStub(layers=Layers(num_layers, BOTTOM_BORDER, TOP_BORDER), led_array=0, camera=CAMERA))
    solver.distances_per_led_and_layer = solver.calc_distance_array()
    sigmas = np.full(num_layers, 0.2)
    target = solver.calc_intensities(sigmas)

    assert np.isfinite(solver.cost_function(sigmas, target))
    assert np.all(np.isfinite(solver.cost_gradient(sigmas, target)))
