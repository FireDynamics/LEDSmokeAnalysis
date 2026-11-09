"""Unit tests for the linear solver of the extinction coefficients.

The intensities are not clipped at zero, so the relative intensity of an LED that has vanished in the noise
of the sensor can be zero or negative. Such an LED is no measurement and must not reach the coefficients.
"""
import numpy as np

from ledsa.analysis.ExtinctionCoefficientsLinear import ExtinctionCoefficientsLinear

# 4 LEDs over 3 layers: the first three cross one layer each, the fourth crosses all of them
DISTANCES = np.array([[1.0, 0.0, 0.0],
                      [0.0, 1.0, 0.0],
                      [0.0, 0.0, 1.0],
                      [1.0, 1.0, 1.0]])
SIGMAS = np.array([0.1, 0.2, 0.3])
REL_INTENSITIES = np.exp(-DISTANCES @ SIGMAS)


def make_solver(distances=DISTANCES) -> ExtinctionCoefficientsLinear:
    """A linear solver with the geometry above, built without __init__ as the config is not needed."""
    solver = ExtinctionCoefficientsLinear.__new__(ExtinctionCoefficientsLinear)
    solver.distances_per_led_and_layer = distances
    solver.lambda_reg = 1e-6
    return solver


class TestLinearSolver:
    def test_the_solution_recovers_the_coefficients(self):
        np.testing.assert_allclose(make_solver().calc_coefficients_of_img(REL_INTENSITIES), SIGMAS, atol=1e-3)

    def test_an_led_at_or_below_zero_is_left_out(self):
        # whatever it reports, the result is that of the other LEDs alone
        without_the_led = make_solver(DISTANCES[1:]).calc_coefficients_of_img(REL_INTENSITIES[1:])
        for value in [0.0, -0.3]:
            rel_intensities = REL_INTENSITIES.copy()
            rel_intensities[0] = value
            np.testing.assert_allclose(make_solver().calc_coefficients_of_img(rel_intensities), without_the_led,
                                       atol=1e-9)

    def test_an_led_without_a_finite_value_is_left_out(self):
        # an LED whose reference intensity is 0 gives NaN or an infinite relative intensity
        without_the_led = make_solver(DISTANCES[1:]).calc_coefficients_of_img(REL_INTENSITIES[1:])
        for value in [np.nan, np.inf, -np.inf]:
            rel_intensities = REL_INTENSITIES.copy()
            rel_intensities[0] = value
            np.testing.assert_allclose(make_solver().calc_coefficients_of_img(rel_intensities), without_the_led,
                                       atol=1e-9)

    def test_an_image_without_any_measured_led_gives_nan(self):
        rel_intensities = np.array([0.0, -0.1, np.nan, -2.0])
        assert np.isnan(make_solver().calc_coefficients_of_img(rel_intensities)).all()
