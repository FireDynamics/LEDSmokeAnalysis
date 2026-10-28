from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest

from ledsa.analysis.ExtinctionCoefficientsLinear import ExtinctionCoefficientsLinear

# 4 LEDs over 3 layers: the first three cross one layer each, the fourth crosses all of them
DISTANCES = np.array([[1.0, 0.0, 0.0],
                      [0.0, 1.0, 0.0],
                      [0.0, 0.0, 1.0],
                      [1.0, 1.0, 1.0]])
SIGMAS = np.array([0.1, 0.2, 0.3])
REL_INTENSITIES = np.exp(-DISTANCES @ SIGMAS)
SNR_FULL_WEIGHT = 10.0


def make_solver(min_layer_weight=None) -> ExtinctionCoefficientsLinear:
    """A linear solver with the geometry above, built without __init__ as the config is not needed."""
    solver = ExtinctionCoefficientsLinear.__new__(ExtinctionCoefficientsLinear)
    solver.distances_per_led_and_layer = DISTANCES
    solver.lambda_reg = 1e-6
    solver.snr_full_weight = SNR_FULL_WEIGHT
    solver.min_layer_weight = min_layer_weight
    return solver


class TestWhichReferencePropertiesCanBeWeighted:
    def test_a_property_without_a_noise_column_is_refused(self):
        experiment = SimpleNamespace(channel=0, path='.', led_array=0)
        with pytest.raises(ValueError, match=r'snr_full_weight needs a reference property with a noise '
                                             r'column.*sidebg_sum_col_val has none.*ob_sum_col_val'):
            ExtinctionCoefficientsLinear(experiment, reference_property='sidebg_sum_col_val',
                                   snr_full_weight=SNR_FULL_WEIGHT)

    def test_the_unclipped_sum_brings_its_noise_column(self):
        experiment = SimpleNamespace(channel=0, path='.', led_array=0)
        coefficients = ExtinctionCoefficientsLinear(experiment, reference_property='ob_sum_col_val',
                                              snr_full_weight=SNR_FULL_WEIGHT)
        assert coefficients.noise_property == 'ob_noise_sum_col_val'

    def test_without_the_option_no_noise_column_is_needed(self):
        experiment = SimpleNamespace(channel=0, path='.', led_array=0)
        assert ExtinctionCoefficientsLinear(experiment, reference_property='sum_col_val').noise_property is None


class TestSmoothingVersusAveraging:
    def test_the_two_ways_of_averaging_cannot_be_combined(self):
        # one summarizes images into fewer rows and renumbers them, the other smooths and keeps the rows
        experiment = SimpleNamespace(channel=0, path='.', led_array=0)
        with pytest.raises(ValueError, match=r'average_images and num_moving_avg_images cannot be '
                                             r'combined'):
            ExtinctionCoefficientsLinear(experiment, average_images=True, num_moving_avg_images=3)

    def test_smoothing_alone_is_allowed(self):
        experiment = SimpleNamespace(channel=0, path='.', led_array=0)
        assert ExtinctionCoefficientsLinear(experiment, num_moving_avg_images=3).num_moving_avg_images == 3


class TestLedWeights:
    def test_equal_weights_reproduce_the_unweighted_solution(self):
        solver = make_solver()
        snr = np.full(DISTANCES.shape[0], 5 * SNR_FULL_WEIGHT)  # every LED well above the full weight
        np.testing.assert_allclose(solver.calc_coefficients_of_img(REL_INTENSITIES, snr),
                                   solver.calc_coefficients_of_img(REL_INTENSITIES), atol=1e-9)

    def test_the_unweighted_solution_recovers_the_coefficients(self):
        np.testing.assert_allclose(make_solver().calc_coefficients_of_img(REL_INTENSITIES), SIGMAS,
                                   atol=1e-3)

    def test_an_led_without_signal_does_not_influence_the_result(self):
        # its weight is 0, so whatever it reports must not reach the coefficients
        solver = make_solver()
        snr = np.array([0.0, 5 * SNR_FULL_WEIGHT, 5 * SNR_FULL_WEIGHT, 5 * SNR_FULL_WEIGHT])
        first, second = REL_INTENSITIES.copy(), REL_INTENSITIES.copy()
        first[0], second[0] = 0.9, 0.01
        np.testing.assert_allclose(solver.calc_coefficients_of_img(first, snr),
                                   solver.calc_coefficients_of_img(second, snr), atol=1e-9)

    def test_a_nan_signal_to_noise_ratio_becomes_weight_zero(self):
        solver = make_solver()
        snr = np.array([np.nan, 5 * SNR_FULL_WEIGHT, 5 * SNR_FULL_WEIGHT, 5 * SNR_FULL_WEIGHT])
        assert np.isfinite(solver.calc_coefficients_of_img(REL_INTENSITIES, snr)).all()

    def test_too_little_signal_in_the_whole_image_gives_nan(self):
        # the weights add up to less than one LED with full weight, so nothing can be computed
        solver = make_solver()
        snr = np.full(DISTANCES.shape[0], 0.1 * SNR_FULL_WEIGHT)
        assert np.isnan(solver.calc_coefficients_of_img(REL_INTENSITIES, snr)).all()


class TestLayersRestingOnNoise:
    SNR = np.array([0.0, 0.0, 0.0, 2 * SNR_FULL_WEIGHT])  # only the LED crossing every layer is measured

    def test_layers_below_the_limit_become_nan(self):
        # every layer carries the path weighted mean weight 0.5, from the one LED that crosses all of them
        assert np.isnan(make_solver(min_layer_weight=0.6).calc_coefficients_of_img(REL_INTENSITIES,
                                                                                   self.SNR)).all()

    def test_layers_above_the_limit_are_kept(self):
        assert np.isfinite(make_solver(min_layer_weight=0.4).calc_coefficients_of_img(REL_INTENSITIES,
                                                                                      self.SNR)).all()

    def test_without_the_limit_nothing_is_discarded(self):
        assert np.isfinite(make_solver().calc_coefficients_of_img(REL_INTENSITIES, self.SNR)).all()


class TestMovingAverage:
    NUM_IMAGES, NUM_LEDS = 5, 2

    def make_coefficients(self, values, noise=None):
        coefficients = ExtinctionCoefficientsLinear.__new__(ExtinctionCoefficientsLinear)
        coefficients.reference_property = 'ob_sum_col_val'
        coefficients.noise_property = 'ob_noise_sum_col_val' if noise is not None else None
        coefficients.num_moving_avg_images = 3
        index = pd.MultiIndex.from_product([range(1, self.NUM_IMAGES + 1), range(self.NUM_LEDS)],
                                           names=['img_id', 'led_id'])
        data = {'led_array_id': 0, 'ob_sum_col_val': values}
        if noise is not None:
            data['ob_noise_sum_col_val'] = noise
        coefficients.calculated_img_data = pd.DataFrame(data, index=index)
        return coefficients

    def test_every_image_keeps_its_own_row(self):
        coefficients = self.make_coefficients(np.arange(self.NUM_IMAGES * self.NUM_LEDS, dtype=float))
        coefficients.apply_moving_average()
        assert len(coefficients.calculated_img_data) == self.NUM_IMAGES * self.NUM_LEDS
        assert coefficients.calculated_img_data.index.get_level_values('img_id').nunique() == self.NUM_IMAGES

    def test_a_spike_is_smoothed_over_its_neighbours(self):
        values = np.zeros(self.NUM_IMAGES * self.NUM_LEDS)
        values[2 * self.NUM_LEDS] = 30.0  # led 0 of the middle image
        coefficients = self.make_coefficients(values)
        coefficients.apply_moving_average()
        smoothed = coefficients.calculated_img_data['ob_sum_col_val'].unstack('led_id')[0].to_numpy()
        np.testing.assert_allclose(smoothed, [0.0, 10.0, 10.0, 10.0, 0.0])

    def test_the_ends_average_over_fewer_images(self):
        coefficients = self.make_coefficients(np.ones(self.NUM_IMAGES * self.NUM_LEDS))
        coefficients.apply_moving_average()
        assert coefficients.calculated_img_data['ob_sum_col_val'].to_numpy() == pytest.approx(1.0)

    def test_the_noise_becomes_the_one_of_the_mean(self):
        coefficients = self.make_coefficients(np.ones(self.NUM_IMAGES * self.NUM_LEDS),
                                              noise=np.full(self.NUM_IMAGES * self.NUM_LEDS, 3.0))
        coefficients.apply_moving_average()
        noise = coefficients.calculated_img_data['ob_noise_sum_col_val'].unstack('led_id')[0].to_numpy()
        # three images in the middle, two at the ends: sqrt(n * sigma^2) / n = sigma / sqrt(n)
        np.testing.assert_allclose(noise, [3 / np.sqrt(2), 3 / np.sqrt(3), 3 / np.sqrt(3),
                                           3 / np.sqrt(3), 3 / np.sqrt(2)])
