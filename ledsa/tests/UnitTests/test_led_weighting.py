from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest

from ledsa.analysis.ExtinctionCoefficients import DEFAULT_SNR_FULL_WEIGHT
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


class TestWhenTheWeightingIsUsed:
    """The weighting is on by default and switched off wherever the data cannot carry it."""

    @staticmethod
    def experiment():
        return SimpleNamespace(channel=0, path='.', led_array=0)

    def test_the_default_weights_the_unclipped_sum_by_its_noise(self):
        coefficients = ExtinctionCoefficientsLinear(self.experiment(), reference_property='sum_col_val')
        assert coefficients.snr_full_weight == DEFAULT_SNR_FULL_WEIGHT
        assert coefficients.noise_property == 'noise_sum_col_val'

    def test_a_property_without_a_noise_column_is_weighted_equally(self, capsys):
        # the default must not stop an analysis of a property that cannot carry the weighting
        coefficients = ExtinctionCoefficientsLinear(self.experiment(),
                                                    reference_property='sidebg_sum_col_val')
        assert coefficients.noise_property is None
        assert 'weighted equally' in capsys.readouterr().out

    def test_the_option_can_be_switched_off(self):
        coefficients = ExtinctionCoefficientsLinear(self.experiment(), reference_property='sum_col_val',
                                                    snr_full_weight=None)
        assert coefficients.noise_property is None

    def test_the_header_says_whether_the_weighting_was_used(self):
        weighted = ExtinctionCoefficientsLinear(self.experiment(), reference_property='sum_col_val')
        unweighted = ExtinctionCoefficientsLinear(self.experiment(), reference_property='sum_col_val',
                                                  snr_full_weight=None)
        assert 'Noise_property: noise_sum_col_val' in str(weighted)
        assert 'Noise_property: None' in str(unweighted)
        assert 'Min_layer_weight: None' in str(unweighted)


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

    def test_an_unknown_noise_keeps_the_full_weight(self):
        # a missing noise estimate is no reason to drop a measured intensity; only a known small SNR is
        solver = make_solver()
        snr = np.array([np.nan, 5 * SNR_FULL_WEIGHT, 5 * SNR_FULL_WEIGHT, 5 * SNR_FULL_WEIGHT])
        np.testing.assert_allclose(solver.calc_coefficients_of_img(REL_INTENSITIES, snr),
                                   solver.calc_coefficients_of_img(REL_INTENSITIES), atol=1e-9)

    def test_an_led_with_a_known_small_ratio_is_downweighted(self):
        solver = make_solver()
        weak = np.array([0.1 * SNR_FULL_WEIGHT] + [5 * SNR_FULL_WEIGHT] * 3)
        corrupted = REL_INTENSITIES.copy()
        corrupted[0] = 0.5
        # the corrupted LED moves the result less than it would with the full weight
        weighted = solver.calc_coefficients_of_img(corrupted, weak)
        unweighted = solver.calc_coefficients_of_img(corrupted)
        clean = solver.calc_coefficients_of_img(REL_INTENSITIES)
        assert np.abs(weighted - clean).max() < np.abs(unweighted - clean).max()

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


class TestTheWeightingFallsBackAtLoadTime:
    """Data without a usable noise column switch the weighting off instead of stopping the analysis."""

    @pytest.fixture
    def coefficients_for(self, monkeypatch):
        import ledsa.analysis.ExtinctionCoefficients as extinction_coefficients

        def build(noise):
            index = pd.MultiIndex.from_product([[1, 2], [0, 1]], names=['img_id', 'led_id'])
            data = {'led_array_id': 0, 'sum_col_val': 100.0}
            if noise is not None:
                data['noise_sum_col_val'] = noise
            img_data = pd.DataFrame(data, index=index)
            monkeypatch.setattr(extinction_coefficients, 'read_hdf', lambda channel, path: img_data)
            coefficients = ExtinctionCoefficientsLinear.__new__(ExtinctionCoefficientsLinear)
            coefficients.experiment = SimpleNamespace(channel=0, path='.', led_array=0)
            coefficients.average_images = False
            coefficients.reference_property = 'sum_col_val'
            coefficients.noise_property = 'noise_sum_col_val'
            coefficients.ref_img_ids_of_loaded_data = None
            coefficients.temperature_correction = None
            coefficients.temperature_corrected = False
            return coefficients
        return build

    def test_a_binary_without_the_noise_column_is_weighted_equally(self, coefficients_for, capsys):
        coefficients = coefficients_for(None)
        coefficients.load_img_data()
        assert coefficients.noise_property is None
        out = capsys.readouterr().out
        assert 'weighted equally' in out and 'Run step 3 again' in out

    def test_a_noise_column_of_only_nan_is_weighted_equally(self, coefficients_for, capsys):
        # JPG and PNG images and sensors without an optical black margin carry no read noise
        coefficients = coefficients_for(float('nan'))
        coefficients.load_img_data()
        assert coefficients.noise_property is None
        assert 'carry no read noise' in capsys.readouterr().out

    def test_a_usable_noise_column_is_kept(self, coefficients_for):
        coefficients = coefficients_for(2.0)
        coefficients.load_img_data()
        assert coefficients.noise_property == 'noise_sum_col_val'
        assert 'noise_sum_col_val' in coefficients.calculated_img_data.columns
