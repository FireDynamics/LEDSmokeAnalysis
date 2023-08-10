import shutil
import tempfile
from unittest import TestCase, mock
from unittest.mock import MagicMock

import numpy as np
import pandas as pd
from pathlib import Path
import os

from ledsa.analysis.ExtinctionCoefficientsNumeric import ExtinctionCoefficientsNumeric
from ledsa.analysis.ExtinctionCoefficients import multiindex_series_to_nparray
from ledsa.analysis.Experiment import Layers, Experiment, Camera, LED


class TestExtinctionCoefficientsNumeric(TestCase):
    def setUp(self):
        layers = Layers(5, 0, 5)
        self.ec = ExtinctionCoefficientsNumeric(num_ref_imgs=2, experiment=Experiment(layers=layers, led_array=1,
                                                                                      camera=Camera(2, 0, 2)))
        self.set_leds()
        self.set_dist_array()
        self.set_calculated_img_data()

    def set_leds(self):
        self.ec.experiment.led_number = 5
        for i in range(self.ec.experiment.led_number):
            led = LED(id=i, pos_x=i, pos_y=2, pos_z=i)
            self.ec.experiment.leds.append(led)

    def set_dist_array(self):
        self.ec.distances_per_led_and_layer = np.ones((self.ec.experiment.led_number, self.ec.experiment.layers.amount))

    def set_calculated_img_data(self):
        index_list = [[1, 2, 3], [0, 1, 2, 3, 4]]
        index = pd.MultiIndex.from_product(index_list, names=['img_id', 'led_id'])
        self.ec.calculated_img_data = pd.DataFrame(np.ones([15, 2]), index=index,
                                                   columns=['line', self.ec.reference_property])


class ComputeReferenceIntensitiesTestCase(TestExtinctionCoefficientsNumeric):
    def test_num_of_ref_intensities_are_right(self):
        self.ec.calc_and_set_ref_intensities()
        self.assertEqual(self.ec.experiment.led_number, len(self.ec.ref_intensities))

    def test_ref_intensities_are_right(self):
        self.ec.calc_and_set_ref_intensities()
        self.assertAlmostEqual(1, self.ec.ref_intensities[0])


class ComputeIntensitiesTestCase(TestExtinctionCoefficientsNumeric):
    def test_num_of_intensities_are_right(self):
        kappas = np.zeros(self.ec.experiment.layers.amount)
        intensities = self.ec.calc_intensities(kappas)
        self.assertEqual(len(intensities), self.ec.experiment.layers.amount)

    def test_intensities_are_right(self):
        kappas = np.zeros(self.ec.experiment.layers.amount)
        intensities = self.ec.calc_intensities(kappas)
        self.assertEqual(intensities[0], 1)


class SaveTestCase(TestExtinctionCoefficientsNumeric):
    def setUp(self):
        super().setUp()
        self.temp_dir = tempfile.mkdtemp()
        self.ec.experiment.path = Path(self.temp_dir)
        self.path = self.ec.experiment.path / 'analysis' / 'AbsorptionCoefficients'
        self.ec.save()

    def tearDown(self):
        shutil.rmtree(self.temp_dir)

    def test_path_was_created(self):
        self.assertTrue(self.path.exists())

    def test_file_was_created(self):
        self.assertEqual(1, len(os.listdir(self.path)))


class CostFunctionTestCase(TestExtinctionCoefficientsNumeric):
    def setUp(self):
        super().setUp()
        intensities = np.ones(self.ec.experiment.led_number)
        intensities /= 2
        self.ec.calc_intensities = MagicMock(return_value=intensities)

    def test_returns_pos_value(self):
        kappas = np.zeros(self.ec.experiment.layers.amount)
        target = np.ones(self.ec.experiment.led_number)
        cost = self.ec.cost_function(kappas, target)
        self.assertTrue(cost >= 0)

    def test_target_same_as_intensities_returns_cost_smaller_equal_zero(self):
        kappas = np.zeros(self.ec.experiment.layers.amount)
        target = np.ones(self.ec.experiment.led_number) / 2
        cost = self.ec.cost_function(kappas, target)
        self.assertTrue(cost <= 0)


class CalcDistArrayTestCase(TestExtinctionCoefficientsNumeric):
    def test_amount_of_distances_equals_amount_of_layers(self):
        self.ec.experiment.calc_traversed_dist_per_layer = MagicMock(return_value=np.ones(5))
        dist = self.ec.calc_distance_array()
        self.assertEqual(self.ec.experiment.layers.amount, len(dist))


class CoefficientCalculationTestCase(TestExtinctionCoefficientsNumeric):
    class MockMinimizeResult:
        def __init__(self):
            self.x = np.ones(5)

    @mock.patch('ledsa.analysis.ExtinctionCoefficientsNumeric.minimize', return_value=MockMinimizeResult())
    def test_calculated_coefficient_matrix_has_right_dim(self, mocked_minimize):
        self.ec.calc_and_set_coefficients()
        self.assertEqual(len(self.ec.calculated_img_data.groupby(level='img_id')),
                         len(self.ec.coefficients_per_image_and_layer), msg='Test Image Dimension')
        self.assertEqual(self.ec.experiment.layers.amount, len(self.ec.coefficients_per_image_and_layer[0]),
                         msg='Test Layer Dim')


class CoefficientCalculationMPTestCase(CoefficientCalculationTestCase):
    class MockMinimizeResult:
        def __init__(self):
            self.x = np.ones(5)

    @mock.patch('ledsa.analysis.ExtinctionCoefficientsNumeric.minimize', return_value=MockMinimizeResult())
    def test_calculated_coefficient_matrix_has_right_dim(self, mocked_minimize):
        self.ec.calc_and_set_coefficients_mp()
        self.assertEqual(len(self.ec.calculated_img_data.groupby(level='img_id')),
                         len(self.ec.coefficients_per_image_and_layer), msg='Test Image Dimension')
        self.assertEqual(self.ec.experiment.layers.amount, len(self.ec.coefficients_per_image_and_layer[0]),
                         msg='Test Layer Dim')


class MultiSeriesArrayTransformTestCase(TestExtinctionCoefficientsNumeric):
    def test_array_has_right_dim(self):
        array = multiindex_series_to_nparray(self.ec.calculated_img_data[self.ec.reference_property])
        self.assertEqual(self.ec.calculated_img_data.index.levshape, array.shape)
