import shutil
import tempfile
from unittest import TestCase, mock
from unittest.mock import MagicMock

import numpy as np
import pandas as pd
from pathlib import Path
import os

from ledsa.analysis.AbsorptionCoefficients import AbsorptionCoefficients
from ledsa.analysis.Experiment import Layers, Experiment, Camera, LED


class TestAbsorptionCoefficients(TestCase):
    def setUp(self):
        layers = Layers(5, 0, 5)
        self.ac = AbsorptionCoefficients(num_ref_imgs=2, experiment=Experiment(layers=layers, led_array=1,
                                                                               camera=Camera(2, 0, 2)))
        self.set_leds()
        self.set_dist_array()
        self.set_calculated_img_data()

    def set_leds(self):
        self.ac.experiment.led_number = 5
        for i in range(self.ac.experiment.led_number):
            led = LED(id=i, pos_x=i, pos_y=2, pos_z=i)
            self.ac.experiment.leds.append(led)

    def set_dist_array(self):
        for i in range(self.ac.experiment.led_number):
            self.ac.distances_per_led_and_layer.append(np.ones(self.ac.experiment.layers.amount))

    def set_calculated_img_data(self):
        index_list = [[1, 2, 3], [0, 1, 2, 3, 4]]
        index = pd.MultiIndex.from_product(index_list, names=['img_id', 'led_id'])
        self.ac.calculated_img_data = pd.DataFrame(np.ones([15, 2]), index=index,
                                                   columns=['line', self.ac.reference_property])


class ComputeReferenceIntensitiesTestCase(TestAbsorptionCoefficients):
    def test_num_of_ref_intensities_are_right(self):
        self.ac.calc_and_set_ref_intensities()
        self.assertEqual(self.ac.experiment.led_number, len(self.ac.ref_intensities))

    def test_ref_intensities_are_right(self):
        self.ac.calc_and_set_ref_intensities()
        self.assertAlmostEqual(1, self.ac.ref_intensities[0])


class ComputeIntensitiesTestCase(TestAbsorptionCoefficients):
    def test_num_of_intensities_are_right(self):
        kappas = np.zeros(self.ac.experiment.layers.amount)
        intensities = self.ac.calc_intensities(kappas)
        self.assertEqual(len(intensities), self.ac.experiment.layers.amount)

    def test_intensities_are_right(self):
        kappas = np.zeros(self.ac.experiment.layers.amount)
        intensities = self.ac.calc_intensities(kappas)
        self.assertEqual(intensities[0], 1)


class SaveTestCase(TestAbsorptionCoefficients):
    def setUp(self):
        super().setUp()
        self.temp_dir = tempfile.mkdtemp()
        self.ac.experiment.path = Path(self.temp_dir)
        self.path = self.ac.experiment.path / 'analysis' / 'AbsorptionCoefficients'
        self.ac.save()

    def tearDown(self):
        shutil.rmtree(self.temp_dir)

    def test_path_was_created(self):
        self.assertTrue(self.path.exists())

    def test_file_was_created(self):
        self.assertEqual(1, len(os.listdir(self.path)))


class CostFunctionTestCase(TestAbsorptionCoefficients):
    def setUp(self):
        super().setUp()
        intensities = np.ones(self.ac.experiment.led_number)
        intensities /= 2
        self.ac.calc_intensities = MagicMock(return_value=intensities)

    def test_returns_pos_value(self):
        kappas = np.zeros(self.ac.experiment.layers.amount)
        target = np.ones(self.ac.experiment.led_number)
        cost = self.ac.cost_function(kappas, target)
        self.assertTrue(cost >= 0)

    def test_target_same_as_intensities_returns_cost_smaller_equal_zero(self):
        kappas = np.zeros(self.ac.experiment.layers.amount)
        target = np.ones(self.ac.experiment.led_number) / 2
        cost = self.ac.cost_function(kappas, target)
        self.assertTrue(cost <= 0)


class CalcDistArrayTestCase(TestAbsorptionCoefficients):
    def test_amount_of_distances_equals_amount_of_layers(self):
        self.ac.experiment.calc_traversed_dist_per_layer = MagicMock(return_value=np.ones(5))
        dist = self.ac.calc_distance_array()
        self.assertEqual(self.ac.experiment.layers.amount, len(dist))


class CoefficientCalculationTestCase(TestAbsorptionCoefficients):
    class MockMinimizeResult:
        def __init__(self):
            self.x = np.ones(5)

    @mock.patch('ledsa.analysis.AbsorptionCoefficients.minimize', return_value=MockMinimizeResult())
    def test_calculated_coefficient_matrix_has_right_dim(self, mocked_minimize):
        self.ac.calc_and_set_coefficients()
        self.assertEqual(len(self.ac.calculated_img_data.groupby(level='img_id')),
                         len(self.ac.coefficients_per_image_and_layer), msg='Test Image Dimension')
        self.assertEqual(self.ac.experiment.layers.amount, len(self.ac.coefficients_per_image_and_layer[0]),
                         msg='Test Layer Dim')
