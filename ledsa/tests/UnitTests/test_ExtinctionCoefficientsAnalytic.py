from unittest import TestCase, mock
from ledsa.analysis.ExtinctionCoefficientsAnalytic import ExtinctionCoefficientsAnalytic
from ledsa.analysis.Experiment import Layers, Experiment, Camera, LED
import numpy as np


class TestExtinctionCoefficientsAnalytic(TestCase):
    def setUp(self) -> None:
        def set_leds(exp: Experiment) -> None:
            exp.led_number = 5
            for i in range(exp.led_number):
                led = LED(id=i, pos_x=i+0.5, pos_y=2, pos_z=i+0.5)
                exp.leds.append(led)

        layers = Layers(5, 0, 5)
        experiment = Experiment(layers=layers, led_array=1, camera=Camera(2, 0, 2.5))
        set_leds(experiment)
        self.rel_intensities = np.array([.1, .2, .5, .2, .1])
        self.ec = ExtinctionCoefficientsAnalytic(experiment=experiment)
        self.kappas = np.array([0, 0, 0.1, 0, 0])
        self.dist_per_led_and_layer = np.array([[0, 1, 1, 1, 1], [2, 2, 2, 2, 2], [3, 3, 3, 3, 3], [4, 4, 4, 4, 4],
                                               [5, 5, 5, 5, 5]])


class SingleCoefficientTestCase(TestExtinctionCoefficientsAnalytic):
    def test_calculation_is_right(self):
        rel_intensity = 0.5
        kappa = self.ec.calc_kappa(self.kappas, 1, self.dist_per_led_and_layer[1], rel_intensity)
        calculated_intensity = np.exp(-kappa*self.dist_per_led_and_layer[1, 1] -
                                      self.kappas[2]*self.dist_per_led_and_layer[1, 2])
        self.assertAlmostEqual(0.5, calculated_intensity)

    def test_div_by_0_returns_nan(self):
        kappa = self.ec.calc_kappa(self.kappas, 0, self.dist_per_led_and_layer[0], 1)
        self.assertTrue(np.isnan(kappa))


class AllCoefficientsTestCase(TestExtinctionCoefficientsAnalytic):
    def setUp(self) -> None:
        super().setUp()
        mean_dist_per_led_and_layer = np.array([[1, 1, 1, 0, 0], [0, 1, 1, 0, 0], [0, 0, 1, 0, 0],
                                                [0, 0, 1, 1, 0], [0, 0, 1, 1, 1]])
        self.ec.calc_mean_dist_per_dummy_led_and_layer = mock.MagicMock(return_value=mean_dist_per_led_and_layer)
        self.kappas = self.ec.calc_coefficients_of_img(self.rel_intensities)
        self.calculated_intensities = np.exp(np.dot(-self.kappas, mean_dist_per_led_and_layer.T))

    def test_all_coefficients_are_calculated(self):
        for i in range(5):
            self.assertNotAlmostEqual(0, self.kappas[i])

    def test_coefficients_are_right(self):
        for i in range(5):
            self.assertAlmostEqual(self.rel_intensities[i], self.calculated_intensities[i], msg=f'layer: {i}')


class TestMeanCalculations(TestExtinctionCoefficientsAnalytic):
    def setUp(self) -> None:
        super().setUp()
        self.ec.experiment.leds[0].pos_x = 0
        self.ec.experiment.leds[0].pos_y = 0.5
        self.ec.experiment.leds[0].pos_z = 0.7
        self.ec.experiment.leds[3].pos_x = 0.5
        self.ec.experiment.leds[3].pos_y = 0.5
        self.ec.experiment.leds[3].pos_z = 0.5


class MeanPositionCalculationsTestCase(TestMeanCalculations):
    def test_mean_position_calculation(self):
        mean_positions = self.ec.calc_mean_led_positions_per_layer()
        self.assertAlmostEqual(mean_positions[0, 0], 0.25)
        self.assertAlmostEqual(mean_positions[0, 1], 0.5)
        self.assertAlmostEqual(mean_positions[0, 2], 0.6)

    def test_layers_without_leds_have_nan_positions(self):
        mean_position = self.ec.calc_mean_led_positions_per_layer()
        self.assertTrue(np.isnan(mean_position[3, 0]))


class MeanIntensitiesCalculationsTestCase(TestMeanCalculations):
    def test_mean_intensity_calculation(self):
        intensities = self.ec.calc_mean_relative_intensities_per_layer(self.rel_intensities)
        self.assertAlmostEqual(0.15, intensities[0])

    def test_layers_without_leds_have_nan_intensities(self):
        intensities = self.ec.calc_mean_relative_intensities_per_layer(self.rel_intensities)
        self.assertTrue(np.isnan(intensities[3]))


class MeanDistanceCalculationTestCase(TestMeanCalculations):
    def setUp(self) -> None:
        super().setUp()
        positions = self.ec.calc_mean_led_positions_per_layer()
        self.dists = self.ec.calc_mean_dist_per_dummy_led_and_layer(positions)

    def test_mean_distances_are_set(self):
        dists_are_bigger_0 = np.sum(self.dists, axis=1) >= 0
        dists_are_not_nan = np.isnan(np.sum(self.dists, axis=1))
        self.assertTrue(np.array_equal(np.logical_or(dists_are_bigger_0, dists_are_not_nan),
                                       np.ones(self.ec.experiment.layers.amount)))

    def test_camera_layer_calculation(self):
        led = self.ec.experiment.leds[2]
        cam = self.ec.experiment.camera
        dist_cam_layer_led = np.sqrt((led.pos_x-cam.pos_x)**2 + (led.pos_y-cam.pos_y)**2 + (led.pos_z-cam.pos_z)**2)
        self.assertAlmostEqual(dist_cam_layer_led, self.dists[2, 2])

    def test_outer_layer_calculation(self):
        led = self.ec.experiment.leds[1]
        cam = self.ec.experiment.camera
        dist_cam_led = np.sqrt((led.pos_x - cam.pos_x) ** 2 + (led.pos_y - cam.pos_y) ** 2 +
                               (led.pos_z - cam.pos_z) ** 2)
        self.assertAlmostEqual(dist_cam_led / 2, self.dists[1, 2])

