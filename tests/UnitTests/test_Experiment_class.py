from unittest import TestCase, mock
from unittest.mock import MagicMock
from ledsa.analysis.Experiment import Experiment, Layers, Camera, LED
import numpy as np


class TestExperiment(TestCase):
    def setUp(self):
        self.layers = Layers(amount=10, bottom_border=0, top_border=5)
        self.camera = Camera(pos_x=1, pos_y=0, pos_z=1)
        self.experiment = Experiment(self.layers, 0, self.camera)
        self.experiment.leds = [LED(0, 0, 3, 0.5), LED(1, 1, 3, 1), LED(2, 2, 3, 1.5), LED(3, 3, 3, 2),
                                LED(4, 4, 3, 2.5)]
        self.experiment.led_number = 5


class HeightMethodTestCase(TestExperiment):
    def test_led_height_eq_camera_height(self):
        led_height = self.camera.pos_z
        th = self.experiment.calc_traversed_height_in_layer(led_height, 0, 2)
        self.assertAlmostEqual(th, 0)

    def test_led_and_camera_on_same_side(self):
        led_height = 0.5
        th = self.experiment.calc_traversed_height_in_layer(led_height, 2, 2.5)
        self.assertAlmostEqual(th, 0)

    def test_led_and_camera_on_different_sides(self):
        led_height = 3
        th = self.experiment.calc_traversed_height_in_layer(led_height, 1.5, 2.5)
        self.assertAlmostEqual(th, 1)

    def test_camera_inside_led_outside(self):
        led_height = 2
        th = self.experiment.calc_traversed_height_in_layer(led_height, 0.5, 1.5)
        self.assertAlmostEqual(th, 0.5)

    def test_led_inside_camera_outside(self):
        led_height = 2
        th = self.experiment.calc_traversed_height_in_layer(led_height, 1.5, 2.5)
        self.assertAlmostEqual(th, 0.5)


class DistConsistencyMethodTestCase(TestExperiment):
    def test_same_distances(self):
        dist_array = np.array([0, 1, 2, 1, 1, 0])
        self.experiment.camera.pos_x = 4
        self.experiment.camera.pos_y = 3
        self.experiment.camera.pos_z = 3.5
        self.assertTrue(self.experiment.distance_calculation_is_consistent(dist_array, self.experiment.leds[0]))

    def test_different_distances(self):
        dist_array = np.array([0, 1, 2, 1, 1, 0])
        self.assertFalse(self.experiment.distance_calculation_is_consistent(dist_array, self.experiment.leds[0]))


class CalcTraversedDistTestCase(TestExperiment):
    def test_calc_traversed_dist_per_layer(self):
        dists = self.experiment.calc_traversed_dist_per_layer(self.experiment.leds[4])
        self.assertTrue(self.experiment.distance_calculation_is_consistent(dists, self.experiment.leds[4]))

    def test_calc_traversed_dist_in_plane(self):
        dists = self.experiment.calc_traversed_dist_in_plane(LED(0, 5, 3, 1))
        self.assertAlmostEqual(5, dists[2])
        self.assertAlmostEqual(0, dists[1])
        self.assertAlmostEqual(0, dists[3])


class GetLEDPositions(TestExperiment):
    @mock.patch('numpy.loadtxt', return_value=np.array([[1, 2]]))
    def test_file_not_formatted_right(self, mock_loadtxt):
        self.assertRaises(IndexError, lambda: self.experiment.get_led_positions(np.array([0, 1, 2])))

    @mock.patch('numpy.loadtxt', return_value=np.array([[0, 0, 0, 1, 1, 1, 0, 0], [1, 0, 0, 1, 2, 3, 0, 0],
                                                        [2, 0, 0, 2, 2, 2, 0, 0]]))
    def test_file_exists_and_works(self, mock_loadtxt):
        led_x, led_y, led_z = self.experiment.get_led_positions(np.array([0, 1, 2]))
        self.assertListEqual(list(led_x), [1, 1, 2])
        self.assertListEqual(list(led_y), [1, 2, 2])
        self.assertListEqual(list(led_z), [1, 3, 2])


class SetLEDs(TestExperiment):
    def setUp(self):
        super().setUp()
        self.experiment.get_led_ids = MagicMock(return_value=[0, 1, 2, 3, 4])
        self.experiment.get_led_positions = MagicMock(return_value=[np.array([0, 1, 2, 3, 4]),
                                                                    np.array([3, 3, 3, 3, 3]),
                                                                    np.array([0.5, 1, 1.5, 2, 2.5])])

    def test_leds_set_right(self):
        self.experiment.set_leds()
        self.assertAlmostEqual(self.experiment.leds[0].pos_x, 0)
        self.assertAlmostEqual(self.experiment.leds[0].pos_y, 3)
        self.assertAlmostEqual(self.experiment.leds[0].pos_z, 0.5)
        self.assertAlmostEqual(self.experiment.leds[4].pos_x, 4)

    def test_led_number_set_right(self):
        self.assertEqual(self.experiment.led_number, len(self.experiment.leds))
