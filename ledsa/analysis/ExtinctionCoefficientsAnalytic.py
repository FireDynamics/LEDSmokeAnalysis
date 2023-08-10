import numpy as np

from ledsa.analysis.Experiment import Experiment, Layers, Camera, LED
from ledsa.analysis.ExtinctionCoefficients import ExtinctionCoefficients


def calc_kappa(kappas: np.ndarray, layer: int, dist_per_layer: np.ndarray,
               rel_intensity: float) -> np.ndarray:
    if dist_per_layer[layer] == 0:
        return np.nan
    kappa_new = (-np.log(rel_intensity) - sum(kappas * dist_per_layer)) / dist_per_layer[layer]
    return kappa_new


class ExtinctionCoefficientsAnalytic(ExtinctionCoefficients):
    def __init__(self, experiment=Experiment(layers=Layers(10, 1.0, 3.35), camera=Camera(pos_x=4.4, pos_y=2, pos_z=2.3),
                                             led_array=3, channel=0),
                 reference_property='sum_col_val', num_ref_imgs=10):
        super().__init__(experiment, reference_property, num_ref_imgs)
        self.type = 'analytic'

    def calc_mean_dist_per_dummy_led_and_layer(self, mean_led_positions_per_layer: np.ndarray) -> np.ndarray:
        mean_dist_per_dummy_led_and_layer = np.zeros((self.experiment.layers.amount, self.experiment.layers.amount))
        for layer in range(self.experiment.layers.amount):
            led = LED(layer, mean_led_positions_per_layer[layer, 0], mean_led_positions_per_layer[layer, 1],
                      mean_led_positions_per_layer[layer, 2])
            mean_dist_per_dummy_led_and_layer[layer] = self.experiment.calc_traversed_dist_per_layer(led)
        return mean_dist_per_dummy_led_and_layer

    def calc_mean_relative_intensities_per_layer(self, rel_intensities: np.ndarray) -> np.ndarray:
        mean_rel_intensity_per_layer = np.zeros(self.experiment.layers.amount)
        for layer in range(self.experiment.layers.amount):
            led_counter = 0
            leds_in_layer = 0
            for led in self.experiment.leds:
                led_counter += 1
                if led in self.experiment.layers[layer]:
                    leds_in_layer += 1
                    mean_rel_intensity_per_layer[layer] += rel_intensities[led_counter - 1]
            if leds_in_layer > 0:
                mean_rel_intensity_per_layer[layer] = mean_rel_intensity_per_layer[layer] / leds_in_layer
            else:
                mean_rel_intensity_per_layer[layer] = np.nan
        return mean_rel_intensity_per_layer

    def calc_mean_led_positions_per_layer(self):
        mean_led_pos_per_layer = np.zeros((self.experiment.layers.amount, 3))
        for layer in range(self.experiment.layers.amount):
            led_counter = 0
            for led in self.experiment.leds:
                if led in self.experiment.layers[layer]:
                    led_counter += 1
                    mean_led_pos_per_layer[layer] += [led.pos_x, led.pos_y, led.pos_z]
            if led_counter > 0:
                mean_led_pos_per_layer[layer] = mean_led_pos_per_layer[layer] / led_counter
            else:
                mean_led_pos_per_layer[layer] = np.array([np.nan, np.nan, np.nan])
        return mean_led_pos_per_layer

    def calc_coefficients_of_img(self, rel_intensities: np.ndarray) -> np.ndarray:
        mean_led_positions = self.calc_mean_led_positions_per_layer()
        mean_dist = self.calc_mean_dist_per_dummy_led_and_layer(mean_led_positions)
        mean_rel_intensity = self.calc_mean_relative_intensities_per_layer(rel_intensities)
        camera_layer = self.find_camera_layer(mean_dist)
        kappas = np.zeros(self.experiment.layers.amount)

        for upper_layer in range(camera_layer, self.experiment.layers.amount):
            kappas[upper_layer] = calc_kappa(kappas, upper_layer,
                                                  mean_dist[upper_layer],
                                                  mean_rel_intensity[upper_layer])
        for bottom_layer in range(camera_layer - 1, -1, -1):
            kappas[bottom_layer] = calc_kappa(kappas, bottom_layer,
                                                   mean_dist[bottom_layer],
                                                   mean_rel_intensity[bottom_layer])
        return kappas

    def find_camera_layer(self, mean_dist_per_led_and_layer: np.ndarray) -> int:
        for layer in range(self.experiment.layers.amount):
            if np.sum(mean_dist_per_led_and_layer[layer] > 0) == 1:
                return layer
