import numpy as np
from ledsa.analysis.Experiment import Experiment, LED


class ExtinctionCoefficientsAlgebraic:
    def __init__(self, experiment: Experiment, rel_intensities: np.ndarray):
        self.experiment = experiment
        mean_led_positions = self.calc_mean_led_positions_per_layer()
        mean_rel_intensity = self.calc_mean_relative_intensities_per_layer(rel_intensities)
        mean_dist = self.calc_mean_dist_per_dummy_led_and_layer(mean_led_positions)
        self.coefficients = self.calc_kappas(mean_dist, mean_rel_intensity, rel_intensities)

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

    def calc_kappas(self, mean_dist_per_dummy_led_and_layer, mean_rel_intensity, rel_intensities):
        kappas = np.zeros(self.experiment.layers.amount)
        camera_layer = 0
        for layer in range(self.experiment.layers.amount):
            if np.sum(mean_dist_per_dummy_led_and_layer[layer] > 0) == 1:
                kappas[layer] = self.calc_kappa(kappas, layer,
                                                mean_dist_per_dummy_led_and_layer[layer],
                                                rel_intensities[layer])
                camera_layer = layer
                break
        for upper_layer in range(camera_layer + 1, self.experiment.layers.amount):
            kappas[upper_layer] = self.calc_kappa(kappas, upper_layer,
                                                  mean_dist_per_dummy_led_and_layer[upper_layer],
                                                  mean_rel_intensity[upper_layer])
        for bottom_layer in range(camera_layer - 1, -1, -1):
            kappas[bottom_layer] = self.calc_kappa(kappas, bottom_layer,
                                                   mean_dist_per_dummy_led_and_layer[bottom_layer],
                                                   mean_rel_intensity[bottom_layer])
        return kappas

    def calc_kappa(self, kappas: np.ndarray, layer: int, dist_per_layer: np.ndarray,
                   rel_intensity: float) -> np.ndarray:
        if dist_per_layer[layer] == 0:
            return np.nan
        kappa_new = (-np.log(rel_intensity) - sum(kappas * dist_per_layer)) / dist_per_layer[layer]
        return kappa_new
