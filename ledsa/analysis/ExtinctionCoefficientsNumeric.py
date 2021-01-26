import numpy as np
from ledsa.analysis.Experiment import Experiment, Layers, Camera
from ledsa.analysis.ExtinctionCoefficients import ExtinctionCoefficients
from scipy.optimize import minimize


class ExtinctionCoefficientsNumeric(ExtinctionCoefficients):
    def __init__(self, experiment=Experiment(layers=Layers(20, 1.0, 3.35), camera=Camera(pos_x=4.4, pos_y=2, pos_z=2.3),
                                             led_array=3, channel=0),
                 reference_property='sum_col_val', num_ref_imgs=10):
        super().__init__(experiment, reference_property, num_ref_imgs)
        self.bounds = [(0, 10) for _ in range(self.experiment.layers.amount)]

        self.type = 'numeric'

    def calc_coefficients_of_img(self, rel_intensities: np.ndarray) -> np.ndarray:
        if len(self.coefficients_per_image_and_layer) == 0:
            kappa0 = np.zeros(self.experiment.layers.amount)
        else:
            kappa0 = self.coefficients_per_image_and_layer[-1]
        fit = minimize(self.cost_function, kappa0, args=rel_intensities,
                       method='TNC', bounds=tuple(self.bounds),
                       options={'maxiter': 200, 'gtol': 1e-5, 'disp': True})
        kappas = np.flip(fit.x)
        return kappas

    def calc_intensities(self, kappas: np.ndarray) -> np.ndarray:
        n_leds = self.experiment.led_number
        intensities = np.zeros(n_leds)
        for led in range(n_leds):
            intensity = 1.0
            for layer in range(len(self.distances_per_led_and_layer[led, :])):
                intensity = intensity * np.exp(-kappas[layer]*self.distances_per_led_and_layer[led, layer])
            intensities[led] = intensity
        return intensities

    def cost_function(self, kappas: np.ndarray, target: np.ndarray) -> float:
        intensities = self.calc_intensities(kappas)
        rmse = np.sqrt(np.sum((intensities - target) ** 2)) / len(intensities)
        curvature = np.sum(np.abs(kappas[0:-2] - 2 * kappas[1:-1] + kappas[2:])) * len(intensities) * 2 * 1e-6
        low_values = - np.sum(kappas) / len(kappas) * 6e-3
        return rmse + curvature + low_values