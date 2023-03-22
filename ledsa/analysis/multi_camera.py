import matplotlib.pyplot as plt
import numpy as np
from scipy.optimize import minimize

class ExtinctionCoefficientsNumeric():
    def __init__(self):
        self.coefficients_per_image_and_layer = []
        self.weighting_preference = -6e-3
        self.weighting_curvature =  1e-6
        self.num_iterations = 2000
        self.type = 'numeric'
        self.layers_amount = 20
        self.num_of_leds = 100
        self.bounds = [(0, 10) for _ in range(self.layers_amount)]
        self.distances_per_cam_and_led_and_layer = None


    def calc_coefficients_of_img(self, rel_intensities: np.ndarray) -> np.ndarray:
        if len(self.coefficients_per_image_and_layer) == 0:
            kappa0 = np.zeros(self.layers_amount)
        else:
            kappa0 = self.coefficients_per_image_and_layer[-1]
        fit = minimize(self.cost_function, kappa0, args=rel_intensities,
                       method='TNC', bounds=tuple(self.bounds),
                       options={'maxiter': self.num_iterations, 'gtol': 1e-5, 'disp': True})
        kappas = np.flip(fit.x)
        return kappas

    def calc_intensities(self, kappas: np.ndarray) -> np.ndarray:
        n_cams = 2
        n_leds = self.num_of_leds
        intensities = np.zeros(n_leds*n_cams)
        i = 0
        for led in range(n_leds):
            intensity = 1.0
            for cam in range(n_cams):
                for layer in range(len(self.distances_per_cam_and_led_and_layer[cam][led, :])):
                    intensity = intensity * np.exp(-kappas[layer]*self.distances_per_cam_and_led_and_layer[cam][led, layer])
                intensities[i] = intensity
                i+=1
        a = 1
        return intensities

    def cost_function(self, kappas: np.ndarray, target: np.ndarray) -> float:
        intensities = self.calc_intensities(kappas)
        rmse = np.sqrt(np.sum((intensities - target) ** 2)) / len(intensities)
        curvature = np.sum(np.abs(kappas[0:-2] - 2 * kappas[1:-1] + kappas[2:])) * len(intensities) * 2 * self.weighting_curvature # TODO: Factor 2 in weighting factor?
        preference = np.sum(kappas) / len(kappas) * self.weighting_preference
        return rmse + curvature + preference

ext = ExtinctionCoefficientsNumeric()
img = 3
dist_1 = np.loadtxt('/Users/kristianboerger/Documents/temp/1_0/cam_0_distances_per_led_and_layer.txt')
dist_2 = np.loadtxt('/Users/kristianboerger/Documents/temp/2_5/cam_0_distances_per_led_and_layer.txt')
rel_int_1 = np.loadtxt(f'/Users/kristianboerger/Documents/temp/1_0/cam_0_rel_intensities_{img}.txt')
rel_int_2 = np.loadtxt(f'/Users/kristianboerger/Documents/temp/2_5/cam_0_rel_intensities_{img}.txt')

# all_dist =  np.array([dist_1, dist_2])
# all_dist =  np.array([dist_1])
# all_dist =  np.array([dist_2])
all_dist = np.array([dist_1, dist_1])

ext.distances_per_cam_and_led_and_layer = all_dist
# all_rel_int = np.concatenate((rel_int_1, rel_int_2))
# all_rel_int = rel_int_1
# all_rel_int = rel_int_2
all_rel_int = np.concatenate((rel_int_1, rel_int_1))

kappas = ext.calc_coefficients_of_img(all_rel_int)
print(kappas)
plt.plot(kappas, range(len(kappas)))
plt.xlim(0, 0.8)
plt.grid(linestyle='--', alpha=0.5)
plt.show()