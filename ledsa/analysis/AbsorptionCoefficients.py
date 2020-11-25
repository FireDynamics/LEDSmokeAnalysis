from typing import List
import numpy as np
import pandas as pd
from scipy.optimize import minimize
from ledsa.analysis.Experiment import Experiment, Layers, Camera
from ledsa.analysis.calculations import read_hdf


class AbsorptionCoefficients:
    def __init__(self, experiment=Experiment(layers=Layers(20, 1.0, 3.35), camera=Camera(pos_x=4.4, pos_y=2, pos_z=2.3),
                                             led_array=3, channel=0),
                 reference_property='sum_col_val', num_ref_imgs=10):
        self.coefficients_per_image_and_layer = []
        self.experiment = experiment
        self.reference_property = reference_property
        self.num_ref_imgs = num_ref_imgs

        self.calculated_img_data = pd.DataFrame()
        self.distances_per_led_and_layer = []
        self.ref_intensities = np.array([])

    def calc_and_set_coefficients(self) -> None:
        self.set_all_member_variables()
        kappa0 = np.zeros(self.experiment.layers.amount)
        bounds = [(0, 10) for _ in range(self.experiment.layers.amount)]
        for img_id, single_img_data in self.calculated_img_data.groupby(level=0):
            single_img_array = single_img_data[self.reference_property].to_numpy()
            rel_intensities = single_img_array / self.ref_intensities

            res = minimize(self.cost_function, kappa0, args=rel_intensities,
                           method='TNC', bounds=tuple(bounds),
                           options={'maxiter': 200, 'gtol': 1e-5, 'disp': True})

            kappa0[:] = res.x
            self.coefficients_per_image_and_layer.append(res.x)
        return None

    def set_all_member_variables(self):
        if len(self.distances_per_led_and_layer) == 0:
            self.distances_per_led_and_layer = self.calc_distance_array()
        if self.calculated_img_data.empty:
            self.load_img_data()
        if self.ref_intensities.shape[0] == 0:
            self.calc_and_set_ref_intensities()

    def save(self) -> None:
        path = self.experiment.path / 'analysis' / 'AbsorptionCoefficients'
        if not path.exists():
            path.mkdir(parents=True)
        path = path / f'absorption_coefficients_channel_{self.experiment.channel}.csv'
        np.savetxt(path, self.coefficients_per_image_and_layer, delimiter=',')

    def load_img_data(self) -> None:
        img_data = read_hdf(self.experiment.channel, path=self.experiment.path)
        img_data_cropped = img_data['img_id', 'led_id', 'line', self.reference_property]
        self.calculated_img_data = img_data_cropped[img_data_cropped['line'] == self.experiment.led_array]

    def calc_distance_array(self) -> List[np.ndarray]:
        distances = []
        for led in self.experiment.leds:
            d = self.experiment.calc_traversed_dist_per_layer(led)
            distances.append(d)
        return distances

    def calc_intensities(self, kappas: np.ndarray) -> np.ndarray:
        n_leds = self.experiment.led_number
        intensities = np.zeros(n_leds)
        for led in range(n_leds):
            intensity = 1.0
            for layer in range(len(self.distances_per_led_and_layer[led])):
                intensity = intensity * np.exp(-kappas[layer]*self.distances_per_led_and_layer[led][layer])
            intensities[led] = intensity
        return intensities

    def cost_function(self, kappas: np.ndarray, target: np.ndarray) -> float:
        intensities = self.calc_intensities(kappas)
        rmse = np.sqrt(np.sum((intensities - target) ** 2)) / len(intensities)
        curvature = np.sum(np.abs(kappas[0:-2] - 2 * kappas[1:-1] + kappas[2:])) * len(intensities) * 2 * 1e-6
        low_values = - np.sum(kappas) / len(kappas) * 6e-3
        return rmse + curvature + low_values

    def calc_and_set_ref_intensities(self) -> None:
        ref_img_data = self.calculated_img_data.query(f'img_id <= {self.num_ref_imgs}')
        ref_intensities = ref_img_data.mean(0, level='led_id')
        self.ref_intensities = ref_intensities[self.reference_property].to_numpy()
