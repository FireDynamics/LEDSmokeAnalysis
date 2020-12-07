from abc import ABC, abstractmethod
import numpy as np
import pandas as pd
from ledsa.analysis.Experiment import Experiment, Layers, Camera
from ledsa.analysis.calculations import read_hdf

class ExtinctionCoefficients(ABC):
    def __init__(self, experiment=Experiment(layers=Layers(10, 1.0, 3.35), camera=Camera(pos_x=4.4, pos_y=2, pos_z=2.3),
                                             led_array=3, channel=0),
                 reference_property='sum_col_val', num_ref_imgs=10):
        self.coefficients_per_image_and_layer = []
        self.experiment = experiment
        self.reference_property = reference_property
        self.num_ref_imgs = num_ref_imgs

        self.calculated_img_data = pd.DataFrame()
        self.distances_per_led_and_layer = np.array([])
        self.ref_intensities = np.array([])

        self.type = None

    def calc_and_set_coefficients(self) -> None:
        self.set_all_member_variables()
        for img_id, single_img_data in self.calculated_img_data.groupby(level=0):
            single_img_array = single_img_data[self.reference_property].to_numpy()
            rel_intensities = single_img_array / self.ref_intensities

            kappas = self.calc_coefficients_of_img(rel_intensities)
            self.coefficients_per_image_and_layer.append(kappas)
        return None

    def set_all_member_variables(self):
        if len(self.distances_per_led_and_layer) == 0:
            self.distances_per_led_and_layer = self.calc_distance_array()
        if self.calculated_img_data.empty:
            self.load_img_data()
        if self.ref_intensities.shape[0] == 0:
            self.calc_and_set_ref_intensities()

    def load_img_data(self) -> None:
        img_data = read_hdf(self.experiment.channel, path=self.experiment.path)
        img_data_cropped = img_data[['line', self.reference_property]]
        idx = pd.IndexSlice
        img_data_cropped = img_data_cropped.loc[idx[0.0:500.0, :]]
        self.calculated_img_data = img_data_cropped[img_data_cropped['line'] == self.experiment.led_array]

    def save(self) -> None:
        path = self.experiment.path / 'analysis' / 'AbsorptionCoefficients'
        if not path.exists():
            path.mkdir(parents=True)
        path = path / f'absorption_coefs_{self.type}_channel_{self.experiment.channel}.csv'
        np.savetxt(path, self.coefficients_per_image_and_layer, delimiter=',')

    def calc_distance_array(self) -> np.ndarray:
        distances = []
        for led in self.experiment.leds:
            d = self.experiment.calc_traversed_dist_per_layer(led)
            distances.append(d)
        return np.array(distances)

    def calc_and_set_ref_intensities(self) -> None:
        ref_img_data = self.calculated_img_data.query(f'img_id <= {self.num_ref_imgs}')
        ref_intensities = ref_img_data.mean(0, level='led_id')
        self.ref_intensities = ref_intensities[self.reference_property].to_numpy()

    @abstractmethod
    def calc_coefficients_of_img(self, rel_intensities: np.ndarray) -> np.ndarray:
        pass
