from abc import ABC, abstractmethod
from multiprocessing import Pool

import numpy as np
import pandas as pd

from ledsa.analysis.Experiment import Experiment, Layers, Camera
from ledsa.core.file_handling import read_hdf, read_hdf_avg, extend_hdf, create_analysis_infos_avg


class ExtinctionCoefficients(ABC):
    """
    Parent class for the calculation of the Extinction Coefficients
    The calc_and_set_coefficients and the save method should be the only methods needed.
    """

    def __init__(self, experiment=Experiment(layers=Layers(10, 1.0, 3.35), camera=Camera(pos_x=4.4, pos_y=2, pos_z=2.3),
                                             led_array=3, channel=0),
                 reference_property='sum_col_val', num_ref_imgs=10, average_images=False):
        self.coefficients_per_image_and_layer = []
        self.experiment = experiment
        self.reference_property = reference_property
        self.num_ref_imgs = num_ref_imgs

        self.calculated_img_data = pd.DataFrame()
        self.distances_per_led_and_layer = np.array([])
        self.ref_intensities = np.array([])
        self.cc_matrix = None
        self.average_images = average_images

        self.type = None

    def __str__(self):
        out = str(self.experiment) + \
              f'reference_property: {self.reference_property}, num_ref_imgs: {self.num_ref_imgs}\n'
        return out

    def calc_and_set_coefficients(self) -> None:
        """
        main loop of the coefficient calculation
        Steps:
        1. Load and calculate all needed variables
        2. Loop over every image
            2.1 Calculate the relative change in intensities compared to the start without smoke
            2.2 Calculate the extinction coefficients depending on child class used
        """
        self.set_all_member_variables()
        for img_id, single_img_data in self.calculated_img_data.groupby(level=0):
            single_img_array = single_img_data[self.reference_property].to_numpy()
            rel_intensities = single_img_array / self.ref_intensities
            camera = 0
            np.savetxt(f'cam_{camera}_rel_intensities_{img_id}.txt', rel_intensities)  # Todo: remove
            kappas = self.calc_coefficients_of_img(rel_intensities)
            self.coefficients_per_image_and_layer.append(kappas)
        return None

    def calc_and_set_coefficients_mp(self, cores=4) -> None:
        """
        See method calc_and_set_coefficients.
        Use pool to distribute workload of the loop to multiple cores
        """
        self.set_all_member_variables()
        img_property_array = multiindex_series_to_nparray(self.calculated_img_data[self.reference_property])
        rel_intensities = img_property_array / self.ref_intensities

        pool = Pool(processes=cores)
        kappas = pool.map(self.calc_coefficients_of_img, rel_intensities)
        pool.close()
        self.coefficients_per_image_and_layer = kappas

    def set_all_member_variables(self) -> None:
        """
        1. Take the ray between led and camera and calculate the distance traveled per layer, for every led
        2. Load the binary file with the parameters calculated with ledsa core
        3. Calculate the intensities from the reference images to calculate the relative changes between smoke/no smoke later
        """
        camera = 0
        if len(self.distances_per_led_and_layer) == 0:
            self.distances_per_led_and_layer = self.calc_distance_array()
            np.savetxt(f'cam_{camera}_distances_per_led_and_layer.txt', self.distances_per_led_and_layer)
        if self.calculated_img_data.empty:
            self.load_img_data()
        if self.ref_intensities.shape[0] == 0:
            self.calc_and_set_ref_intensities()

    def load_img_data(self) -> None:
        if self.average_images:
            img_data = read_hdf_avg(self.experiment.channel, path=self.experiment.path)
            create_analysis_infos_avg()
        else:
            img_data = read_hdf(self.experiment.channel, path=self.experiment.path)
        img_data_cropped = img_data[['line', self.reference_property]]
        self.calculated_img_data = img_data_cropped[img_data_cropped['line'] == self.experiment.led_array]
        if self.calculated_img_data.empty:
            exit(f"Apparently there are no intensity values for line {self.experiment.led_array}!")

    def save(self) -> None:
        path = self.experiment.path / 'analysis' / 'AbsorptionCoefficients'
        if not path.exists():
            path.mkdir(parents=True)
        path = path / f'absorption_coefs_{self.type}_channel_{self.experiment.channel}_{self.reference_property}_led_array_{self.experiment.led_array}.csv'
        header = str(self)
        header += 'layer0'
        for i in range(self.experiment.layers.amount - 1):
            header += f',layer{i + 1}'
        np.savetxt(path, self.coefficients_per_image_and_layer, delimiter=',', header=header)

    def calc_distance_array(self) -> np.ndarray:
        distances = np.zeros((self.experiment.led_number, self.experiment.layers.amount))
        count = 0
        for led in self.experiment.leds:
            d = self.experiment.calc_traversed_dist_per_layer(led)
            distances[count] = d
            count += 1
        return distances

    def calc_and_set_ref_intensities(self) -> None:
        ref_img_data = self.calculated_img_data.query(f'img_id <= {self.num_ref_imgs}')
        ref_intensities = ref_img_data.groupby(level='led_id').mean()

        self.ref_intensities = ref_intensities[self.reference_property].to_numpy()

    def apply_color_correction(self, cc_matrix, on='sum_col_val',
                               nchannels=3) -> None:  # TODO: remove hardcoding of nchannels
        """ Apply color correction on channel values based on color correction matrix.

        """
        self.cc_matrix = cc_matrix
        cc_matrix_inv = np.linalg.inv(self.cc_matrix)
        quanity = on
        fit_params_list = []
        for channel in range(nchannels):
            fit_parameters = read_hdf(channel)[quanity]
            fit_params_list.append(fit_parameters)
        raw_val_array = pd.concat(fit_params_list, axis=1)
        cc_val_array = np.dot(cc_matrix_inv, raw_val_array.T).T
        cc_val_array = cc_val_array.astype(np.int16)
        for channel in range(nchannels):
            extend_hdf(channel, quanity + '_cc', cc_val_array[:, channel])
        print(f"Color correction applied on {nchannels} Channels!")

    @abstractmethod
    def calc_coefficients_of_img(self, rel_intensities: np.ndarray) -> np.ndarray:
        """
        Calculate the extinction coefficients for a single image.
        Needs to be implemented by the child class.
        :param rel_intensities: Array of relative change in intensity of every led in the image
        :return: Array of the coefficients
        """
        pass


def multiindex_series_to_nparray(multi_series: pd.Series) -> np.ndarray:
    index = multi_series.index
    num_leds = pd.Series(multi_series.groupby(level=0).size()).iloc[0]
    num_imgs = pd.Series(multi_series.groupby(level=1).size()).iloc[0]
    array = np.zeros((num_imgs, num_leds))
    for i in range(num_imgs):
        array[i] = multi_series.loc[i + 1]
    return array
