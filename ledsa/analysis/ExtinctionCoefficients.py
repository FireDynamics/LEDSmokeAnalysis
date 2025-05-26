from abc import ABC, abstractmethod
from multiprocessing import Pool

import numpy as np
import pandas as pd

from ledsa.analysis.Experiment import Experiment, Layers, Camera
from ledsa.core.file_handling import read_hdf, read_hdf_avg, extend_hdf, create_analysis_infos_avg


class ExtinctionCoefficients(ABC):
    """
    Parent class for the calculation of the Extinction Coefficients.

    :ivar coefficients_per_image_and_layer: List of coefficients for each image and layer.
    :vartype coefficients_per_image_and_layer: list[np.ndarray]
    :ivar experiment: Object representing the experimental setup.
    :vartype experiment: Experiment
    :ivar reference_property: Reference property to be analysed.
    :vartype reference_property: str
    :ivar num_ref_imgs: Number of reference images.
    :vartype num_ref_imgs: int
    :ivar calculated_img_data: DataFrame containing calculated image data.
    :vartype calculated_img_data: pd.DataFrame
    :ivar distances_per_led_and_layer: Array of distances traversed between camera and LEDs in each layer.
    :vartype distances_per_led_and_layer: np.ndarray
    :ivar ref_intensities: Array of reference intensities for all LEDs.
    :vartype ref_intensities: np.ndarray
    :ivar cc_matrix: Color correction matrix.
    :vartype cc_matrix: np.ndarray or None
    :ivar average_images: Flag to determine if intensities are computed as an average from two consecutive images.
    :vartype average_images: bool
    :ivar solver: Indication whether the calculation is to be carried out numerically or analytically.
    :vartype type: str
    """
    def __init__(self, experiment, reference_property='sum_col_val', num_ref_imgs=10, average_images=False):
        """
        :param experiment: Object representing the experimental setup.
        :type experiment: Experiment
        :param reference_property: Reference property to be analysed.
        :type reference_property: str
        :param num_ref_imgs: Number of reference images.
        :type num_ref_imgs: int
        :param average_images: Flag to determine if intensities are computed as an average from two consecutive images.
        :type average_images: bool
        """
        self.coefficients_per_image_and_layer = []
        self.experiment = experiment
        self.reference_property = reference_property
        self.num_ref_imgs = num_ref_imgs
        self.calculated_img_data = pd.DataFrame()
        self.distances_per_led_and_layer = np.array([])
        self.ref_intensities = np.array([])
        self.cc_matrix = None
        self.average_images = average_images
        self.solver = None

    def __str__(self):
        out = str(self.experiment) + \
              f'reference_property: {self.reference_property}, num_ref_imgs: {self.num_ref_imgs}\n'
        return out

    def calc_and_set_coefficients(self) -> None:
        """
        Serial calculation of extinction coefficients for every image

        """
        # Load and calculate all needed variables
        self.set_all_member_variables()
        for img_id, single_img_data in self.calculated_img_data.groupby(level=0):
            single_img_array = single_img_data[self.reference_property].to_numpy()
            rel_intensities = single_img_array / self.ref_intensities

            # Calculate the extinction coefficients depending on child class used
            sigmas = self.calc_coefficients_of_img(rel_intensities)
            self.coefficients_per_image_and_layer.append(sigmas)

    def calc_and_set_coefficients_mp(self, cores=4) -> None:
        """
        Uses multiprocessing to calculate and set extinction coefficients.

        :param cores: Number of cores to use.
        :type cores: int
        """
        # Load and calculate all needed variables
        self.set_all_member_variables()
        img_property_array = multiindex_series_to_nparray(self.calculated_img_data[self.reference_property])
        rel_intensities = img_property_array / self.ref_intensities

        # Calculate the extinction coefficients depending on child class used
        pool = Pool(processes=cores)
        sigmas = pool.map(self.calc_coefficients_of_img, rel_intensities)
        pool.close()
        self.coefficients_per_image_and_layer = sigmas

    def set_all_member_variables(self) -> None:
        """
        Calculate distance traveled per layer, for every led, load image data from binary file and calculate reference intensities for each LED

        """
        if len(self.distances_per_led_and_layer) == 0:
            self.distances_per_led_and_layer = self.calc_distance_array()
            np.savetxt(f'distances_per_led_and_layer.txt', self.distances_per_led_and_layer)
        if self.calculated_img_data.empty:
            self.load_img_data()
        if self.ref_intensities.shape[0] == 0:
            self.calc_and_set_ref_intensities()

    def load_img_data(self) -> None:
        """
        Load processed image data from binary file

        """
        if self.average_images:
            img_data = read_hdf_avg(self.experiment.channel, path=self.experiment.path)
            create_analysis_infos_avg()
        else:
            img_data = read_hdf(self.experiment.channel, path=self.experiment.path)
        img_data_cropped = img_data[['led_array_id', self.reference_property]]
        self.calculated_img_data = img_data_cropped[img_data_cropped['led_array_id'] == self.experiment.led_array]
        if self.calculated_img_data.empty:
            exit(f"Apparently there are no intensity values for led array {self.experiment.led_array}!")

    def save(self) -> None:
        """
        Save the computed extinction coefficients to a file.

        """
        path = self.experiment.path / 'analysis' / 'extinction_coefficients' / self.solver
        if not path.exists():
            path.mkdir(parents=True)
        path = path / f'extinction_coefficients_{self.solver}_channel_{self.experiment.channel}_{self.reference_property}_led_array_{self.experiment.led_array}.csv'
        header = str(self)
        header += 'layer0'
        for i in range(self.experiment.layers.amount - 1):
            header += f',layer{i + 1}'
        np.savetxt(path, self.coefficients_per_image_and_layer, delimiter=',', header=header)

    def calc_distance_array(self) -> np.ndarray:
        """
        Calculate the distances traversed between camera and LEDs in each layer.

        :return: Array of distances traversed between camera and LEDs in each layer.
        :rtype: np.ndarray
        """
        distances = np.zeros((self.experiment.num_leds, self.experiment.layers.amount))
        count = 0
        for led in self.experiment.leds:
            d = self.experiment.calc_traversed_dist_per_layer(led)
            distances[count] = d
            count += 1
        return distances

    def calc_and_set_ref_intensities(self) -> None:
        """
         Calculate and set the reference intensities for all LEDs based on the reference images.

         """
        ref_img_data = self.calculated_img_data.query(f'img_id <= {self.num_ref_imgs}')
        ref_intensities = ref_img_data.groupby(level='led_id').mean()

        self.ref_intensities = ref_intensities[self.reference_property].to_numpy()

    def apply_color_correction(self, cc_matrix, on='sum_col_val',
                               nchannels=3) -> None:  # TODO: remove hardcoding of nchannels
        """ Apply color correction on channel values based on color correction matrix.

        :param cc_matrix: Color correction matrix.
        :type cc_matrix: np.ndarray
        :param on: Reference property to apply the color correction on.
        :type on: str
        :param nchannels: Number of channels.
        :type nchannels: int
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

        :param rel_intensities: Array of relative change in intensity of every LED.
        :return: Array of the computed extinction coefficients
        :rtype: np.ndarray
        """
        pass


def multiindex_series_to_nparray(multi_series: pd.Series) -> np.ndarray:
    """
    Convert a multi-index series to a NumPy array.

    :param multi_series: Series with multi-index to convert.
    :type multi_series: pd.Series
    :return: Converted array.
    :rtype: np.ndarray
    """
    num_leds = pd.Series(multi_series.groupby(level=0).size()).iloc[0]
    num_imgs = pd.Series(multi_series.groupby(level=1).size()).iloc[0]
    array = np.zeros((num_imgs, num_leds))
    for i in range(num_imgs):
        array[i] = multi_series.loc[i + 1]
    return array
