import warnings
from abc import ABC, abstractmethod
from multiprocessing import Pool
from typing import List

import numpy as np
import pandas as pd

from ledsa.analysis.Experiment import Experiment, Layers, Camera
from ledsa.core.file_handling import read_hdf, read_hdf_avg, extend_hdf, create_analysis_infos_avg
from ledsa.data_extraction.data_integrity import check_intensity_normalization

from importlib.metadata import version


class ExtinctionCoefficients(ABC):
    """
    Parent class for the calculation of the Extinction Coefficients.

    :ivar coefficients_per_image_and_layer: List of coefficients for each image and layer.
    :vartype coefficients_per_image_and_layer: list[np.ndarray]
    :ivar experiment: Object representing the experimental setup.
    :vartype experiment: Experiment
    :ivar reference_property: Reference property to be analysed.
    :vartype reference_property: str
    :ivar num_ref_imgs: Number of reference images. # TODO: create test for this
    :vartype num_ref_imgs: int
    :ivar ref_img_indices: Indices of reference images to use. If None, use num_ref_imgs.
    :vartype ref_img_indices: list[int] or None
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

    def __init__(self, experiment, reference_property='sum_col_val', num_ref_imgs=10, ref_img_indices=None,
                 average_images=False):
        """
        :param experiment: Object representing the experimental setup.
        :type experiment: Experiment
        :param reference_property: Reference property to be analysed.
        :type reference_property: str
        :param num_ref_imgs: Number of reference images.
        :type num_ref_imgs: int
        :param ref_img_indices: Indices of reference images to use. If None, use num_ref_imgs.
        :type ref_img_indices: list[int] or None
        :param average_images: Flag to determine if intensities are computed as an average from two consecutive images.
        :type average_images: bool
        """
        self.coefficients_per_image_and_layer = []
        self.experiment = experiment
        self.reference_property = reference_property
        self.num_ref_imgs = num_ref_imgs
        self.ref_img_indices = ref_img_indices
        self.calculated_img_data = pd.DataFrame()
        self.analysed_img_ids = []
        self.distances_per_led_and_layer = np.array([])
        self.ref_intensities = np.array([])
        self.cc_matrix = None
        self.average_images = average_images
        self.solver = None
        self.lambda_reg = None

    def __str__(self):
        out = str(self.experiment) + \
              f'Reference_property: {self.reference_property}, Ref_img_indices: {self.ref_img_indices}, Solver: {self.solver}, Lambda_reg: {self.lambda_reg}, LEDSA {version("ledsa")}\n'
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

    def set_all_member_variables(self, save_distances: bool = True) -> None:
        """
        Calculate distance traveled per layer, for every led, load image data from binary file and calculate reference intensities for each LED

        :param save_distances: If True, write the distance matrix to a text file
            in the current working directory.  Pass False when calling from a
            stacked context that manages its own distance file.
        :type save_distances: bool
        """
        if len(self.distances_per_led_and_layer) == 0:
            self.distances_per_led_and_layer = self.calc_distance_array()
            if save_distances:
                file_name = f'led_array_{self.experiment.led_array}_distances_per_layer.txt'
                np.savetxt(file_name, self.distances_per_led_and_layer)
        if self.calculated_img_data.empty:
            self.load_img_data()
        if self.ref_intensities.shape[0] == 0:
            self.calc_and_set_ref_intensities()

    def load_img_data(self) -> None:
        """
        Load processed image data from binary file

        :raises ValueError: If the reference property contains NaN values.
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
        # The solvers cannot handle NaN: the linear one stops with an unspecific error, the nonlinear one returns
        # zeros without a warning. sidebg_sum_col_val is NaN for LEDs without side band pixels, for example.
        nan_values = self.calculated_img_data[self.reference_property].isna()
        if nan_values.any():
            nan_led_ids = sorted(self.calculated_img_data[nan_values].index.get_level_values('led_id').unique())
            led_ids_text = ', '.join(map(str, nan_led_ids[:10])) + (', ...' if len(nan_led_ids) > 10 else '')
            raise ValueError(f'{int(nan_values.sum())} values of {self.reference_property} of led array '
                             f'{self.experiment.led_array} are NaN (led ids {led_ids_text}). For sidebg_sum_col_val '
                             f'this means that no side band pixels were found for these LEDs, see sideband_distances '
                             f'in config.ini.')
        # Images without analysis results are missing from the binary data, so the IDs are not necessarily a
        # gapless range. They are kept to pair every computed coefficient with the time of its own image.
        self.analysed_img_ids = sorted(
            self.calculated_img_data.index.get_level_values('img_id').unique().tolist())

    def save(self) -> None:
        """
        Save the computed extinction coefficients to a file.

        """
        path = self.experiment.path / 'analysis' / 'extinction_coefficients' / self.solver
        if not path.exists():
            path.mkdir(parents=True)
        path = path / f'extinction_coefficients_{self.solver}_channel_{self.experiment.channel}_{self.reference_property}_led_array_{self.experiment.led_array}.csv'
        header = str(self)
        header += 'Layer_bottom_height[m],'
        header += ','.join(map(str, self.experiment.layers.borders[:-1])) + '\n'
        header += 'Layer_top_height[m],'
        header += ','.join(map(str, self.experiment.layers.borders[1:])) + '\n'
        header += 'Layer_center_height[m],'
        header += ','.join(map(lambda x: f"{x:.2f}", (np.array(self.experiment.layers.borders[:-1]) + np.array(
            self.experiment.layers.borders[1:])) / 2)) + '\n'
        header += 'Experiment_Time[s],'
        header += 'layer0'
        header += ''.join(f',layer{i + 1}' for i in range(self.experiment.layers.amount - 1))

        experiment_times = _get_experiment_times_from_image_infos_file(self.average_images,
                                                                       self.analysed_img_ids)
        if len(experiment_times) != len(self.coefficients_per_image_and_layer):
            raise ValueError(f'Got {len(experiment_times)} experiment times for '
                             f'{len(self.coefficients_per_image_and_layer)} computed images. The image infos '
                             f'file and the analysis results of the channel are out of sync.')
        coefficients_per_time_and_layer = np.column_stack(
            (experiment_times, self.coefficients_per_image_and_layer))
        np.savetxt(path, coefficients_per_time_and_layer, delimiter=',', header=header)

    def calc_distance_array(self) -> np.ndarray:
        """
        Calculate the distances traversed between camera and LEDs in each layer.

        :return: Array of distances traversed between camera and LEDs in each layer.
        :rtype: np.ndarray
        """
        distances = np.zeros((self.experiment.num_leds, self.experiment.layers.amount))
        count = 0
        # self.experiment.leds need to be reversed to build the distance array the right way
        for led in reversed(self.experiment.leds):
            d = self.experiment.calc_traversed_dist_per_layer(led)
            distances[count] = d
            count += 1
        return distances

    def calc_and_set_ref_intensities(self) -> None:
        """
         Calculate and set the reference intensities for all LEDs based on the reference images.

         :raises ValueError: if none of the reference images holds LED data.
         """
        if self.ref_img_indices is not None:
            requested_img_ids = list(self.ref_img_indices)
            ref_img_data = self.calculated_img_data.query(f'img_id == {self.ref_img_indices}')
        else:
            # The reference window is the first num_ref_imgs images, whether or not they were analysed,
            # so a skipped image among them is missing from the average and has to be reported
            requested_img_ids = list(range(1, self.num_ref_imgs + 1))
            ref_img_data = self.calculated_img_data.query(f'img_id <= {self.num_ref_imgs}')
        self._check_requested_ref_imgs_available(ref_img_data, requested_img_ids)
        print(
            f"Images with indices {ref_img_data.index.get_level_values('img_id').unique().values} were used for calculating reference intensities.")

        ref_intensities = ref_img_data.groupby(level='led_id')[self.reference_property].mean()
        check_intensity_normalization(ref_img_data, ref_intensities, self.reference_property)
        self.ref_intensities = ref_intensities.to_numpy()

    def _check_requested_ref_imgs_available(self, ref_img_data: pd.DataFrame,
                                            requested_img_ids: List[int]) -> None:
        """
        Report the requested reference images that hold no LED data.

        Images skipped during the intensity extraction keep their ID in the image infos but
        produce no intensities, so a reference image ID naming one of them would silently
        drop out of the average instead of being noticed. This holds for the images named by
        :attr:`ref_img_indices` and for the first :attr:`num_ref_imgs` images alike.

        :param ref_img_data: Intensity data of the requested reference images.
        :type ref_img_data: pd.DataFrame
        :param requested_img_ids: IDs of the reference images the configuration asks for.
        :type requested_img_ids: List[int]
        :raises ValueError: if none of the requested reference images holds LED data.
        """
        available = set(ref_img_data.index.get_level_values('img_id').unique())
        missing = [img_id for img_id in requested_img_ids if img_id not in available]
        if not missing:
            return
        msg = (f'The reference images {missing} of LED array {self.experiment.led_array} in '
               f'{self.experiment.path} hold no intensity values. They were either skipped during '
               f'the extraction of the LED intensities or they lie outside the analysed images.')
        if not available:
            raise ValueError(f'{msg} No reference intensities can be calculated. Set num_ref_images or '
                             f'ref_img_indices in config_analysis.ini to images that hold results.')
        warnings.warn(f'{msg} The reference intensities are averaged over the remaining images '
                      f'{sorted(available)}.', category=UserWarning, stacklevel=2)

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
    # Images without analysis results are missing, so the IDs are not necessarily a gapless range
    img_ids = sorted(multi_series.index.get_level_values(0).unique().tolist())
    array = np.zeros((len(img_ids), num_leds))
    for i, img_id in enumerate(img_ids):
        array[i] = multi_series.loc[img_id]
    return array


def _get_experiment_times_from_image_infos_file(average_images, img_ids=None):
    """
    Read the experiment times of the analysed images.

    :param average_images: Whether the averaged image infos file is used.
    :type average_images: bool
    :param img_ids: IDs of the images to return the times of. All times are returned if None.
    :type img_ids: list[int] or None
    :return: Experiment time of every requested image, ordered by image ID.
    :rtype: np.ndarray
    """
    if average_images == True:
        image_infos_file = 'analysis/image_infos_analysis_avg.csv'
    else:
        image_infos_file = 'analysis/image_infos_analysis.csv'
    image_info_df = pd.read_csv(image_infos_file)
    if img_ids is not None:
        image_info_df = image_info_df[image_info_df['#ID'].isin(img_ids)]
    return image_info_df['Experiment_Time[s]'].to_numpy()