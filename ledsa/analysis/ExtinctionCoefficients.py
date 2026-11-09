import warnings
from abc import ABC, abstractmethod
from multiprocessing import Pool
from typing import List

import numpy as np
import pandas as pd

from ledsa.analysis.Experiment import Experiment, Layers, Camera
from ledsa.analysis.temperature_correction import TemperatureCorrection
from ledsa.core.file_handling import (NOISE_COLUMNS, NUM_IMGS_PER_AVERAGE, avg_img_ids_of_ref_imgs,
                                      create_analysis_infos_avg, extend_hdf,
                                      get_extinction_coefficients_file_path, read_analysis_img_ids,
                                      read_experiment_times,
                                      read_hdf, read_hdf_avg)
from ledsa.data_extraction.data_integrity import check_intensity_normalization

from importlib.metadata import version


# Signal-to-noise ratio from which on an LED gets the full weight. Measured on the Warsaw tests V008 and
# V009: at the fog stages that still carry signal the weighting changes the extinction coefficients by
# less than a percent, and where the transmission approaches the read noise it is the only thing that
# separates "no signal" from "small signal".
DEFAULT_SNR_FULL_WEIGHT = 10.0

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
    :ivar ref_img_ids_of_loaded_data: Indices the reference images carry in the loaded image data.
        They differ from ref_img_indices when the averaged data are loaded, which are numbered
        from 1 again.
    :vartype ref_img_ids_of_loaded_data: list[int] or None
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
    :ivar n_summarize: Number of consecutive images summarized into one when averaging.
    :vartype n_summarize: int
    :ivar temperature_correction: Reduction factors of the LED intensities over time and height, None for
        an analysis without the temperature correction.
    :vartype temperature_correction: TemperatureCorrection or None
    :ivar temperature_corrected: Whether the reduction factors have been divided out of the loaded data.
    :vartype temperature_corrected: bool
    :ivar solver: Indication whether the calculation is to be carried out numerically or analytically.
    :vartype type: str
    """

    def __init__(self, experiment, reference_property='sum_col_val', num_ref_imgs=10, ref_img_indices=None,
                 average_images=False, n_summarize=NUM_IMGS_PER_AVERAGE,
                 temperature_correction=None, snr_full_weight=DEFAULT_SNR_FULL_WEIGHT,
                 num_moving_avg_images=1):
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
        :param n_summarize: Number of consecutive images summarized into one when averaging.
        :type n_summarize: int
        :param temperature_correction: Reduction factors of the LED intensities of the channel. None
            leaves the intensities as they were extracted.
        :type temperature_correction: TemperatureCorrection or None
        :param snr_full_weight: Signal-to-noise ratio from which on an LED gets the full weight in the
            solver. Below it the weight falls linearly to zero at a signal of zero. None weights all LEDs
            equally. The weighting needs a reference property with a noise column and noise values in the
            data; where either is missing it is switched off with a note, so that the default does not
            stop an analysis of data that cannot carry it.
        :type snr_full_weight: float or None
        :param num_moving_avg_images: Number of consecutive images over which the reference property is
            smoothed by a centred moving average before the normalisation. 1 leaves it as it is. Unlike
            average_images this keeps every image with its own time and does not renumber the images.
        :type num_moving_avg_images: int
        :raises ValueError: If snr_full_weight is given for a reference property without a noise column, or
            if average_images and num_moving_avg_images are combined.
        """
        if average_images and int(num_moving_avg_images) > 1:
            raise ValueError('average_images and num_moving_avg_images cannot be combined. '
                             'average_images summarizes n_summarize images into one and numbers the '
                             'result from 1 again, while num_moving_avg_images smooths the reference '
                             'property over neighbouring images and keeps every image with its own '
                             'time. Pick the one that suits the time resolution you need.')
        self.coefficients_per_image_and_layer = []
        self.experiment = experiment
        self.reference_property = reference_property
        self.num_ref_imgs = num_ref_imgs
        self.ref_img_indices = ref_img_indices
        self.ref_img_ids_of_loaded_data = ref_img_indices
        self.calculated_img_data = pd.DataFrame()
        self.analysed_img_ids = []
        self.distances_per_led_and_layer = np.array([])
        self.ref_intensities = np.array([])
        self.cc_matrix = None
        self.average_images = average_images
        self.n_summarize = n_summarize
        self.temperature_correction = temperature_correction
        self.temperature_corrected = False
        self.solver = None
        self.lambda_reg = None
        self.snr_full_weight = snr_full_weight
        self.num_moving_avg_images = int(num_moving_avg_images)
        self.noise_property = None
        if snr_full_weight is not None:
            if reference_property in NOISE_COLUMNS:
                self.noise_property = NOISE_COLUMNS[reference_property]
            else:
                print(f'The LEDs are weighted equally: {reference_property} has no noise column, so the '
                      f'signal-to-noise ratio of an LED is unknown. Columns with one: '
                      f'{", ".join(sorted(NOISE_COLUMNS))}.')

    def __str__(self):
        out = str(self.experiment) + \
              f'Reference_property: {self.reference_property}, Ref_img_indices: {self.ref_img_indices}, Snr_full_weight: {self.snr_full_weight}, Noise_property: {self.noise_property}, Min_layer_weight: {getattr(self, "min_layer_weight", None) if self.noise_property is not None else None}, Num_moving_avg_images: {self.num_moving_avg_images}, Solver: {self.solver}, Lambda_reg: {self.lambda_reg}, Temperature_correction: {self.temperature_correction_info()}, LEDSA {version("ledsa")}\n'
        return out

    def temperature_correction_info(self) -> str:
        """
        Note for the result file stating whether the channel of this run is temperature corrected.

        :return: The channel and the reduction factors it was corrected with, 'None' if the
            intensities were analysed as they were extracted.
        :rtype: str
        """
        if self.temperature_correction is None:
            return 'None'
        return f'channel {self.experiment.channel} from {self.temperature_correction}'

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
            if self.noise_property is None:
                sigmas = self.calc_coefficients_of_img(rel_intensities)
            else:
                snr = single_img_array / single_img_data[self.noise_property].to_numpy()
                sigmas = self.calc_coefficients_of_img(rel_intensities, snr)
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
        if self.noise_property is None:
            sigmas = pool.map(self.calc_coefficients_of_img, rel_intensities)
        else:
            snr = img_property_array / multiindex_series_to_nparray(
                self.calculated_img_data[self.noise_property])
            sigmas = pool.starmap(self.calc_coefficients_of_img, zip(rel_intensities, snr))
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
        # before the reference intensities are averaged, whether or not the data were loaded here
        self.apply_temperature_correction()
        if self.ref_intensities.shape[0] == 0:
            self.calc_and_set_ref_intensities()

    def load_img_data(self) -> None:
        """
        Load processed image data from binary file

        :raises ValueError: If the reference property is not among the columns of the binary data or
            contains NaN values.
        """
        if self.average_images:
            # the binary and the image infos of the averaged images are built from the same
            # counts and reference images, so that they hold the same image IDs
            img_data = read_hdf_avg(self.experiment.channel, path=self.experiment.path,
                                    n_summarize=self.n_summarize, num_ref_imgs=self.num_ref_imgs,
                                    ref_img_ids=self.ref_img_indices)
            create_analysis_infos_avg(n_summarize=self.n_summarize, num_ref_imgs=self.num_ref_imgs,
                                      ref_img_ids=self.ref_img_indices, path=self.experiment.path)
            if self.ref_img_indices is not None:
                # The reference images are kept unaveraged, so each of them is still an image of
                # its own, but the averaged images are numbered from 1 again and the configured
                # indices would name other images in them.
                self.ref_img_ids_of_loaded_data = avg_img_ids_of_ref_imgs(
                    self.ref_img_indices, read_analysis_img_ids(self.experiment.path),
                    n_summarize=self.n_summarize)
        else:
            img_data = read_hdf(self.experiment.channel, path=self.experiment.path)
        if self.reference_property not in img_data.columns:
            # The color corrected columns are not part of the step 3 result files, the analysis adds them
            # to the binary data itself, so they are gone as soon as the binary data are rebuilt.
            hint = ('Run the analysis with --cc to compute it again, it is added to the binary data by the '
                    'analysis itself and is lost whenever they are rebuilt from the step 3 result files.'
                    if self.reference_property.endswith('_cc') else
                    'Check the reference_property option of the analysis configuration.')
            raise ValueError(f'{self.reference_property} is not among the columns of the binary data of '
                             f'channel {self.experiment.channel} '
                             f'({", ".join(map(str, img_data.columns))}). {hint}')
        columns = ['led_array_id', self.reference_property]
        if self.noise_property is not None:
            if self.noise_property not in img_data.columns:
                print(f'The LEDs are weighted equally: {self.noise_property} is not among the columns of '
                      f'the binary data of channel {self.experiment.channel}. Run step 3 again with a '
                      f'LEDSA version that writes it to weight them by their signal-to-noise ratio.')
                self.noise_property = None
            else:
                columns.append(self.noise_property)
        img_data_cropped = img_data[columns]
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
        if self.noise_property is not None and self.calculated_img_data[self.noise_property].isna().all():
            # JPG and PNG images and sensors without an optical black margin have no read noise
            print(f'The LEDs are weighted equally: {self.noise_property} is NaN for every LED of led '
                  f'array {self.experiment.led_array}, so the images carry no read noise.')
            self.calculated_img_data = self.calculated_img_data.drop(columns=[self.noise_property])
            self.noise_property = None

        # Images without analysis results are missing from the binary data, so the IDs are not necessarily a
        # gapless range. They are kept to pair every computed coefficient with the time of its own image.
        self.analysed_img_ids = sorted(
            self.calculated_img_data.index.get_level_values('img_id').unique().tolist())
        if self.num_moving_avg_images > 1:
            self.apply_moving_average()

    def apply_moving_average(self) -> None:
        """
        Replace the reference property of every LED by its centred moving average over
        num_moving_avg_images consecutive analysed images, fewer at the start and the end of the series.
        The noise, where it is used, becomes the standard deviation of that mean, so the root of the summed
        variances divided by the number of images.

        Every image keeps its own time and its own row, so the series is smoothed rather than decimated:
        the time resolution becomes num_moving_avg_images times the image interval while the number of
        extinction coefficients stays the same. average_images with n_summarize is the other way round.
        """
        n = self.num_moving_avg_images
        columns = [self.reference_property] + ([self.noise_property] if self.noise_property else [])
        averaged = {}
        for column in columns:
            wide = self.calculated_img_data[column].unstack('led_id')
            if column == self.noise_property:
                count = wide.notna().astype(float).rolling(n, center=True, min_periods=1).sum()
                wide = np.sqrt((wide ** 2).rolling(n, center=True, min_periods=1).sum()) / count
            else:
                wide = wide.rolling(n, center=True, min_periods=1).mean()
            averaged[column] = wide.stack()
        data = self.calculated_img_data.copy()
        for column, values in averaged.items():
            data[column] = values.reindex(data.index).to_numpy()
        self.calculated_img_data = data
        print(f'Reference property smoothed over {n} consecutive images (centred moving average).')

    def apply_temperature_correction(self) -> None:
        """
        Divide the reference property by the reduction factor of the time and height of every value.

        Does nothing if the channel has no reduction factors configured or if they were divided out
        already, so that the factors enter the intensities exactly once however often this is called.

        The correction runs before the reference intensities are averaged, so that I0 is a corrected
        intensity as well and the reduction of the reference images cancels out of the relative
        intensities. The factors are looked up on the (img_id, led_id) index of the loaded data rather
        than on a positional array, because the LED axis of the intensities is ordered by LED ID while
        the distance matrix follows the reversed LEDs of the experiment.

        :raises ValueError: If an image or an LED of the loaded data has no time or height.
        """
        if self.temperature_correction is None or self.temperature_corrected:
            return
        index = self.calculated_img_data.index
        times = self._experiment_times_of(index.get_level_values('img_id').to_numpy())
        heights = self._heights_of(index.get_level_values('led_id').to_numpy())
        factors = self.temperature_correction.factors_at(times, heights)
        corrected_intensities = self.calculated_img_data[self.reference_property] / factors
        self.calculated_img_data = self.calculated_img_data.assign(
            **{self.reference_property: corrected_intensities})
        self.temperature_corrected = True
        print(f'Temperature correction applied to {self.reference_property} of channel '
              f'{self.experiment.channel}, LED array {self.experiment.led_array}, with the reduction '
              f'factors of {self.temperature_correction}.')

    def _experiment_times_of(self, img_ids: np.ndarray) -> np.ndarray:
        """
        Experiment time of every given image, read from the image infos of the experiment.

        :param img_ids: IDs of the images to look up, with repetitions.
        :type img_ids: np.ndarray
        :return: Experiment time [s] of every given image, in the order of the argument.
        :rtype: np.ndarray
        :raises ValueError: If one of the images is missing from the image infos.
        """
        times_per_img_id = read_experiment_times(self.experiment.path, self.average_images)
        missing = sorted(set(img_ids.tolist()) - set(times_per_img_id))
        if missing:
            raise ValueError(f'The images {missing} of channel {self.experiment.channel} hold intensity '
                             f'values but no experiment time, so the reduction factors of the '
                             f'temperature correction cannot be interpolated to them. The image infos '
                             f'and the analysis results in {self.experiment.path} are out of sync.')
        return np.array([times_per_img_id[img_id] for img_id in img_ids], dtype=float)

    def _heights_of(self, led_ids: np.ndarray) -> np.ndarray:
        """
        Height of every given LED, taken from the coordinates of the LED array.

        :param led_ids: IDs of the LEDs to look up, with repetitions.
        :type led_ids: np.ndarray
        :return: Height [m] of every given LED, in the order of the argument.
        :rtype: np.ndarray
        :raises ValueError: If one of the LEDs has no coordinates.
        """
        height_per_led_id = {led.id: led.pos_z for led in self.experiment.leds}
        missing = sorted(set(led_ids.tolist()) - set(height_per_led_id))
        if missing:
            raise ValueError(f'The LEDs {missing} of LED array {self.experiment.led_array} have no '
                             f'coordinates, so the reduction factors of the temperature correction '
                             f'cannot be interpolated to their height. Run the computation of the LED '
                             f'coordinates (-coord) in {self.experiment.path} before the analysis.')
        return np.array([height_per_led_id[led_id] for led_id in led_ids], dtype=float)

    def save(self) -> None:
        """
        Save the computed extinction coefficients to a file.

        The file goes into a directory of its solver and its analysed quantity, so that the
        results of the quantities of one analysis run do not overwrite each other.
        """
        path = get_extinction_coefficients_file_path(self.solver, self.experiment.channel,
                                                     self.reference_property, self.experiment.led_array,
                                                     root=self.experiment.path)
        path.parent.mkdir(parents=True, exist_ok=True)
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
        if self.ref_img_ids_of_loaded_data is not None:
            requested_img_ids = list(self.ref_img_ids_of_loaded_data)
            ref_img_data = self.calculated_img_data.query(f'img_id == {requested_img_ids}')
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
        # The corrected values stay floats, see ledsa.analysis.data_preparation.apply_color_correction
        cc_val_array = np.dot(cc_matrix_inv, raw_val_array.T).T
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