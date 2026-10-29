import configparser as cp
from typing import Iterable, List, Optional, Union

from ledsa.analysis.temperature_correction import TEMPERATURE_CORRECTION_CHANNELS
from ledsa.core.config_file import save_config
from ledsa.analysis.ExtinctionCoefficients import DEFAULT_SNR_FULL_WEIGHT

DEFAULT_REFERENCE_PROPERTY = 'sum_col_val'


def parse_reference_properties(value: str) -> List[str]:
    """
    Split the value of the reference_property option into the quantities to be analysed.

    Step 3 writes several intensity quantities per LED, and one analysis run computes the
    extinction coefficients of every quantity it is given, so the option holds one or more
    quantity names separated by whitespace. A repeated name is dropped, because its results
    would be written into the same file twice.

    :param value: Value of the reference_property option.
    :type value: str
    :return: Quantities in the order the option names them.
    :rtype: List[str]
    :raises ValueError: If the option names no quantity.
    """
    properties = []
    for name in str(value).split():
        if name.lower() == 'none':
            continue
        if name not in properties:
            properties.append(name)
    if not properties:
        raise ValueError(f"The reference_property option names no quantity ({value!r}). Give one or "
                         f"more quantities separated by spaces, e.g. 'sum_col_val bgsub_sum_col_val'.")
    return properties


def format_reference_properties(reference_property: Union[str, Iterable[str]]) -> str:
    """
    Render one quantity or a list of quantities the way the config file holds them.

    :param reference_property: Quantity or quantities to be analysed.
    :type reference_property: Union[str, Iterable[str]]
    :return: Value of the reference_property option.
    :rtype: str
    """
    if isinstance(reference_property, str):
        return reference_property
    return ' '.join(str(name) for name in reference_property)


class ConfigDataAnalysis(cp.ConfigParser):
    """
    Class responsible for handling the configuration data related to LEDSA's data analysis.

    """
    def __init__(self, load_config_file=True, camera_position=None, num_layers=20, domain_bounds=None,
                 led_array_indices=None, num_ref_images=10, ref_img_indices=None, camera_channels=0, num_cores=1,
                 reference_property='sum_col_val',
                 average_images=False, n_summarize=2, solver='linear', weighting_preference=-6e-3,
                 weighting_curvature=1e-2,
                 num_iterations=1000, lambda_reg=1e-3, snr_full_weight=DEFAULT_SNR_FULL_WEIGHT,
                 min_layer_weight=0.5,
                 num_moving_avg_images=1, reduction_files=None):
        """
        :param load_config_file: Determines whether to load the config file on initialization. Defaults to True.
        :type load_config_file: bool
        :param camera_position: Global X, Y, Z position of the camera. Defaults to None.
        :type camera_position: list[float] or None
        :param num_layers: Number of horizontal smoke layers. Defaults to 20.
        :type num_layers: int
        :param domain_bounds: Lower and upper bounds of the computational domain. Defaults to None.
        :type domain_bounds: list[float] or None
        :param led_array_indices: Indices of the LED arrays for which the extinction coefficients should be computed. Defaults to None.
        :type led_array_indices: list[int] or None
        :param num_ref_images: Number of images used to compute normalize LED intensities. Defaults to 10.
        :type num_ref_images: int
        :param ref_img_indices: Indices of reference images to use. If None, use num_ref_imgs.
        :type ref_img_indices: list[int] or None
        :param camera_channels: Camera channels to be considered in the analysis. Defaults to 0.
        :type camera_channels: List[int]
        :param num_cores: Number of CPU cores for (multicore) processing. If greater than 1, multicore processing is applied. Defaults to 1.
        :type num_cores: int
        :param reference_property: Quantity or quantities the extinction coefficients are computed
            from. Defaults to 'sum_col_val'. Use 'bgsub_sum_col_val' to base the analysis on
            intensities with the local background subtracted, estimated from the border of each LED
            search area. Several quantities, given as a list or separated by spaces, are analysed one
            after another, each into a directory of its own.
        :type reference_property: Union[str, Iterable[str]]
        :param average_images: Determines if intensities are computed as an average from two consecutive images. Defaults to False.
        :type average_images: bool
        :param n_summarize: Number of consecutive images summarized into one if average_images is set. The first
            num_ref_images images are kept as they are. Defaults to 2.
        :type n_summarize: int
        :param solver: Method used to compute extinction coefficients - can be 'linear' or 'nonlinear'. Defaults to 'linear'.
        :type solver: str
        :param weighting_preference: Weighting factor for the preference to push the nonlinear solver to high or low values for the extinction coeffiientes. Defaults to -6e-3.
        :type weighting_preference: float
        :param weighting_curvature:  Weighting factor for the smoothness of the solution. Defaults to 1e-2.
        :type weighting_curvature: float
        :param num_iterations: Maximum number of iterations for the nonlinear solver. Defaults to 1000.
        :type num_iterations: int
        :param lambda_reg: Regularization parameter for the linear solver. Defaults to 1e-3.
        :type lambda_reg: float
        :param snr_full_weight: Linear solver only: signal-to-noise ratio from which on an LED gets the
            full weight. Below it the weight falls linearly to zero at a signal of zero. None weights all
            LEDs equally. The weighting needs a reference property with a noise column, which only
            sum_col_val has, and noise values in the data; where either is missing it is switched off with
            a note.
        :type snr_full_weight: float or None
        :param min_layer_weight: Linear solver with snr_full_weight only: a layer whose path length
            weighted mean LED weight stays below this value is set to NaN. Defaults to 0.5.
        :type min_layer_weight: float
        :param num_moving_avg_images: Linear solver only: number of consecutive images over which the
            reference property is smoothed by a centred moving average, every image keeping its own time.
            1 leaves it as it is and is the default. Cannot be combined with average_images.
        :type num_moving_avg_images: int
        :param reduction_files: Path of the reduction factor file of the temperature correction per
            camera channel. Channels without an entry are analysed without the correction, which is
            the default for every channel.
        :type reduction_files: dict[int, str] or None
        """
        cp.ConfigParser.__init__(self, allow_no_value=True)
        if load_config_file:
            self.load()
        else:
            self['DEFAULT'] = {}
            self.set('DEFAULT', '# Variables used in multiple parts of LEDSA')
            self.set('DEFAULT', '   # Number of CPUs, multicore processing is applied if > 1')
            self['DEFAULT']['   num_cores'] = str(num_cores)
            self.set('DEFAULT', '   # Quantities the extinction coefficients are computed from, '
                                'separated by spaces')
            self['DEFAULT']['   reference_property'] = format_reference_properties(reference_property)
            self.set('DEFAULT', '   # Number images used to compute normalize LED intensities')
            self['DEFAULT']['   num_ref_images'] = str(num_ref_images)
            self.set('DEFAULT', '   # Indices of reference images to use')
            self['DEFAULT']['   ref_img_indices'] = str(ref_img_indices)
            self.set('DEFAULT', '   # Camera channels to be considered in the analysis')
            self['DEFAULT']['   camera_channels'] = str(camera_channels)
            self.set('DEFAULT', '   # Intensities are computed as average from n_summarize consecutive images if set to True ')
            self['DEFAULT']['   average_images'] = str(average_images)
            self.set('DEFAULT', '   # Number of consecutive images summarized into one, the reference images excluded ')
            self['DEFAULT']['   n_summarize'] = str(n_summarize)
            self.set('DEFAULT', '   # Extinction coefficients can be computed by linear or numeric solver ')
            self['DEFAULT']['   solver'] = str(solver)
            self.set('DEFAULT', '   # Options for nonlinear solver (ignored when linear solver is used) ')
            self['DEFAULT']['   weighting_preference'] = str(weighting_preference)
            self['DEFAULT']['   weighting_curvature'] = str(weighting_curvature)
            self['DEFAULT']['   num_iterations'] = str(num_iterations)
            self.set('DEFAULT', '   # Options for linear solver (ignored when nonlinear solver is used) ')
            self['DEFAULT']['   lambda_reg'] = str(lambda_reg)
            self.set('DEFAULT', '   # SNR from which on an LED gets the full weight, None for equal weights.')
            self.set('DEFAULT', '   # Needs a reference property with a noise column (sum_col_val) and')
            self.set('DEFAULT', '   # noise values in the data, otherwise the LEDs are weighted equally.')
            self['DEFAULT']['   snr_full_weight'] = str(snr_full_weight)
            self.set('DEFAULT', '   # Layers whose mean LED weight stays below this value become NaN')
            self['DEFAULT']['   min_layer_weight'] = str(min_layer_weight)
            self.set('DEFAULT', '   # Images for a centred moving average of the reference property, 1 is off.')
            self.set('DEFAULT', '   # Every image keeps its own time, unlike average_images with n_summarize.')
            self['DEFAULT']['   num_moving_avg_images'] = str(num_moving_avg_images)

            self['experiment_geometry'] = {}
            self.set('experiment_geometry', '# Global X Y Z position of the camera ')
            self['experiment_geometry']['   camera_position'] = str(camera_position)

            self['model_parameters'] = {}
            self.set('model_parameters', '# Parameters regarding the discretization of the spatial domain ')
            self.set('model_parameters', '   # LED arrays for that the extinction coefficients should be computed  ')
            self['model_parameters']['   led_array_indices'] = str(led_array_indices)
            self.set('model_parameters', '   # Number of horizontal smoke layers  ')
            self['model_parameters']['   num_layers'] = str(num_layers)
            self.set('model_parameters', '   # Lower and upper bounds of the computational domain  ')
            self['model_parameters']['   domain_bounds'] = str(domain_bounds)

            reduction_files = reduction_files if reduction_files is not None else {}
            self['temperature_correction'] = {}
            self.set('temperature_correction', '# Correction of the LED intensities for the temperature of the LEDs.')
            self.set('temperature_correction', '   # Path of the reduction factor file of every camera channel. A channel with a')
            self.set('temperature_correction', '   # file is corrected with it, a channel set to None is analysed as extracted.')
            self.set('temperature_correction', '   # The factors are computed outside LEDSA and read from a CSV file holding the')
            self.set('temperature_correction', '   # experiment time [s] in its first column, the LED height [m] in its header')
            self.set('temperature_correction', '   # and the factor the intensity of that time and height is reduced by in its')
            self.set('temperature_correction', '   # cells. They are interpolated to the time of every image and the height of')
            self.set('temperature_correction', '   # every LED, and the intensities are divided by them.')
            for channel in TEMPERATURE_CORRECTION_CHANNELS:
                self['temperature_correction'][f'   reduction_file_channel_{channel}'] = str(
                    reduction_files.get(channel))

            with open('config_analysis.ini', 'w', encoding='utf-8') as configfile:
                self.write(configfile)
            print('config_analysis.ini created')

    def load(self) -> None:
        """
        Loads the configuration data from 'config_analysis.ini' file.

        Raises:
            FileNotFoundError: If 'config_analysis.ini' is not found in the working directory.
        """
        try:
            with open('config_analysis.ini', encoding='utf-8-sig') as config_file:
                self.read_file(config_file)
        except FileNotFoundError:
            print(
                'config_analysis.ini not found in working directory! Please create it with argument "--config_analysis".')
            exit(1)
        print('config_analysis.ini loaded!')

    def save(self) -> None:
        """
        Saves the current configuration to 'config_analysis.ini' file.

        The comments of the file, which hold the help texts of the options, are kept.
        """
        save_config(self, 'config_analysis.ini')
        print('config_analysis.ini saved')

    def get_reference_properties(self) -> List[str]:
        """
        Return the quantities the extinction coefficients are to be computed from.

        :return: Quantities in the order the config file names them.
        :rtype: List[str]
        :raises ValueError: If the reference_property option names no quantity.
        """
        return parse_reference_properties(
            self.get('DEFAULT', 'reference_property', fallback=DEFAULT_REFERENCE_PROPERTY))

    def get_list_of_values(self, section: str, option: str, dtype=int, fallback=None):
        """
        Return a list[dtype] for 'section'/'option'.
        - Missing section/option → warn and return fallback
        - Value 'None' or empty → return None
        - Values are split on whitespace
        """
        if not self.has_option(section, option):
            print(
                "Config option missing: [%s].%s — using fallback=%r",
                section, option, fallback
            )
            return fallback

        raw = self.get(section, option, fallback=None)
        if raw is None:
            print(
                "Config option unreadable: [%s].%s — using fallback=%r",
                section, option, fallback
            )
            return fallback

        raw = raw.strip()
        if raw.lower() == 'none' or raw == '':
            return None

        try:
            return [dtype(item) for item in raw.split()]
        except Exception as e:
            print(
                "Config parse error for [%s].%s=%r (%s) — using fallback=%r",
                section, option, raw, e, fallback
            )
            return fallback

    def get_reduction_file(self, channel: int) -> Optional[str]:
        """
        Path of the reduction factor file of the temperature correction of a camera channel.

        A config file written before the temperature correction existed has no
        [temperature_correction] section at all, which asks for no correction just like the option
        left at None, so that an older analysis is reproduced unchanged.

        :param channel: The camera channel.
        :type channel: int
        :return: The configured path, None if the channel is analysed without the correction.
        :rtype: str or None
        """
        option = f'reduction_file_channel_{channel}'
        if not self.has_section('temperature_correction'):
            return None
        if not self.has_option('temperature_correction', option):
            return None
        raw = self.get('temperature_correction', option, fallback='None').strip()
        if raw.lower() == 'none' or raw == '':
            return None
        return raw

    def in_camera_channels(self) -> None:
        """
        Prompts the user to input the camera channels to analyse and updates the configuration.
        """
        self['DEFAULT']['camera_channels'] = input('Please give the camera channels that should be considered in the '
                                                   'analysis: ')

    def in_camera_position(self) -> None:
        """
        Prompts the user to input the camera's global X, Y, Z coordinates and updates the configuration.
        """
        self['experiment_geometry']['camera_position'] = input('Please give the global X Y Z [m] coordinates of the '
                                                               'camera : ')

    def in_num_layers(self) -> None:
        """
        Prompts the user to input the number of layers for spatial domain discretization and updates the configuration.
        """
        self['model_parameters']['num_layers'] = input('Please give number of layers the spatial domain should '
                                                          'be discretized to : ')

    def in_domain_bounds(self) -> None:
        """
        Prompts the user to input the lower and upper height of the spatial domain and updates the configuration.
        """
        self['model_parameters']['domain_bounds'] = input('Please give lower and upper height [m] of the spatial '
                                                          'domain : ')

    def in_led_array_indices(self) -> None:
        """
        Prompts the user to input the indices of (merged) LED Arrays for computation and updates the configuration.
        """
        self['model_parameters']['led_array_indices'] = input('Please give indices of (merged) LED Arrays to compute: ')


if __name__ == 'main':
    ConfigDataAnalysis()
