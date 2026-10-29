from dataclasses import dataclass

from ledsa.analysis.ConfigDataAnalysis import ConfigDataAnalysis
from ledsa.analysis.ExtinctionCoefficients import DEFAULT_SNR_FULL_WEIGHT
from ledsa.analysis.Experiment import Camera, Layers
# Todo: Import init_function for request of missing data
from ledsa.core.ConfigData import ConfigData
from ledsa.core.image_handling import get_analysis_img_ids


@dataclass
class ExperimentData:
    """
    Dataclass containing the data for the extinction coefficient calculation
    from the experiment_data.txt input file.

    :ivar config: Configuration data.
    :type config: ConfigData
    :ivar config_analysis: Analysis configuration data.
    :type config_analysis: ConfigDataAnalysis
    :ivar camera: Camera data.
    :type camera: Camera
    :ivar layers: Layer data.
    :type layers: Layers
    :ivar channels: List of channels.
    :type channels: List[int]
    :ivar led_arrays: List of LED arrays.
    :type led_arrays: List[int]
    :ivar n_cpus: Number of CPUs.
    :type n_cpus: int
    :ivar weighting_preference: Weighting preference for nonlinear solver.
    :type weighting_preference: float
    :ivar weighting_curvature: Weighting curvature for nonlinear solver.
    :type weighting_curvature: float
    :ivar num_iterations: Number of iterations.
    :type num_iterations: int
    :ivar num_ref_images: Number of reference images.
    :type num_ref_images: int
    :ivar ref_img_indices: Indices of reference images to use. If None, use num_ref_imgs.
    :type ref_img_indices: list[int] or None
    :ivar snr_full_weight: Signal-to-noise ratio from which on an LED gets the full weight in the linear
        solver. None weights all LEDs equally. Configurations written before the option existed do not
        hold it and therefore get the default.
    :type snr_full_weight: float or None
    :ivar min_layer_weight: Only with snr_full_weight: a layer whose path length weighted mean LED weight
        stays below this value is set to NaN.
    :type min_layer_weight: float
    :ivar num_moving_avg_images: Number of consecutive images for a centred moving average of the reference
        property. 1 leaves it as it is.
    :type num_moving_avg_images: int
    :ivar average_images: Intensities are computed as an average from consecutive images.
    :type average_images: bool
    :ivar n_summarize: Number of consecutive images summarized into one when averaging.
    :type n_summarize: int
    :ivar reference_properties: Reference properties to be analysed, one analysis each.
    :type reference_properties: List[str]
    :ivar reduction_files: Path of the reduction factor file of the temperature correction per camera
        channel, None for the channels analysed without the correction.
    :type reduction_files: dict[int, str or None]
    :ivar merge_led_arrays: Merge LED arrays option.
    :type merge_led_arrays: str
    :ivar lambda_reg: Regularization parameter for linear solver.
    :type lambda_reg: float
    """
    def __init__(self, load_config_file=True):
        self.config = ConfigData(load_config_file=load_config_file)
        self.config_analysis = ConfigDataAnalysis(load_config_file=load_config_file)
        self.camera = None
        self.layers = None
        self.channels = None
        self.led_arrays = None
        self.n_cpus = None
        self.weighting_preference = None
        self.weighting_curvature = None
        self.num_iterations = None
        self.num_ref_images = None
        self.ref_img_indices = None
        self.average_images = False
        self.n_summarize = 2
        self.lambda_reg = None
        self.snr_full_weight = DEFAULT_SNR_FULL_WEIGHT
        self.min_layer_weight = 0.5
        self.num_moving_avg_images = 1
        self.reference_properties = None
        self.reduction_files = {}
        self.merge_led_arrays = None
        self.solver = None
        self.load_config_parameters()  # Todo: Does that belong here?

    def _ref_img_indices_of_config(self) -> list:
        """
        Analysis IDs of the images config.ini names for the normalization of the LED intensities.

        ref_img_experiment_ids of config.ini names them by the ID their file name is built from,
        since the data extraction has to know them before the analysis IDs exist. The analysis
        works with the analysis IDs, which the image infos translate them into.

        :return: Analysis IDs of the reference images, or None if config.ini names none.
        :rtype: list or None
        """
        experiment_img_ids = self.config.get_ref_img_experiment_ids()
        if experiment_img_ids is None:
            return None
        return get_analysis_img_ids(experiment_img_ids, self.config['DEFAULT']['img_name_string'])

    def load_config_parameters(self) -> None:
        """
        Load experiment data from configuration file.

        """
        config_analysis = self.config_analysis
        num_layers = int(config_analysis['model_parameters']['num_layers'])
        self.channels = config_analysis.get_list_of_values('DEFAULT', 'camera_channels')
        self.ref_img_indices = config_analysis.get_list_of_values('DEFAULT', 'ref_img_indices')
        if self.ref_img_indices is None:
            self.ref_img_indices = self._ref_img_indices_of_config()
        if self.ref_img_indices is None:
            self.num_ref_images = int(config_analysis['DEFAULT']['num_ref_images'])
        self.average_images = config_analysis.getboolean('DEFAULT', 'average_images', fallback=False)
        # The number of images per group was named by no config key at all, so the averaging kept
        # the two it was written with whatever the experiment asked for.
        self.n_summarize = int(config_analysis.get('DEFAULT', 'n_summarize', fallback='2'))
        self.solver = config_analysis['DEFAULT']['solver']
        if self.solver == 'nonlinear':
            self.weighting_preference = float(config_analysis['DEFAULT']['weighting_preference'])
            self.weighting_curvature = float(config_analysis['DEFAULT']['weighting_curvature'])
            self.num_iterations = int(config_analysis['DEFAULT']['num_iterations'])
        elif self.solver == 'linear':
            self.lambda_reg = float(config_analysis['DEFAULT']['lambda_reg'])
            if config_analysis.has_option('DEFAULT', 'snr_full_weight'):
                # an explicit None switches the weighting off, which the default no longer does
                entry = config_analysis['DEFAULT']['snr_full_weight'].strip()
                self.snr_full_weight = None if entry.lower() == 'none' else float(entry)
            if config_analysis.has_option('DEFAULT', 'min_layer_weight'):
                self.min_layer_weight = float(config_analysis['DEFAULT']['min_layer_weight'])
            if config_analysis.has_option('DEFAULT', 'num_moving_avg_images'):
                self.num_moving_avg_images = int(config_analysis['DEFAULT']['num_moving_avg_images'])
        self.reference_properties = config_analysis.get_reference_properties()
        self.reduction_files = {channel: config_analysis.get_reduction_file(channel)
                                for channel in (self.channels or [])}

        self.led_arrays = config_analysis.get_list_of_values('model_parameters', 'led_array_indices')
        if self.led_arrays is None:
            config_analysis.in_led_array_indices()
            config_analysis.save()
        self.led_arrays = config_analysis.get_list_of_values('model_parameters', 'led_array_indices')

        domain_bounds = config_analysis.get_list_of_values('model_parameters', 'domain_bounds', dtype=float)
        if domain_bounds is None:
            config_analysis.in_domain_bounds()
            config_analysis.save()
        domain_bounds = config_analysis.get_list_of_values('model_parameters', 'domain_bounds', dtype=float)

        camera_position = config_analysis.get_list_of_values('experiment_geometry', 'camera_position', dtype=float)
        if camera_position is None:
            config_analysis.in_camera_position()
            config_analysis.save()
        camera_position = config_analysis.get_list_of_values('experiment_geometry', 'camera_position', dtype=float)

        self.layers = Layers(num_layers, *domain_bounds)
        self.camera = Camera(*camera_position)
        self.n_cpus = int(config_analysis['DEFAULT']['num_cores'])
        self.merge_led_arrays = str(self.config['analyse_positions']['merge_led_array_indices'])

    def request_config_parameters(self) -> None:
        """
        Prompts the user to input missing parameters of analysis configuration and updates the configuration.

        """
        config = self.config_analysis
        if config['experiment_geometry']['camera_position'] == 'None':
            config.in_camera_position()
            config.save()
        if config['model_parameters']['num_layers'] == 'None':
            config.in_num_layers()
            config.save()
        if config['model_parameters']['domain_bounds'] == 'None':
            config.in_domain_bounds()
            config.save()
        if config['model_parameters']['led_array_indices'] == 'None':
            config.in_led_array_indices()
            config.save()
