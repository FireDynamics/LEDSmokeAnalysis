from dataclasses import dataclass

from ledsa.analysis.ConfigDataAnalysis import ConfigDataAnalysis
from ledsa.analysis.Experiment import Camera, Layers
# Todo: Import init_function for request of missing data
from ledsa.core.ConfigData import ConfigData


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
    :ivar reference_property: Reference property to be analysed.
    :type reference_property: str
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
        self.lambda_reg = None
        self.reference_property = None
        self.merge_led_arrays = None
        self.solver = None
        self.load_config_parameters()  # Todo: Does that belong here?

    def load_config_parameters(self) -> None:
        """
        Load experiment data from configuration file.

        """
        config_analysis = self.config_analysis
        num_layers = int(config_analysis['model_parameters']['num_layers'])
        self.channels = config_analysis.get_list_of_values('DEFAULT', 'camera_channels')
        self.num_ref_images = int(config_analysis['DEFAULT']['num_ref_images'])
        self.weighting_preference = float(config_analysis['DEFAULT']['weighting_preference'])
        self.weighting_curvature = float(config_analysis['DEFAULT']['weighting_curvature'])
        self.lambda_reg = float(config_analysis['DEFAULT']['lambda_reg'])
        self.num_iterations = int(config_analysis['DEFAULT']['num_iterations'])
        self.reference_property = config_analysis['DEFAULT']['reference_property']
        self.solver = config_analysis['DEFAULT']['solver']

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
