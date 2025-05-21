import configparser as cp


class ConfigDataAnalysis(cp.ConfigParser):
    """
    Class responsible for handling the configuration data related to LEDSA's data analysis.

    """
    def __init__(self, load_config_file=True, camera_position=None, num_layers=20, domain_bounds=None,
                 led_array_indices=None, num_ref_images=10, camera_channels=0, num_cores=1,
                 reference_property='sum_col_val',
                 average_images=False, solver='linear', weighting_preference=-6e-3, weighting_curvature=1e-6,
                 num_iterations=200, lambda_reg=1e-3):
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
        :param camera_channels: Camera channels to be considered in the analysis. Defaults to 0.
        :type camera_channels: List[int]
        :param num_cores: Number of CPU cores for (multicore) processing. If greater than 1, multicore processing is applied. Defaults to 1.
        :type num_cores: int
        :param reference_property: Property used for reference in LEDSA. Defaults to 'sum_col_val'.
        :type reference_property: str
        :param average_images: Determines if intensities are computed as an average from two consecutive images. Defaults to False.
        :type average_images: bool
        :param solver: Method used to compute extinction coefficients - can be 'linear' or 'nonlinear'. Defaults to 'linear'.
        :type solver: str
        :param weighting_preference: Weighting factor for the preference to push the nonlinear solver to high or low values for the extinction coeffiientes. Defaults to -6e-3.
        :type weighting_preference: float
        :param weighting_curvature:  Weighting factor for the smoothness of the solution. Defaults to 1e-6.
        :type weighting_curvature: float
        :param num_iterations: Maximum number of iterations for the nonlinear solver. Defaults to 200.
        :type num_iterations: int
        :param lambda_reg: Regularization parameter for the linear solver. Defaults to 1e-3.
        :type lambda_reg: float
        """
        cp.ConfigParser.__init__(self, allow_no_value=True)
        if load_config_file:
            self.load()
        else:
            self['DEFAULT'] = {}
            self.set('DEFAULT', '# Variables used in multiple parts of LEDSA')
            self.set('DEFAULT', '   # Number of CPUs, multicore processing is applied if > 1')
            self['DEFAULT']['   num_cores'] = str(num_cores)
            self['DEFAULT']['   reference_property'] = str(reference_property)
            self.set('DEFAULT', '   # Number images used to compute normalize LED intensities')
            self['DEFAULT']['   num_ref_images'] = str(num_ref_images)
            self['DEFAULT']['   camera_channels'] = str(camera_channels)
            self.set('DEFAULT', '   # Intensities are computed as average from two consecutive images if set to True ')
            self['DEFAULT']['   average_images'] = str(average_images)
            self.set('DEFAULT', '   # Extinction coefficients can be computed by linear or numeric solver ')
            self['DEFAULT']['   solver'] = str(solver)
            self.set('DEFAULT', '   # Options for nonlinear solver (ignored when linear solver is used) ')
            self['DEFAULT']['   weighting_preference'] = str(weighting_preference)
            self['DEFAULT']['   weighting_curvature'] = str(weighting_curvature)
            self['DEFAULT']['   num_iterations'] = str(num_iterations)
            self.set('DEFAULT', '   # Options for linear solver (ignored when nonlinear solver is used) ')
            self['DEFAULT']['   lambda_reg'] = str(lambda_reg)

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

            with open('config_analysis.ini', 'w') as configfile:
                self.write(configfile)
            print('config_analysis.ini created')

    def load(self) -> None:
        """
        Loads the configuration data from 'config_analysis.ini' file.

        Raises:
            FileNotFoundError: If 'config_analysis.ini' is not found in the working directory.
        """
        try:
            self.read_file(open('config_analysis.ini'))
        except FileNotFoundError:
            print(
                'config_analysis.ini not found in working directory! Please create it with argument "--config_analysis".')
            exit(1)
        print('config_analysis.ini loaded!')

    def save(self) -> None:
        """
        Saves the current configuration to 'config_analysis.ini' file.
        """
        with open('config_analysis.ini', 'w') as configfile:
            self.write(configfile)
        print('config_analysis.ini saved')

    def get_list_of_values(self, section:str, option:str, dtype=int) -> None:
        """
        Returns a list of values of a specified dtype from a given section and option.

        :param section: Section in the configuration file.
        :type section: str
        :param option: Option under the specified section.
        :type option: str
        :param dtype: Data type of the values to be returned. Defaults to int.
        :type dtype: type
        :return: List of values or None if the option's value is 'None'.
        :rtype: list or None
        """
        if self[section][option] == 'None':
            return None
        values = [dtype(i) for i in self[section][option].split()]
        return values

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
