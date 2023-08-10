import configparser as cp


class ConfigDataAnalysis(cp.ConfigParser):

    def __init__(self, load_config_file=True, camera_position=None, num_of_layers=20, domain_bounds=None,
                 led_arrays=None, num_ref_images=10, camera_channels=0, num_of_cores=1, reference_property='sum_col_val',
                 average_images=False, solver='numeric', weighting_preference=-6e-3, weighting_curvature=1e-6,
                 num_iterations=200):
        cp.ConfigParser.__init__(self, allow_no_value=True)
        if load_config_file:
            self.load()
        else:
            self['DEFAULT'] = {}
            self.set('DEFAULT', '# Variables used in multiple parts of LEDSA')
            self.set('DEFAULT', '   # Number of CPUs, multicore processing is applied if > 1')
            self['DEFAULT']['   num_of_cores'] = str(num_of_cores)
            self['DEFAULT']['   reference_property'] = str(reference_property)
            self.set('DEFAULT', '   # Number images used to compute normalize LED intensities')
            self['DEFAULT']['   num_ref_images'] = str(num_ref_images)
            self['DEFAULT']['   camera_channels'] = str(camera_channels)
            self.set('DEFAULT', '   # Intensities are computed as average from two consecutive images if set to True ')
            self['DEFAULT']['   average_images'] = str(average_images)
            self.set('DEFAULT', '   # Extinction coefficients can be computed by linear or numeric solver ')
            self['DEFAULT']['   solver'] = str(solver)
            self.set('DEFAULT', '   # Options for numeric solver ')
            self['DEFAULT']['   weighting_preference'] = str(weighting_preference)
            self['DEFAULT']['   weighting_curvature'] = str(weighting_curvature)
            self['DEFAULT']['   num_iterations'] = str(num_iterations)

            self['experiment_geometry'] = {}
            self.set('experiment_geometry', '# Global X Y Z position of the camera ')
            self['experiment_geometry']['   camera_position'] = str(camera_position)

            self['model_parameters'] = {}
            self.set('model_parameters', '# Parameters regarding the discretization of the spatial domain ')
            self.set('model_parameters', '   # LED arrays for that the extinction coefficients should be computed  ')
            self['model_parameters']['   led_arrays'] = str(led_arrays)
            self.set('model_parameters', '   # Number of horizontal smoke layers  ')
            self['model_parameters']['   num_of_layers'] = str(num_of_layers)
            self.set('model_parameters', '   # Lower and upper bounds of the computational domain  ')
            self['model_parameters']['   domain_bounds'] = str(domain_bounds)

            with open('config_analysis.ini', 'w') as configfile:
                self.write(configfile)
            print('config_analysis.ini created')

    def load(self):
        try:
            self.read_file(open('config_analysis.ini'))
        except FileNotFoundError:
            print('config_analysis.ini not found in working directory! Please create it with argument "--config_analysis".')
            exit(1)
        print('config_analysis.ini loaded!')


    def save(self):
        with open('config_analysis.ini', 'w') as configfile:
            self.write(configfile)
        print('config_analysis.ini saved')

    def get_list_of_values(self, section, option, dtype=int):
        if self[section][option] == 'None':
            return None
        values = [dtype(i) for i in self[section][option].split()]
        return values

    def in_camera_channels(self):
        self['DEFAULT']['camera_channels'] = input('Please give the camera channels that should be considered in the '
                                                      'analysis: ')
    def in_camera_position(self):
        self['experiment_geometry']['camera_position'] = input('Please give the global X Y Z [m] coordinates of the '
                                                                  'camera : ')
    def in_num_of_layers(self):
        self['model_parameters']['num_of_layers'] = input('Please give number of layers the spatial domain should '
                                                                'be discretized to : ')
    def in_domain_bounds(self):
        self['model_parameters']['domain_bounds'] = input('Please give lower and upper height [m] of the spatial '
                                                                'domain : ')
    def in_led_arrays(self):
        self['model_parameters']['led_arrays'] = input('Please give IDs of (merged) LED Arrays to compute: ')


if __name__ == 'main':
    ConfigDataAnalysis()