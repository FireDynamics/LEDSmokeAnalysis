import configparser as cp


class ConfigDataAnalysis(cp.ConfigParser):

    def __init__(self, load_config_file=True, camera_position=None, num_of_layers=None, domain_bounds=None,
                 led_arrays=None, merge_led_arrays=False,
                 camera_channels=0, multicore_processing=False, num_of_cores=1, reference_property='sum_col_val',
                 average_images=False, color_correction=False, solver='numeric', weighting_preference=-6e-3,
                 weighting_curvature=1e-6, num_iterations=200):
        cp.ConfigParser.__init__(self, allow_no_value=True)
        if load_config_file:
            self.load()
        else:
            self['DEFAULT'] = {}
            self.set('DEFAULT', '# Variables used in multiple parts of LEDSA')
            self.set('DEFAULT', '   # Set to True if Multiprocessing should be used')
            self['DEFAULT']['   multicore_processing'] = str(multicore_processing)
            self['DEFAULT']['   num_of_cores'] = str(num_of_cores)
            self['DEFAULT']['   reference_property'] = str(reference_property)
            self['DEFAULT']['   camera_channels'] = str(camera_channels)
            self['DEFAULT']['   average_images'] = str(average_images)
            self['DEFAULT']['   color_correction'] = str(color_correction)
            self['DEFAULT']['   solver'] = str(solver)
            self['DEFAULT']['   weighting_preference'] = str(weighting_preference)
            self['DEFAULT']['   weighting_curvature'] = str(weighting_curvature)
            self['DEFAULT']['   num_iterations'] = str(num_iterations)
            self['boundary_conditions'] = {}
            self['boundary_conditions']['   led_arrays'] = str(led_arrays)
            self['boundary_conditions']['   merge_led_arrays'] = str(merge_led_arrays)
            self['boundary_conditions']['   camera_position'] = str(camera_position)
            self['boundary_conditions']['   num_of_layers'] = str(num_of_layers)
            self['boundary_conditions']['   domain_bounds'] = str(domain_bounds)
            with open('config_analysis.ini', 'w') as configfile:
                self.write(configfile)
            print('config_analysis.ini created')

    def load(self):
        self.read('config_analysis.ini')
        print('config_analysis.ini loaded')

    def save(self):
        with open('config_analysis.ini', 'w') as configfile:
            self.write(configfile)
        print('config_analysis.ini saved')

    def in_camera_channels(self):
        self['DEFAULT']['   camera_channels'] = input('Please give the camera channels that should be consiered in the '
                                                      'analysis: ')
    def in_average_images(self):
        self['DEFAULT']['   average_images'] = input('Please state if the values from two following images should be '
                                                      'averaged : ')

    def in_color_correction(self):
        self['DEFAULT']['   color_correction'] = input('Please state if color correction should be applied on the image '
                                                      'data : ')
    def in_camera_position(self):
        self['boundary_conditions']['   camera_position'] = input('Please give the global X Y Z [m] coordinates of the '
                                                                  'camera : ')
    def in_num_of_layers(self):
        self['boundary_conditions']['   num_of_layers'] = input('Please give number of layers the spatial domain should '
                                                                'be discretized to : ')
    def in_domain_bounds(self):
        self['boundary_conditions']['   domain_bounds'] = input('Please give lower and upper height [m] of the spatial '
                                                                'domain : ')






if __name__ == 'main':
    ConfigDataAnalysis()