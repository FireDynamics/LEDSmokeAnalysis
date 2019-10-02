import configparser as cp


class ConfigData(cp.ConfigParser):
    def __init__(self, load_data=False, img_directory='.', window_radius=10, num_of_arrays=None,
                 multicore_processing=False, num_of_cores=2, reference_img=None):
        cp.ConfigParser.__init__(self)
        if load_data:
            self.load()
        else:
            self['DEFAULT'] = {}
            self['DEFAULT']['img_directory'] = str(img_directory)
            self['DEFAULT']['window_radius'] = str(window_radius)
            self['DEFAULT']['num_of_arrays'] = str(num_of_arrays)
            self['DEFAULT']['multicore_processing'] = str(multicore_processing)
            self['DEFAULT']['num_of_cores'] = str(num_of_cores)

            self['find_search_areas'] = {}
            self['find_search_areas']['reference_img'] = str(reference_img)

            self['analyse_positions'] = {}
            self['analyse_positions']['ignore_indices'] = 'None'
            self['analyse_positions']['line_edge_indices'] = 'None'

            self['analyse_photo'] = {}

            with open('config.ini', 'w') as configfile:
                self.write(configfile)
            print('config.ini created')

    def load(self):
        self.read('config.ini')
        print('Config.ini loaded')

    def save(self):
        with open('config.ini', 'w') as configfile:
            self.write(configfile)
        print('config.ini saved')


if __name__ == 'main':
    ConfigData()
