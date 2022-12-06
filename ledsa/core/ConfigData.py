import configparser as cp
from datetime import datetime, timedelta
import numpy as np

from ledsa.core.image_reading import get_exif_entry


class ConfigData(cp.ConfigParser):
    def __init__(self, load_config_file=True, img_directory=None, window_radius=10, threshold_factor=0.25,
                 num_of_arrays=None, num_of_cores=1, date=None, start_time=None, time_img=None, time_ref_img_time=None,
                 time_diff_to_image_time=None, img_name_string=None, img_number_overflow=None, first_img_experiment=None,
                 last_img_experiment=None, reference_img=None, ignore_indices=None, line_edge_indices=None,
                 line_edge_coordinates=None, first_img_analysis=None, last_img_analysis=None, skip_imgs=0,
                 skip_leds=0):
        cp.ConfigParser.__init__(self, allow_no_value=True)
        if load_config_file:
            self.load()
        else:
            self['DEFAULT'] = {}
            self.set('DEFAULT', '# Variables used in multiple parts of LEDSA')
            self['DEFAULT']['   img_directory'] = str(img_directory)
            self.set('DEFAULT', '# String representing the naming convention of the image files')
            self['DEFAULT']['   img_name_string'] = str(img_name_string)
            self['DEFAULT']['   img_number_overflow'] = str(img_number_overflow)
            self.set('DEFAULT', '# Specify which images ar part of the experiment')
            self['DEFAULT']['   first_img'] = str(first_img_experiment)
            self['DEFAULT']['   last_img'] = str(last_img_experiment)
            self.set('DEFAULT', '   # Number of CPUs, multicore processing is applied if > 1')
            self['DEFAULT']['   num_of_cores'] = str(num_of_cores)
            self.set('DEFAULT', '')
            self.set('DEFAULT', '# Variables used to calculate the timeline of the experiment')
            self['DEFAULT']['   date'] = str(date)
            self.set('DEFAULT', '   # Beginning of the experiment, will be calculated from first_img if None')
            self.set('DEFAULT', '   # in DEFAULT')
            self['DEFAULT']['   start_time'] = str(start_time)
            self.set('DEFAULT', '   # Image with a clock to calculate the offset of the camera time. Can be None,')
            self.set('DEFAULT', '   # if time_diff_to_img_time in seconds is given')
            self['DEFAULT']['   time_img'] = str(time_img)
            self.set('DEFAULT', '   # Time shown on the time_img')
            self['DEFAULT']['   time_ref_img_time'] = str(time_ref_img_time)
            self['DEFAULT']['   exif_time_infront_real_time'] = str(time_diff_to_image_time)

            self['find_search_areas'] = {}
            self.set('find_search_areas', '# Variables used to find the pixel positions of every led')
            self.set('find_search_areas', '   # Reference image used to find and label the leds')
            self['find_search_areas']['   reference_img'] = str(reference_img)
            self.set('find_search_areas', '   # Threshold factor for the detection of LEDs on the reference image')
            self['find_search_areas']['   threshold_factor'] = str(threshold_factor)
            self.set('find_search_areas', '   # Radius of pixels assigned to each LED')
            self['find_search_areas']['   window_radius'] = str(window_radius)


            self['analyse_positions'] = {}
            self.set('analyse_positions', '# Variables used to find the physical positions of every led')
            self.set('analyse_positions', '   # Number of LED arrays')
            self['analyse_positions']['   num_of_arrays'] = str(num_of_arrays)
            self['analyse_positions']['   ignore_indices'] = str(ignore_indices)
            self.set('analyse_positions', '   # Pairs of led IDs of the edges of each led array')
            self['analyse_positions']['   line_edge_indices'] = str(line_edge_indices)
            self.set('analyse_positions', '   # Six coordinates per led array representing the physical positions of '
                                          'the')
            self.set('analyse_positions', '   # edges given in line_edge_indices')
            self['analyse_positions']['   line_edge_coordinates'] = str(line_edge_coordinates)

            self['analyse_photo'] = {}
            self.set('analyse_photo', '# Variables used for the final fitting of the intensity function')
            self.set('analyse_photo', ' ')
            self.set('analyse_photo', '# Specify which images are used for analysis')
            self['analyse_photo']['   first_img'] = str(first_img_analysis)
            self['analyse_photo']['   last_img'] = str(last_img_analysis)
            self['analyse_photo']['   skip_imgs'] = str(skip_imgs)
            self.set('analyse_photo', '   # Will only fit leds with id dividable by skip_leds + 1. Used for testing')
            self['analyse_photo']['   skip_leds'] = str(skip_leds)

            with open('config.ini', 'w') as configfile:
                self.write(configfile)
            print('config.ini created')

    def load(self):
        try:
            self.read_file(open('config.ini'))
        except FileNotFoundError:
            print('config.ini not found in working directory! Please create it with argument "--config".')
        print('config.ini loaded')

    def save(self):
        with open('config.ini', 'w') as configfile:
            self.write(configfile)
        print('config.ini saved')

    def get2dnparray(self, section, option, num_col=2, dtype=int):
        if self[section][option] == 'None':
            return None
        indices_tmp = [dtype(i) for i in self[section][option].split()]
        indices = np.zeros((len(indices_tmp) // num_col, num_col), dtype=dtype)
        for i in range(len(indices_tmp) // num_col):
            indices[i][:] = indices_tmp[num_col * i:num_col * i + num_col]
            # indices[i][1] = indices_tmp[2 * i + 1]
        return indices

    def get_datetime(self, option='start_time'):
        time = self['DEFAULT'][option]
        date = self['DEFAULT']['date']
        try:
            date_time = _get_datetime_from_str(date, time)
        except Exception as e:
            print(e)
            exit(1)
        else:
            return date_time

    def in_img_dir(self):
        self['DEFAULT']['img_directory'] = input('Please give the path where the images are stored: ')
    def in_first_img_experiment(self):
        self['DEFAULT']['first_img'] = input('Please give the number of the first image of the experiment: ')

    def in_last_img_experiment(self):
        self['DEFAULT']['last_img'] = input('Please give the number of the last image image of the experiment: ')
    def in_ref_img(self):
        self['find_search_areas']['reference_img'] = input('Please give the name of the reference image, from where the'
                                                           ' led positions are calculated and which will be the start '
                                                           'of the experiment time calculation: ')

    def in_time_img(self):
        self['DEFAULT']['time_img'] = input('Please give the name of the time reference image, the image where a clock '
                                            'is visible, to synchronise multiple cameras in one experiment: ')

    def in_num_of_arrays(self):
        self['analyse_positions']['num_of_arrays'] = input('Please give the number of LED lines: ')

    def in_time_diff_to_img_time(self):
        if self['DEFAULT']['time_ref_img_time'] == 'None':
            time = input('Please give the time shown on the clock in the time reference image in hh:mm:ss: ')
            self['DEFAULT']['time_ref_img_time'] = str(time)
        time = self['DEFAULT']['time_ref_img_time']
        print(self['DEFAULT']['img_directory'] + self['DEFAULT']['time_img'])
        tag = 'EXIF DateTimeOriginal'
        exif_entry = get_exif_entry(self['DEFAULT']['img_directory'] + self['DEFAULT']['time_img'], tag)
        date, time_meta = exif_entry.split(' ')
        self['DEFAULT']['date'] = date
        img_time = _get_datetime_from_str(date, time_meta)
        real_time = _get_datetime_from_str(date, time)
        time_diff = img_time - real_time
        self['DEFAULT']['exif_time_infront_real_time'] = str(int(time_diff.total_seconds()))

    def in_img_name_string(self):
        self['DEFAULT']['img_name_string'] = input(
            'Please give the name structure of the images in the form img{}.extension where '
            '{} denotes the increasing number of the image files: ')

    def in_img_number_overflow(self):
        self['DEFAULT']['img_number_overflow'] = input(
            'Please give the maximal number an image file can have (typically 9999): ')

    def in_first_img_analysis(self):
        self['analyse_photo']['first_img'] = input('Please give the number of the first image file to be analysed: ')

    def in_last_img_analysis(self):
        self['analyse_photo']['last_img'] = input('Please give the number of the last image file to be analysed: ')

    def in_line_edge_indices(self):
        print('The edges of the LED arrays are needed. Please enter the labels of the top most and bottom most LED of '
              'each array. Separate the two labels with a whitespace.')
        labels = str()
        for i in range(int(self['analyse_positions']['num_of_arrays'])):
            line = input(str(i) + '. array: ')
            labels += '\t    ' + line + '\n'
        self['analyse_positions']['line_edge_indices'] = '\n' + labels

    def in_line_edge_coordinates(self):
        print('Please enter the coordinates of the top most and bottom most LED of each array corresponding to the '
              'order of the line edge indices. Separate the two coordinates with a whitespace.')
        coordinates = str()
        for i in range(int(self['analyse_positions']['num_of_arrays'])):
            line = input(str(i) + '. array: ')
            coordinates += '\t    ' + line + '\n'
        self['analyse_positions']['line_edge_coordinates'] = '\n' + coordinates

    # get the start time from the first experiment image
    def get_start_time(self):
        exif_entry = get_exif_entry(self['DEFAULT']['img_directory'] + self['DEFAULT']['img_name_string'].format(
            self['analyse_photo']['first_img']), 'EXIF DateTimeOriginal')
        date, time_meta = exif_entry.split(' ')
        time_img = _get_datetime_from_str(date, time_meta)
        start_time = time_img - timedelta(seconds=self['DEFAULT'].getint('exif_time_infront_real_time'))
        self['DEFAULT']['start_time'] = start_time.strftime('%H:%M:%S')


def _get_datetime_from_str(date, time):
    if date.find(":") != -1:
        date_time = datetime.strptime(date + ' ' + time, '%Y:%m:%d %H:%M:%S')
    else:
        date_time = datetime.strptime(date + ' ' + time, '%d.%m.%Y %H:%M:%S')
    return date_time


def _get_exif(filename):
    image = Image.open(filename)
    image.verify()
    return image._getexif()


if __name__ == 'main':
    ConfigData()
