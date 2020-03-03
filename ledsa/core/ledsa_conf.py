import configparser as cp
import numpy as np
from os import path
from PIL import Image
from PIL.ExifTags import TAGS

import ledsa.core._times as times


class ConfigData(cp.ConfigParser):
    # don't give img_directory a standard value
    def __init__(self, load_config_file=True, img_directory='.', window_radius=10, num_of_arrays=None,
                 multicore_processing=False, num_of_cores=1, reference_img=None, date=None, start_time=None,
                 time_diff_to_image_time=None, time_img=None, img_name_string=None, first_img=None, last_img=None,
                 first_analyse_img=None, last_analyse_img=None, skip_imgs=0, skip_leds=0, channel=0):
        cp.ConfigParser.__init__(self)
        if img_directory[-1] != path.sep:
            img_directory += path.sep
        if load_config_file:
            self.load()
        else:
            self['DEFAULT'] = {}
            self['DEFAULT']['img_directory'] = str(img_directory)
            self['DEFAULT']['window_radius'] = str(window_radius)
            self['DEFAULT']['num_of_arrays'] = str(num_of_arrays)
            self['DEFAULT']['multicore_processing'] = str(multicore_processing)
            self['DEFAULT']['num_of_cores'] = str(num_of_cores)

            self['DEFAULT']['date'] = str(date)
            self['DEFAULT']['start_time'] = str(start_time)
            self['DEFAULT']['time_diff_to_img_time'] = str(time_diff_to_image_time)
            self['DEFAULT']['time_img'] = str(time_img)

            self['DEFAULT']['img_name_string'] = str(img_name_string)

            self['DEFAULT']['first_img'] = str(first_img)
            self['DEFAULT']['last_img'] = str(last_img)

            self['find_search_areas'] = {}
            self['find_search_areas']['reference_img'] = str(reference_img)

            self['analyse_positions'] = {}
            self['analyse_positions']['ignore_indices'] = 'None'
            self['analyse_positions']['line_edge_indices'] = 'None'
            self['analyse_positions']['line_edge_coordinates'] = 'None'

            self['analyse_photo'] = {}
            self['analyse_photo']['first_img'] = str(first_analyse_img)
            self['analyse_photo']['last_img'] = str(last_analyse_img)
            self['analyse_photo']['skip_imgs'] = str(skip_imgs)
            self['analyse_photo']['skip_leds'] = str(skip_leds)
            self['analyse_photo']['channel'] = str(channel)

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

    def get2dnparray(self, section, option, num_col=2, dtype=int):
        if self[section][option] == 'None':
            return None
        indices_tmp = [dtype(i) for i in self[section][option].split()]
        indices = np.zeros((len(indices_tmp) // num_col, num_col), dtype=dtype)
        for i in range(len(indices_tmp) // num_col):
            indices[i][:] = indices_tmp[num_col * i:num_col * i + num_col]
            # indices[i][1] = indices_tmp[2 * i + 1]
        return indices

    def in_ref_img(self):
        self['find_search_areas']['reference_img'] = input('Please give the name of the reference image, from where the'
                                                           ' led positions are calculated and which will be the start '
                                                           'of the experiment time calculation: ')

    def in_time_img(self):
        self['DEFAULT']['time_img'] = input('Please give the name of the time reference image, the image where a clock '
                                            'is visible, to synchronise multiple cameras in one experiment: ')

    def in_num_of_arrays(self):
        self['DEFAULT']['num_of_arrays'] = input('Please give the number of LED lines: ')

    def in_time_diff_to_img_time(self):
        time = input('Please give the time shown on the clock in the time reference image in hh:mm:ss: ')

        exif = _get_exif(self['DEFAULT']['img_directory'] + self['DEFAULT']['time_img'])
        if not exif:
            raise ValueError("No EXIF metadata found")

        for (idx, tag) in TAGS.items():
            if tag == 'DateTimeOriginal':
                if idx not in exif:
                    raise ValueError("No EXIF time found")
                date, time_meta = exif[idx].split(' ')
                self['DEFAULT']['date'] = date
                self['DEFAULT']['time_diff_to_img_time'] = str(times.time_diff(time_meta, time))

    def in_img_name_string(self):
        self['DEFAULT']['img_name_string'] = input(
            'Please give the name structure of the images in the form img{}.jpg where '
            '{} denotes the increasing number of the image files: ')

    def in_first_img(self):
        self['DEFAULT']['first_img'] = input('Please give the number of the first image file of the experiment: ')

    def in_last_img(self):
        self['DEFAULT']['last_img'] = input('Please give the number of the last image file of the experiment:  ')

    def in_line_edge_coordinates(self):
        print('Please enter the coordinates of the top most and bottom most LED of each array corresponding to the '
              'order of the line edge indices. Separate the two coordinates with a whitespace.')
        coordinates = str()
        for i in range(int(self['num_of_arrays'])):
            line = input(str(i) + '. array: ')
            coordinates += '\t    ' + line + '\n'
        self['analyse_positions']['line_edge_coordinates'] = '\n' + coordinates

    # get the uncorrected start time from the reference image
    def get_start_time(self):
        exif = _get_exif(self['DEFAULT']['img_directory'] + self['find_search_areas']['reference_img'])
        if not exif:
            raise ValueError("No EXIF metadata found")

        for (idx, tag) in TAGS.items():
            if tag == 'DateTimeOriginal':
                if idx not in exif:
                    raise ValueError("No EXIF time found")
                _, time_meta = exif[idx].split(' ')
                print(time_meta)
                self['DEFAULT']['start_time'] = time_meta


def _get_exif(filename):
    image = Image.open(filename)
    image.verify()
    return image._getexif()


if __name__ == 'main':
    ConfigData()
