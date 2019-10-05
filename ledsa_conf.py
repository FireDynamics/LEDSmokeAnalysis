import configparser as cp
import numpy as np
from PIL import Image
from PIL.ExifTags import TAGS


class ConfigData(cp.ConfigParser):
    def __init__(self, load_data=False, img_directory='.', window_radius=10, num_of_arrays=None,
                 multicore_processing=False, num_of_cores=2, reference_img=None, date=None, start_time=None,
                 time_diff_to_image_time=None, time_img=None, img_name_string=None, first_img=None, last_img=None,
                 skip_imgs=0):
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

            self['DEFAULT']['date'] = str(date)
            self['DEFAULT']['start_time'] = str(start_time)
            self['DEFAULT']['time_diff_to_image_time'] = str(time_diff_to_image_time)
            self['DEFAULT']['time_img'] = str(time_img)

            self['DEFAULT']['img_name_string'] = str(img_name_string)
            self['DEFAULT']['first_img'] = str(first_img)
            self['DEFAULT']['last_img'] = str(last_img)
            self['DEFAULT']['skip_imgs'] = str(skip_imgs)

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

    def get2dnparray(self, section, option):
        if self[section][option] == 'None':
            return None
        indices_tmp = [int(i) for i in self[section][option].split()]
        indices = np.zeros((len(indices_tmp)//2, 2), dtype=int)
        for i in range(len(indices_tmp)//2):
            indices[i][0] = indices_tmp[2*i]
            indices[i][1] = indices_tmp[2*i+1]
        return indices

    def in_time_img(self):
        self['DEFAULT']['time_img'] = input('Please give the name of the time reference image: ')

    def in_time_diff_to_img_time(self):
        time = input('Please give the time of the time reference image in hh:mm:ss. can be added: ')
        time = time.split(':')

        exif = _get_exif(self['DEFAULT']['self[img_directory'] + self['DEFAULT']['time_img'])
        if not exif:
            raise ValueError("No EXIF metadata found")

        for (idx, tag) in TAGS.items():
            if tag == 'DateTimeOriginal':
                if idx not in exif:
                    raise ValueError("No EXIF time found")
                date, time_meta = exif[idx].split(' ')
                self['DEFAULT']['date'] = date
                self['DEFAULT']['time_diff_to_img_time'] = _time_diff(time, time_meta)

    def in_img_name_string(self):
        self['DEFAULT']['time_img'] = input('Please give the name structure of the images in the form img{}.jpg where '
                                            '{} denotes the increasing number of the image files')

    def in_first_img(self):
        self['DEFAULT']['first_img'] = input('Please give the number of the first image file to use: ')

    def in_last_img(self):
        self['DEFAULT']['first_img'] = input('Please give the number of the last image file to use: ')

#in work
    def get_img_data(self):
        if self.time_diff_to_img_time == 'None':
            raise ValueError('Time difference to image time not set.')
        img_data = []

        for i in range(self.getint('DEFAULT','first_img'), self.getint('DEFAULT','first_img'),
                       self.getint('DEFAULT','skip_imgs')+1):
            img_data.append(str(i) + ',' + self['DEFAULT']['img_name_string'].format(i) + time + experimet_time)


def _time_diff(time1, time2):
    t1 = time1.split(':')
    t2 = time2.split(':')
    return (t1[0]-t2[0])*3600 + t1[1]-t2[1]*60 + t1[2]-t2[2]


def _get_exif(filename):
    image = Image.open(filename)
    image.verify()
    return image._getexif()


if __name__ == 'main':
    ConfigData()

