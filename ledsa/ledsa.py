#!/usr/bin/env python

import argparse
import os
import numpy as np
import matplotlib.pyplot as plt

from .core import _led_helper as led
from .core import ledsa_conf as lc

# os path separator
sep = os.path.sep


class LEDSA:
    
    def __init__(self, load_config_file=True, build_experiment_infos=True):
        self.config = lc.ConfigData(load_config_file=load_config_file)
            
        # declarations of global variables
        # 2D numpy array with dimension (# of LEDs) x (LED_id, x, y)
        self.search_areas = None
        # 2D list with dimension (# of LED arrays) x (# of LEDs per array)
        self.line_indices = None

        # creating needed directories
        if not os.path.exists('plots'):
            os.mkdir('plots')
            print("Directory plots created ")
        if not os.path.exists('analysis'):
            os.mkdir('analysis')
            print("Directory analysis created ")
        if not os.path.exists('analysis{}channel{}'.format(sep, self.config['analyse_photo']['channel'])):
            os.mkdir('analysis{}channel{}'.format(sep, self.config['analyse_photo']['channel']))
            print("Directory analysis{}channel{} created".format(sep, self.config['analyse_photo']['channel']))

        # request all unset default parameters
        # not complete
        if self.config['DEFAULT']['time_img'] == 'None' and self.config['DEFAULT']['exif_time_infront_real_time'] == 'None':
            self.config.in_time_img()
            self.config.save()
        if self.config['find_search_areas']['reference_img'] == 'None':
            self.config.in_ref_img()
            self.config.save()
        if self.config['DEFAULT']['exif_time_infront_real_time'] == 'None':
            self.config.in_time_diff_to_img_time()
            self.config.save()
        if self.config['DEFAULT']['img_name_string'] == 'None':
            self.config.in_img_name_string()
            self.config.save()
        if self.config['DEFAULT']['first_img'] == 'None':
            self.config.in_first_img()
            self.config.save()
        if self.config['DEFAULT']['last_img'] == 'None':
            self.config.in_last_img()
            self.config.save()
        if self.config['DEFAULT']['num_of_arrays'] == 'None':
            self.config.in_num_of_arrays()
            self.config.save()

        # creates an info file with infos to all images of the experiment
        img_data = led.get_img_data(self.config, build_experiment_infos=build_experiment_infos)
        out_file = open('image_infos.csv', 'w')
        out_file.write("#Count,Name,Time,Experiment_Time[s]\n")
        out_file.write(img_data)
        out_file.close()

    # """
    # ------------------------------------
    # LED area search
    # ------------------------------------
    # """
    
    def find_search_areas(self, img_filename):
        """
        finds all LEDs in a single image file and defines the search areas, in
        which future LEDs will be searched
        """
        config = self.config['find_search_areas']
        filename = "{}{}".format(config['img_directory'], img_filename)
        out_filename = 'analysis{}led_search_areas.csv'.format(sep)

        data = led.read_file(filename, channel=0)
        self.search_areas = led.find_search_areas(data, skip=1, window_radius=int(config['window_radius']))

        np.savetxt(out_filename, self.search_areas, delimiter=',',
                   header='LED id, pixel position x, pixel position y', fmt='%d')

    def load_search_areas(self):
        """loads the search areas from the csv file"""
        filename = 'analysis{}led_search_areas.csv'.format(sep)
        self.search_areas = led.load_file(filename, delim=',')

    def plot_search_areas(self, img_filename):
        """plots the search areas with their labels"""
        config = self.config['find_search_areas']
        if self.search_areas is None:
            self.load_search_areas()

        filename = "{}{}".format(config['img_directory'], img_filename)
        data = led.read_file(filename, channel=0)

        plt.figure(dpi=1200)
        ax = plt.gca()

        for i in range(self.search_areas.shape[0]):
            ax.add_patch(plt.Circle((self.search_areas[i, 2], self.search_areas[i, 1]),
                                    radius=int(config['window_radius']),
                                    color='Red', fill=False, alpha=0.25,
                                    linewidth=0.1))
            ax.text(self.search_areas[i, 2] + int(config['window_radius']),
                    self.search_areas[i, 1] + int(config['window_radius'])//2,
                    '{}'.format(self.search_areas[i, 0]), fontsize=1)

        plt.imshow(data, cmap='Greys')
        plt.colorbar()
        plt.savefig('plots{}led_search_areas.plot.pdf'.format(sep))

    # """
    # ------------------------------------
    # LED array analysis
    # ------------------------------------
    # """

    def analyse_positions(self):
        """analyses, which LED belongs to which LED line array"""
        if self.search_areas is None:
            self.load_search_areas()
        self.line_indices = led.analyse_position_man(self.search_areas, self.config)

        # save the labeled LEDs
        for i in range(len(self.line_indices)):
            out_file = open('analysis{}line_indices_{:03}.csv'.format(sep, i), 'w')
            for iled in self.line_indices[i]:
                out_file.write('{}\n'.format(iled))
            out_file.close()
            
    def load_line_indices(self):
        """loads the search areas from the csv file"""
        self.line_indices = []
        for i in range(int(self.config['DEFAULT']['num_of_arrays'])):
            filename = 'analysis{}line_indices_{:03}.csv'.format(sep, i)
            self.line_indices.append(led.load_file(filename, dtype='int'))
            
    def plot_lines(self):
        """plot the labeled LEDs"""
        # plot the labeled LEDs
        if self.line_indices is None:
            self.load_line_indices()
        if self.search_areas is None:
            self.load_search_areas()
        for i in range(len(self.line_indices)):
            plt.scatter(self.search_areas[self.line_indices[i], 2],
                        self.search_areas[self.line_indices[i], 1],
                        s=0.1, label='led strip {}'.format(i))

        plt.legend()
        plt.savefig('plots{}led_lines.pdf'.format(sep))
        
    # """
    # ------------------------------------
    # LED smoke analysis
    # ------------------------------------
    # """
    
    def process_image_data(self):
        """process the image data to find the changes in light intensity"""
        config = self.config['analyse_photo']
        if self.search_areas is None:
            self.load_search_areas()
        if self.line_indices is None:
            self.load_line_indices()

        img_filenames = led.load_file('images_to_process.csv', dtype=str)
        if config.getboolean('multicore_processing'):
            from multiprocessing import Pool

            print('images are getting processed, this may take a while')
            with Pool(int(config['num_of_cores'])) as p:
                p.map(self.process_file, img_filenames)
        else:
            for i in range(len(img_filenames)):
                self.process_file(img_filenames[i])
                print('image ', i+1, '/', len(img_filenames), ' processed')

        os.remove('images_to_process.csv')

    def process_file(self, img_filename):
        """workaround for pool.map"""
        img_data = led.process_file(img_filename, self.search_areas, self.line_indices, self.config['analyse_photo'])

        img_id = led.get_img_id(img_filename)
        out_file = open('analysis{}channel{}{}{}_led_positions.csv'.format(sep, self.config['analyse_photo']['channel'],
                                                                           sep, img_id), 'w')

        # find the root and the experiment time
        root = os.getcwd()
        root = root.split(sep)
        img_infos = led.load_file('analysis{}image_infos_analysis.csv'.format(sep), dtype='str', delim=',', silent=True)

        # create the header
        out_file.write('# image root = {}, photo file name = {}, channel = {}, time[s] = {}\n'.format
                       (root[-1], img_filename, self.config['analyse_photo']['channel'], img_infos[int(img_id)-1][3]))
        out_file.write("# id,         line,   x,         y,        dx,        dy,"
                       "         A,     alpha,        wx,        wy, fit_success,"
                       "   fit_fun, fit_nfev // all spatial quantities in pixel coordinates\n")

        out_file.write(img_data)
        out_file.close()
        print('Image {} processed'.format(img_id))

    def setup_step3(self):
        led.create_img_infos_analysis(self.config)
        led.create_imgs_to_process()

    def setup_restart(self):
        led.find_calculated_imgs(self.config['analyse_photo'])
