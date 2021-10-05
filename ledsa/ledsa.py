#!/usr/bin/env python

import os
import numpy as np
import matplotlib.pyplot as plt

from .core import led_helper as led
from .core import ledsa_conf as lc

sep = os.path.sep


class LEDSA:
    
    def __init__(self, channels=[0], load_config_file=True, build_experiment_infos=True, fit_leds=True):
        self.config = lc.ConfigData(load_config_file=load_config_file)

        self.channels = list(channels)
        self.fit_leds = fit_leds

        # 2D numpy array with dimension (# of LEDs) x (LED_id, x, y)
        self.search_areas = None
        # 2D list with dimension (# of LED arrays) x (# of LEDs per array)
        self.line_indices = None

        led.create_needed_directories(self.channels)
        led.request_config_parameters(self.config)

        led.generate_image_infos_csv(self.config, build_experiment_infos=build_experiment_infos)

    # """
    # ------------------------------------
    # Step 1 - find LED search areas
    # ------------------------------------
    # """

    def load_search_areas(self):
        """loads the search areas from the csv file"""
        filename = 'analysis{}led_search_areas.csv'.format(sep)
        self.search_areas = led.load_file(filename, delim=',')
        # self.last_fit_results = self.search_areas.shape[0] * [[10, 10, 2., 2., 200., 1.0, 1.0, 1.0]]
    
    def find_search_areas(self, img_filename):
        """
        finds all LEDs in a single image file and defines the search areas, in
        which future LEDs will be searched
        """
        config = self.config['find_search_areas']
        ref_img_name = "{}{}".format(config['img_directory'], img_filename)
        data = led.read_file(ref_img_name, channel=0)

        self.search_areas = led.find_search_areas(data, skip=1, window_radius=int(config['window_radius']),
                                                  threshold_factor=float(config['threshold_factor']))

        out_filename = 'analysis{}led_search_areas.csv'.format(sep)
        np.savetxt(out_filename, self.search_areas, delimiter=',',
                   header='LED id, pixel position x, pixel position y', fmt='%d')

    def plot_search_areas(self, img_filename):
        """plots the search areas with their labels"""
        config = self.config['find_search_areas']
        if self.search_areas is None:
            self.load_search_areas()

        filename = "{}{}".format(config['img_directory'], img_filename)
        data = led.read_file(filename, channel=0)

        plt.figure(dpi=1200)
        ax = plt.gca()
        led.add_search_areas_to_plot(self.search_areas, ax, config)
        plt.imshow(data, cmap='Greys')
        plt.colorbar()
        plt.savefig('plots{}led_search_areas.plot.pdf'.format(sep))

    # """
    # ------------------------------------
    # Step 2 - match LEDs to arrays
    # ------------------------------------
    # """

    def match_leds_to_led_arrays(self):
        """analyses, which LED belongs to which LED line array"""
        if self.search_areas is None:
            self.load_search_areas()
        self.line_indices = led.match_leds_to_led_arrays(self.search_areas, self.config)
        led.generate_line_indices_files(self.line_indices)
        led.generate_labeled_led_arrays_plot(self.line_indices, self.search_areas)
        self.line_indices = led.merge_led_arrays(self.line_indices, self.config)
        led.generate_line_indices_files(self.line_indices, filename_extension='_merge')
        led.generate_labeled_led_arrays_plot(self.line_indices, self.search_areas, filename_extension='_merge')


    def load_line_indices(self):
        """loads the line indices from the csv file"""
        if self.config['DEFAULT']['merge_led_arrays'] != 'None':
            num_of_arrays = len(self.config.get2dnparray('DEFAULT', 'merge_led_arrays','var'))
            file_extension = '_merge'
            print("WARNING: ARRAY MERGE IS ACTIVE!!!")
        else:
            num_of_arrays = int(self.config['DEFAULT']['num_of_arrays'])
            file_extension = ''
        self.line_indices = []
        for i in range(num_of_arrays):
            filename = 'analysis{}line_indices_{:03}{}.csv'.format(sep, i, file_extension)
            self.line_indices.append(led.load_file(filename, dtype='int'))

    # """
    # ------------------------------------
    # Step 3 - LED smoke analysis
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
                p.map(self.process_img_file, img_filenames)
        else:
            for i in range(len(img_filenames)):
                self.process_img_file(img_filenames[i])
                print('image ', i+1, '/', len(img_filenames), ' processed')

        os.remove('images_to_process.csv')

    def process_img_file(self, img_filename):
        """workaround for pool.map"""
        img_id = led.get_img_id(img_filename)
        for channel in self.channels:
            img_data = led.generate_analysis_data(img_filename, channel, self.search_areas, self.line_indices,
                                                  self.config['analyse_photo'], self.fit_leds)
            led.create_fit_result_file(img_data, img_id, channel)
        print('Image {} processed'.format(img_id))

    def setup_step3(self):
        led.generate_image_infos_csv(self.config, build_analysis_infos=True)
        led.create_imgs_to_process_file()

    def setup_restart(self):
        if len(self.channels) > 1:
            print('Restart of a run currently only supports one channel. \nExiting...')
            exit(1)
        led.find_not_analysed_imgs(self.channels[0])
