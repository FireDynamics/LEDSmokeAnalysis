#!/usr/bin/env python

import os

import ledsa.core.file_handling
import ledsa.core.image_handling
import ledsa.data_extraction.step_1_functions
import ledsa.data_extraction.step_2_functions
import ledsa.data_extraction.step_3_functions
import numpy as np
import matplotlib.pyplot as plt

from ledsa.data_extraction import init_functions as led
from ledsa.core.ConfigData import ConfigData

sep = os.path.sep


class DataExtractor:
    """
    Class for extracting the data from the experiment images.
    """
    
    def __init__(self, channels=[0], load_config_file=True, build_experiment_infos=True, fit_leds=True):
        self.config = ConfigData(load_config_file=load_config_file)

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
        self.search_areas = ledsa.core.file_handling.read_table(filename, delim=',')
        # self.last_fit_results = self.search_areas.shape[0] * [[10, 10, 2., 2., 200., 1.0, 1.0, 1.0]]
    
    def find_search_areas(self, img_filename):
        """
        finds all LEDs in a single image file and defines the search areas, in
        which future LEDs will be searched
        """
        config = self.config['find_search_areas']
        ref_img_name = "{}{}".format(config['img_directory'], img_filename)
        data = ledsa.core.image_reading.read_img(ref_img_name, channel=0)

        self.search_areas = ledsa.data_extraction.step_1_functions.find_search_areas(data, skip=1, window_radius=int(config['window_radius']),
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
        data = ledsa.core.image_reading.read_img(filename, channel=0)

        plt.figure(dpi=1200)
        ax = plt.gca()
        ledsa.data_extraction.step_1_functions.add_search_areas_to_plot(self.search_areas, ax, config)
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
        self.line_indices = ledsa.data_extraction.step_2_functions.match_leds_to_led_arrays(self.search_areas, self.config)
        ledsa.data_extraction.step_2_functions.generate_line_indices_files(self.line_indices)
        ledsa.data_extraction.step_2_functions.generate_labeled_led_arrays_plot(self.line_indices, self.search_areas)

    def load_line_indices(self):
        """loads the line indices from the csv file"""
        self.line_indices = []
        for i in range(int(self.config['DEFAULT']['num_of_arrays'])):
            filename = 'analysis{}line_indices_{:03}.csv'.format(sep, i)
            self.line_indices.append(ledsa.core.file_handling.read_table(filename, dtype='int'))

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

        img_filenames = ledsa.core.file_handling.read_table('images_to_process.csv', dtype=str)
        num_of_cores = int(config['num_of_cores'])
        if num_of_cores > 1:
            from multiprocessing import Pool
            print('images are getting processed, this may take a while')
            with Pool(num_of_cores) as p:
                p.map(self.process_img_file, img_filenames)
        else:
            for i in range(len(img_filenames)):
                self.process_img_file(img_filenames[i])
                print('image ', i+1, '/', len(img_filenames), ' processed')

        os.remove('images_to_process.csv')

    def process_img_file(self, img_filename):
        """workaround for pool.map"""
        img_id = ledsa.core.image_handling.get_img_id(img_filename)
        for channel in self.channels:
            img_data = ledsa.data_extraction.step_3_functions.generate_analysis_data(img_filename, channel, self.search_areas, self.line_indices,
                                                                                     self.config, self.fit_leds)
            ledsa.data_extraction.step_3_functions.create_fit_result_file(img_data, img_id, channel)
        print('Image {} processed'.format(img_id))

    def setup_step3(self):
        led.generate_image_infos_csv(self.config, build_analysis_infos=True)
        ledsa.data_extraction.step_3_functions.create_imgs_to_process_file()

    def setup_restart(self):
        if len(self.channels) > 1:
            print('Restart of a run currently only supports one channel. \nExiting...')
            exit(1)
        ledsa.data_extraction.step_3_functions.find_and_save_not_analysed_imgs(self.channels[0])
