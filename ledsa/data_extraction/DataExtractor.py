#!/usr/bin/env python

import os

import matplotlib.pyplot as plt
import numpy as np

import ledsa.core.file_handling
import ledsa.core.image_handling
import ledsa.core.image_reading
import ledsa.data_extraction.step_1_functions
import ledsa.data_extraction.step_2_functions
import ledsa.data_extraction.step_3_functions
from ledsa.core.ConfigData import ConfigData
from ledsa.data_extraction import init_functions as led


class DataExtractor:
    """
    A class responsible for extracting data from experiment images.

    :ivar config: Configuration data object.
    :vartype config: ConfigData
    :ivar channels: Channels to be processed.
    :vartype channels: Tuple
    :ivar fit_leds: Whether to fit LEDs or not.
    :vartype fit_leds: bool
    :ivar search_areas: 2D numpy array with dimension (# of LEDs) x (LED_id, x, y).
    :vartype search_areas: numpy.ndarray, optional
    :ivar line_indices: 2D list with dimension (# of LED arrays) x (# of LEDs per array) or None.
    :vartype line_indices: list[list[int]], optional
    """
    def __init__(self, channels=(0), load_config_file=True, build_experiment_infos=True, fit_leds=True):
        """
        :param channels: Channels to be processed. Defaults to (0).
        :type channels: tuple, optional
        :param load_config_file: Whether to load existing configuration file. Defaults to True.
        :type load_config_file: bool, optional
        :param build_experiment_infos: Whether to create 'image_infos.csv'. Defaults to True.
        :type build_experiment_infos: bool, optional
        :param fit_leds: Whether to fit LEDs or not. Defaults to True.
        :type fit_leds: bool, optional
        """
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

    def load_search_areas(self) -> None:
        """
        Load LED search areas from the 'led_search_areas.csv' file. #TODO be consistent with search areas and ROIS
        """
        file_path = os.path.join('analysis', 'led_search_areas.csv')
        self.search_areas = ledsa.core.file_handling.read_table(file_path, delim=',')

    def find_search_areas(self, img_filename: str) -> None:
        """
        Identify all LEDs in a single image and define the areas where LEDs will be searched in the experiment images.

        :param img_filename: The name of the image file to be processed.
        :type img_filename: str
        """
        config = self.config['find_search_areas']
        in_file_path = os.path.join(config['img_directory'], img_filename)
        data = ledsa.core.image_reading.read_img(in_file_path, channel=0)

        self.search_areas = ledsa.data_extraction.step_1_functions.find_search_areas(data, skip=1, window_radius=int(
            config['window_radius']), threshold_factor=float(config['threshold_factor']))

        out_file_path = os.path.join('analysis', 'led_search_areas.csv')
        np.savetxt(out_file_path, self.search_areas, delimiter=',',
                   header='LED id, pixel position x, pixel position y', fmt='%d')

    def plot_search_areas(self, img_filename: str) -> None:
        """
        Plot the identified LED search areas with their ID labels.

        :param img_filename: The name of the image file to be plotted.
        :type img_filename: str
        """
        config = self.config['find_search_areas']
        if self.search_areas is None:
            self.load_search_areas()

        in_file_path = os.path.join(config['img_directory'], img_filename)
        data = ledsa.core.image_reading.read_img(in_file_path, channel=0)

        plt.figure(dpi=1200)
        ax = plt.gca()
        ledsa.data_extraction.step_1_functions.add_search_areas_to_plot(self.search_areas, ax, config)
        plt.imshow(data, cmap='Greys')
        plt.colorbar()
        out_file_path = os.path.join('plots', 'led_search_areas.plot.pdf')
        plt.savefig(out_file_path)

    # """
    # ------------------------------------
    # Step 2 - match LEDs to arrays
    # ------------------------------------
    # """

    def match_leds_to_led_arrays(self) -> None:
        """
        Analyze which LEDs belong to which LED line and save this mapping.
        """
        if self.search_areas is None:
            self.load_search_areas()
        self.line_indices = ledsa.data_extraction.step_2_functions.match_leds_to_led_arrays(self.search_areas,
                                                                                            self.config)
        ledsa.data_extraction.step_2_functions.generate_line_indices_files(self.line_indices)
        ledsa.data_extraction.step_2_functions.generate_labeled_led_arrays_plot(self.line_indices, self.search_areas)
        self.line_indices, merge = ledsa.data_extraction.step_2_functions.merge_led_arrays(self.line_indices,
                                                                                           self.config)
        if merge:
            ledsa.data_extraction.step_2_functions.generate_labeled_led_arrays_plot(self.line_indices,
                                                                                    self.search_areas,
                                                                                    filename_extension='_merge')
            ledsa.data_extraction.step_2_functions.generate_line_indices_files(self.line_indices,
                                                                               filename_extension='_merge')

    def load_line_indices(self) -> None:
        """
        Load LED line indices from the 'line_indices_{...}.csv' files.
        """
        if self.config['analyse_positions']['merge_led_arrays'] != 'None':
            num_of_arrays = len(self.config.get2dnparray('analyse_positions', 'merge_led_arrays', 'var'))
            file_extension = '_merge'
            print("ARRAY MERGE IS ACTIVE!")
        else:
            num_of_arrays = int(self.config['analyse_positions']['num_of_arrays'])
            file_extension = ''
        self.line_indices = []
        for i in range(num_of_arrays):
            file_path = os.path.join('analysis', f'line_indices_{i:03}{file_extension}.csv')
            self.line_indices.append(ledsa.core.file_handling.read_table(file_path, dtype='int'))

    # """
    # ------------------------------------
    # Step 3 - LED smoke analysis
    # ------------------------------------
    # """

    def process_image_data(self) -> None:
        """
        Process all the image data to detect changes in light intensity in the search areas across the images.
        Removes 'images_to_process.csv' file afterward.
        """
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
                print('image ', i + 1, '/', len(img_filenames), ' processed')

        os.remove('images_to_process.csv')

    def process_img_file(self, img_filename: str) -> None:
        """
        Process a single image file to extract relevant data. This is a workaround for pool.map.

        :param img_filename: The name of the image file to be processed.
        :type img_filename: str
        """
        img_id = ledsa.core.image_handling.get_img_id(img_filename)
        for channel in self.channels:
            img_data = ledsa.data_extraction.step_3_functions.generate_analysis_data(img_filename, channel,
                                                                                     self.search_areas,
                                                                                     self.line_indices,
                                                                                     self.config, self.fit_leds)
            ledsa.data_extraction.step_3_functions.create_fit_result_file(img_data, img_id, channel)
        print('Image {} processed'.format(img_id))

    def setup_step3(self) -> None:
        """
        Setup the third step of the data extraction process by creating 'image_infos_analysis.csv' and 'images_to_process.csv' files.
        """
        led.generate_image_infos_csv(self.config, build_analysis_infos=True)
        ledsa.data_extraction.step_3_functions.create_imgs_to_process_file()

    def setup_restart(self) -> None:
        """
        Setup a restart in case the data extraction process was interrupted earlier.
        """
        # if len(self.channels) > 1: #TODO: deactivated for testing
        #     print('Restart of a run currently only supports one channel. \nExiting...')
        #     exit(1)
        ledsa.data_extraction.step_3_functions.find_and_save_not_analysed_imgs(self.channels[0])
