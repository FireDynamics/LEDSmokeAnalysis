#!/usr/bin/env python

import os

import matplotlib.pyplot as plt
import numpy as np
from tqdm import tqdm

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
        self.search_areas = ledsa.core.file_handling.read_table(file_path, delim=',', dtype='int')

    def write_search_areas(self, reorder_leds=False) -> None:
        """
        Writes LED search areas to a CSV file.

        :param reorder_leds: A flag indicating whether the LED IDs have been reordered. Affects the header of the output file.
        :type reorder_leds: bool
        """
        out_file_path = os.path.join('analysis', 'led_search_areas.csv')
        header = 'LED id (reordered), pixel position x, pixel position y' if reorder_leds else ('LED id, pixel '
                                                                                                'position x, '
                                                                                                'pixel position y')
        np.savetxt(out_file_path, self.search_areas, delimiter=',',
                   header=header, fmt='%d')

    def find_search_areas(self) -> None:
        """
        Identify all LEDs in the reference image and define the areas where LEDs will be searched in the experiment images.
        """
        config = self.config['find_search_areas']
        in_file_path = os.path.join(config['img_directory'], config['img_name_string'].format(int(config['ref_img_id'])))
        channel = config['channel']
        search_area_radius = int(config['search_area_radius'])
        max_num_leds = int(config['max_num_leds'])
        pixel_value_percentile = float(config['pixel_value_percentile'])
        if channel == 'all':
            data, _ = ledsa.core.image_reading.read_img_array_from_raw_file(in_file_path, channel=0) # TODO: Channel to be removed here!
        else:
            channel = int(channel)
            data = ledsa.core.image_reading.read_channel_data_from_img(in_file_path, channel=channel)

        self.search_areas = ledsa.data_extraction.step_1_functions.find_search_areas(data, search_area_radius=search_area_radius, max_n_leds=max_num_leds, pixel_value_percentile=pixel_value_percentile)
        self.write_search_areas()
        self.plot_search_areas()
        ledsa.core.file_handling.remove_flag('reorder_leds')

    def plot_search_areas(self, reorder_leds=False) -> None:
        """
        Plot the identified LED search areas with their ID labels.

        :param reorder_leds: A flag indicating whether the LED IDs have been reordered. Affects the name of the output file.
        :type reorder_leds: bool
        """
        try:
            os.remove(os.path.join('plots', 'led_search_areas.plot_reordered.pdf'))
        except OSError:
            pass

        config = self.config['find_search_areas']
        if self.search_areas is None:
            self.load_search_areas()

        in_file_path = os.path.join(config['img_directory'], config['img_name_string'].format(int(config['ref_img_id'])))
        data = ledsa.core.image_reading.read_channel_data_from_img(in_file_path, channel=0)
        search_area_radius = int(config['search_area_radius'])
        plt.figure(dpi=1200)
        ax = plt.gca()
        ledsa.data_extraction.step_1_functions.add_search_areas_to_plot(self.search_areas, search_area_radius, ax)
        plt.imshow(data, cmap='Greys')
        plt.xlim(self.search_areas[:, 2].min() - 5 * search_area_radius, self.search_areas[:, 2].max() + 5 * search_area_radius)
        plt.ylim(self.search_areas[:, 1].max() + 5 * search_area_radius, self.search_areas[:, 1].min() - 5 * search_area_radius)
        plt.colorbar()
        plot_filename = 'led_search_areas.plot_reordered.pdf' if reorder_leds else 'led_search_areas.plot.pdf'
        out_file_path = os.path.join('plots', plot_filename)
        plt.savefig(out_file_path)
        plt.close()

    # """
    # ------------------------------------
    # Step 2 - match LEDs to arrays
    # ------------------------------------
    # """

    def match_leds_to_led_arrays(self) -> None:
        """
        Analyze which LEDs belong to which LED array and save this mapping.
        """
        if ledsa.core.file_handling.check_flag('reorder_leds'):
            exit("LED IDs have been reordered. Please run step S1 again before trying to match LEDs to LED lines!")
        else:
            if self.search_areas is None:
                self.load_search_areas()
            self.line_indices = ledsa.data_extraction.step_2_functions.match_leds_to_led_arrays(self.search_areas,
                                                                                            self.config)
            self.search_areas = ledsa.data_extraction.step_2_functions.reorder_search_areas(self.search_areas,
                                                                                            self.line_indices)
            self.write_search_areas(reorder_leds=True)
            self.line_indices = ledsa.data_extraction.step_2_functions.reorder_led_indices(self.line_indices)
            self.plot_search_areas(reorder_leds=True)
            print("LED IDs reordered successfully!")
            ledsa.core.file_handling.set_flag('reorder_leds')

            ledsa.data_extraction.step_2_functions.generate_led_array_indices_files(self.line_indices)
            self.plot_led_arrays()


        if self.config['analyse_positions']['merge_led_array_indices'] != 'None':
            self.line_indices = ledsa.data_extraction.step_2_functions.merge_indices_of_led_arrays(self.line_indices, self.config)
            self.plot_led_arrays(merge_led_arrays=True)

    def load_led_array_indices(self) -> None:
        """
        Load LED array indices from the 'led_array_indices_{...}.csv' files.
        """
        if self.config['analyse_positions']['merge_led_array_indices'] != 'None':
            num_arrays = len(self.config.get2dnparray('analyse_positions', 'merge_led_array_indices', 'var'))
            file_extension = '_merge'
            print("ARRAY MERGE IS ACTIVE!")
        else:
            num_arrays = int(self.config['analyse_positions']['num_arrays'])
            file_extension = ''
        self.line_indices = []
        for i in range(num_arrays):
            file_path = os.path.join('analysis', f'led_array_indices_{i:03}{file_extension}.csv')
            self.line_indices.append(ledsa.core.file_handling.read_table(file_path, dtype='int'))

    def plot_led_arrays(self, merge_led_arrays=False) -> None:
        """
        Plots the arrangement of LEDs as identified in the LED arrays and saves the plot as a PDF file.

        :param merge_led_arrays: A flag indicating whether LED arrays have been merged. Affects the naming of the output file.
        :type merge_led_arrays: bool
        """
        for i in range(len(self.line_indices)):
            plt.scatter(self.search_areas[self.line_indices[i], 2],
                        -self.search_areas[self.line_indices[i], 1],
                        s=0.1, label='LED Array {}'.format(i))
        plt.legend()
        plt.xticks([])
        plt.yticks([])
        plot_filename = 'led_arrays_merged.pdf' if merge_led_arrays else 'led_arrays.pdf'
        out_file_path = os.path.join('plots', plot_filename)
        plt.savefig(out_file_path)
        plt.close()

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
            self.load_led_array_indices()

        img_filenames = ledsa.core.file_handling.read_table('images_to_process.csv', dtype=str)
        num_cores = int(config['num_cores'])
        if num_cores > 1:
            from multiprocessing import Pool
            print('images are getting processed, this may take a while')
            with Pool(num_cores) as p:
                for _ in tqdm(p.imap(self.process_img_file, img_filenames), total=len(img_filenames), desc="Processing images", unit="image"):
                    pass
        else:
            for img_filename in tqdm(img_filenames, desc="Processing images", unit="image"):
                self.process_img_file(img_filename)

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
