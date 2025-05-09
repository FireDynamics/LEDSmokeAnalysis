import configparser as cp
from datetime import datetime, timedelta

import numpy as np

from ledsa.core.image_reading import get_exif_entry


class ConfigData(cp.ConfigParser):
    """
    Class responsible for handling configuration data related to LEDSA.

    """
    def __init__(self, load_config_file=True, img_directory=None, window_radius=10, pixel_value_percentile=99.875, channel=0,
                 max_num_of_leds=None, num_of_arrays=None, num_of_cores=1, date=None, start_time=None, time_img=None, time_ref_img_time=None,
                 time_diff_to_image_time=None, img_name_string=None, img_number_overflow=None,
                 first_img_num_experiment=None, last_img_num_experiment=None, reference_img=None, ignore_indices=None,
                 line_edge_indices=None, line_edge_coordinates=None, first_img_num_analysis=None,
                 last_img_num_analysis=None, skip_imgs=0, skip_leds=0, merge_led_arrays=None):  # TODO: merge LED arrays
        """
        :param load_config_file: Determines whether to load the config file on initialization. Defaults to True.
        :type load_config_file: bool
        :param img_directory: Path to the image directory. Defaults to None.
        :type img_directory: str or None
        :param window_radius: Pixel radius of ROI assigned to each LED. Defaults to 10.
        :type window_radius: int
        :param pixel_value_percentile: Threshold percentile of pixel value for LED detection. Defaults to 99.875.
        :type pixel_value_percentile: float
        :param channel: Camera channel on which the LED should be detected. Defaults to 0.
        :type channel: int
        :param max_num_of_leds: Maximum number of LEDs to be detected on the reference image. Defaults to None
        :type max_num_of_leds: int
        :param num_of_arrays: Number of LED arrays. Defaults to None.
        :type num_of_arrays: int or None
        :param num_of_cores: Number of CPU cores for (multicore) processing. If greater than 1, multicore processing is applied. Defaults to 1.
        :type num_of_cores: int
        :param date: Date of the experiment. Defaults to None. #TODO: format
        :type date: str or None
        :param start_time: Start time for the experiment. Will be calculated from first image if None. Defaults to None. #TODO: format
        :type start_time: str or None
        :param time_img: ID of image with a visible clock used for synchronization. Defaults to None.
        :type time_img: str or None
        :param time_ref_img_time: Time shown on the time_img. Defaults to None. #TODO: format
        :type time_ref_img_time: str or None
        :param time_diff_to_image_time: Time difference in seconds to the actual image time. Defaults to None.
        :type time_diff_to_image_time: int or None
        :param img_name_string: Naming convention of the image files. Defaults to None.
        :type img_name_string: str or None
        :param img_number_overflow: Maximal number an image file ID can have. Defaults to None.
        :type img_number_overflow: int or None
        :param first_img_num_experiment: ID of the first image of the experiment. Defaults to None.
        :type first_img_num_experiment: int or None
        :param last_img_num_experiment: ID of the last image of the experiment. Defaults to None.
        :type last_img_num_experiment: int or None
        :param reference_img: Reference image used to identify and label the LEDs. Defaults to None.
        :type reference_img: str or None
        :param ignore_indices: IDs of LEDs to ignore during analysis. Defaults to None.
        :type ignore_indices: list[int] or None
        :param line_edge_indices: Pairs of LED IDs of the edges of each LED array. Defaults to None.
        :type line_edge_indices: list[int] or None
        :param line_edge_coordinates: Physical positions of each LED array edges given in line_edge_indices. Defaults to None.
        :type line_edge_coordinates: list[float] or None
        :param first_img_num_analysis: ID of the first image for analysis. Defaults to None. #TODO: rename analysis to extraction? Otherwise confusing
        :type first_img_num_analysis: int or None
        :param last_img_num_analysis: ID of the last image for analysis. Defaults to None.
        :type last_img_num_analysis: int or None
        :param skip_imgs: Number of images to skip during analysis. Defaults to 0.
        :type skip_imgs: int
        :param skip_leds: Only consider LEDs with ID divisible by skip_leds + 1. Defaults to 0.
        :type skip_leds: int
        :param merge_led_arrays: Flag to merge LED arrays for analysis. Defaults to None. TODO: not a flag but list of arrays to merge?
        :type merge_led_arrays: bool or None
        """
        cp.ConfigParser.__init__(self, allow_no_value=True)
        if load_config_file:
            self.load()
        else:
            self['DEFAULT'] = {}
            self.set('DEFAULT', '# Variables used in multiple parts of LEDSA.')
            self.set('DEFAULT', '   # Directory where the images for one experiment are stored.')
            self['DEFAULT']['   img_directory'] = str(img_directory)
            self.set('DEFAULT', '   # String representing the naming convention of the image files.')
            self['DEFAULT']['   img_name_string'] = str(img_name_string)
            self.set('DEFAULT', '   # Maximum number of image counter.')
            self['DEFAULT']['   img_number_overflow'] = str(img_number_overflow)
            self.set('DEFAULT', '# Specify which images ar part of the experiment by number.')
            self['DEFAULT']['   first_img_num'] = str(first_img_num_experiment)
            self['DEFAULT']['   last_img_num'] = str(last_img_num_experiment)
            self.set('DEFAULT', '   # Number of CPUs, multicore processing is applied if > 1.')
            self['DEFAULT']['   num_of_cores'] = str(num_of_cores)
            self.set('DEFAULT', '')
            self.set('DEFAULT', '# Variables used to calculate the timeline of the experiment.')
            self.set('DEFAULT', '   # Date of the experiment.')
            self['DEFAULT']['   date'] = str(date)
            self.set('DEFAULT', '   # Beginning of the experiment, will be calculated from image with first_img_num if None.')
            self['DEFAULT']['   start_time'] = str(start_time)
            self.set('DEFAULT', '   # Image name with a clock to calculate the offset of the camera time. Can be None,')
            self.set('DEFAULT', '   # if time_diff_to_img_time in seconds is given.')
            self['DEFAULT']['   time_img'] = str(time_img)
            self.set('DEFAULT', '   # Time shown on the time_img')
            self['DEFAULT']['   time_ref_img_time'] = str(time_ref_img_time)
            self['DEFAULT']['   exif_time_infront_real_time'] = str(time_diff_to_image_time)

            self['find_search_areas'] = {}
            self.set('find_search_areas', '# Variables used to find the pixel positions of every led.')
            self.set('find_search_areas', '   # Name of reference image used to find and label the leds.')
            self['find_search_areas']['   reference_img'] = str(reference_img)
            self.set('find_search_areas', '   # Threshold percentile of pixel value to find LEDs on the reference image.')
            self['find_search_areas']['   pixel_value_percentile'] = str(pixel_value_percentile)
            self.set('find_search_areas', '   # Camera channel on which LEDs should be found, "all" only works for RAW '
                                          'image files')
            self['find_search_areas']['   channel'] = str(channel)
            self.set('find_search_areas', '   # Radius of pixels assigned to each LED')
            self['find_search_areas']['   window_radius'] = str(window_radius)
            self.set('find_search_areas', '   # Maximum number of LEDs to detect on the reference image')
            self['find_search_areas']['   max_num_of_leds'] = str(max_num_of_leds)

            self['analyse_positions'] = {}
            self.set('analyse_positions', '# Variables used to find the physical positions of every led')
            self.set('analyse_positions', '   # Number of LED arrays')
            self['analyse_positions']['   num_of_arrays'] = str(num_of_arrays)
            self.set('analyse_positions', '# IDs of LEDs to be ignored in further analysis')
            self['analyse_positions']['   ignore_indices'] = str(ignore_indices)
            self.set('analyse_positions', '   # Pairs of led IDs of the edges of each led array')
            self['analyse_positions']['   line_edge_indices'] = str(line_edge_indices)
            self.set('analyse_positions', '   # Six coordinates per led array representing the physical positions of '
                                          'the')
            self.set('analyse_positions', '   # edges given in line_edge_indices.')
            self['analyse_positions']['   line_edge_coordinates'] = str(line_edge_coordinates)
            self.set('analyse_positions', '   # IDs of the LED arrays to merge for further analysis.')
            self['analyse_positions']['   merge_led_arrays'] = str(merge_led_arrays)

            self['analyse_photo'] = {}
            self.set('analyse_photo', '# Variables used for the final fitting of the intensity function.')
            self.set('analyse_photo', ' ')
            self.set('analyse_photo', '# Specify which images are used for analysis by image numbers/')
            self['analyse_photo']['   first_img_num'] = str(first_img_num_analysis)
            self['analyse_photo']['   last_img_num'] = str(last_img_num_analysis)
            self.set('analyse_photo', '   # In the analysis, only images are considered with id dividable by skip_imgs + 1.')
            self['analyse_photo']['   skip_imgs'] = str(skip_imgs)
            self.set('analyse_photo', '   # In the analysis, only LEDs are considered with id dividable by skip_leds + 1.')
            self['analyse_photo']['   skip_leds'] = str(skip_leds)

            with open('config.ini', 'w') as configfile:
                self.write(configfile)
            print('config.ini created')

    def load(self) -> None:
        """
        Loads the configuration data from 'config.ini' file.

        Raises:
            FileNotFoundError: If 'config.ini' is not found in the working directory.
        """
        try:
            self.read_file(open('config.ini'))
        except FileNotFoundError:
            print('config.ini not found in working directory! Please create it with argument "--config".')
        print('config.ini loaded')

    def save(self) -> None:
        """
        Saves the current configuration to 'config.ini' file.
        """
        with open('config.ini', 'w') as configfile:
            self.write(configfile)
        print('config.ini saved')

    def get2dnparray(self, section, option, num_col=2, dtype=int) -> np.ndarray:
        """
        Extracts a 2D NumPy array from multi-line/ multi-column inputs the configuration.

        :param section: Section name in the configuration.
        :type section: str
        :param option: Option name within the section.
        :type option: str
        :param num_col: Number of columns expected in the array. Defaults to 2.
        :type num_col: int or 'var'
        :param dtype: Data type of the elements in the output array. Defaults to int.
        :type dtype: type
        :return: 2D array from the configuration.
        :rtype: :class:`numpy.ndarray` or None
        """
        if self[section][option] == 'None':
            return None
        if num_col == 'var':
            indices_array = [i for i in self[section][option].split('\n')]
            indices_array = [i for i in indices_array if i]  # Remove empty strings
            indices = []
            for i in indices_array:
                indices.append([dtype(j) for j in i.split()])
            return indices

        indices_tmp = [dtype(i) for i in self[section][option].split()]
        indices = np.zeros((len(indices_tmp) // num_col, num_col), dtype=dtype)
        for i in range(len(indices_tmp) // num_col):
            indices[i][:] = indices_tmp[num_col * i:num_col * i + num_col]
            # indices[i][1] = indices_tmp[2 * i + 1]
        return indices

    def get_datetime(self, option='start_time') -> None:
        """
        Retrieves a datetime object from the configuration based on a provided option.

        :param option: The option name for which the datetime is required. Defaults to 'start_time'.
        :type option: str
        :return: Datetime object corresponding to the option.
        :rtype: datetime.datetime
        """
        time = self['DEFAULT'][option]
        date = self['DEFAULT']['date']
        try:
            date_time = _get_datetime_from_str(date, time)
        except Exception as e:
            print(e)
            exit(1)
        else:
            return date_time

    def in_img_dir(self) -> None:
        """
        Prompts the user for the image directory path and updates the configuration.
        """
        self['DEFAULT']['img_directory'] = input('Please give the path where the images are stored: ')

    def in_first_img_num_experiment(self) -> None:
        """
        Prompts the user for the ID of the first image of the experiment and updates the configuration.
        """
        self['DEFAULT']['first_img_num'] = input('Please give the number of the first image of the experiment: ')

    def in_last_img_num_experiment(self) -> None:
        """
        Prompts the user for the ID of the last image of the experiment and updates the configuration.
        """
        self['DEFAULT']['last_img_num'] = input('Please give the number of the last image image of the experiment: ')

    def in_ref_img(self) -> None:
        """
        Prompts the user for the reference image name and updates the configuration.
        """
        self['find_search_areas']['reference_img'] = input('Please give the name of the reference image, from where the'
                                                           ' led positions are calculated and which will be the start '
                                                           'of the experiment time calculation: ')

    def in_max_num_of_leds(self) -> None:
        """
        Prompts the user for the maximum number of LEDs to be detected on the reference image.
        """
        self['find_search_areas']['max_num_of_leds'] = input('Please give the maximum number of LEDs to be detected on the reference '
                                            'image: ')

    def in_time_img(self) -> None:
        """
        Prompts the user for the time reference image name and updates the configuration.
        """
        self['DEFAULT']['time_img'] = input('Please give the name of the time reference image, the image where a clock '
                                            'is visible, to synchronise multiple cameras in one experiment: ')

    def in_num_of_arrays(self) -> None:
        """
        Prompts the user for the number of LED lines and updates the configuration.
        """
        self['analyse_positions']['num_of_arrays'] = input('Please give the number of LED lines: ')

    def in_time_diff_to_img_time(self) -> None:
        """
        Update the configuration with the time difference between the reference image's timestamp and the real time.
        If the 'time_ref_img_time' is not set, prompts the user to provide the time shown on the clock in the time reference image.
        """
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

    def in_img_name_string(self) -> None:
        """
        Prompts the user for the image naming convention and updates the configuration.
        """
        self['DEFAULT']['img_name_string'] = input(
            'Please give the name structure of the images in the form img{}.extension where '
            '{} denotes the increasing number of the image files: ')

    def in_img_number_overflow(self) -> None:
        """
        Prompts the user for the maximal image file ID and updates the configuration.
        """
        self['DEFAULT']['img_number_overflow'] = input(
            'Please give the maximal number an image file can have (typically 9999): ')

    def in_first_img_num_analysis(self) -> None:
        """
        Prompts the user for the number of the first image for analysis and updates the configuration.
        """
        ...
        self['analyse_photo']['first_img_num'] = input('Please give the number of the first image file to be analysed: ')

    def in_last_img_num_analysis(self) -> None:
        """
        Prompts the user for the number of the last image for analysis and updates the configuration.
        """
        self['analyse_photo']['last_img_num'] = input('Please give the number of the last image file to be analysed: ')

    def in_line_edge_indices(self) -> None:
        """
        Prompt the user to input the labels of the topmost and bottommost LED of each array.

        Note:
            The edges of the LED arrays are required. Labels for each array are separated by whitespace.
        """
        print('The edges of the LED arrays are needed. Please enter the labels of the top most and bottom most LED of '
              'each array. Separate the two labels with a whitespace.')
        labels = str()
        for i in range(int(self['analyse_positions']['num_of_arrays'])):
            line = input(str(i) + '. array: ')
            labels += '\t    ' + line + '\n'
        self['analyse_positions']['line_edge_indices'] = '\n' + labels

    def in_line_edge_coordinates(self) -> None:
        """
        Prompt the user to input the physical coordinates of the topmost and bottommost LED of each array.

        Note:
            The coordinates should correspond to the order of the line edge indices.
            Coordinates are given in form X Y Z X Y Z and are separated by whitespace.
        :raises ValueError: If any of the inputs are not formatted correctly.
        """
        print('Please enter the coordinates of the top most and bottom most LED of each array corresponding to the '
              'order of the line edge indices. Separate the two coordinates with a whitespace.')
        coordinates = str()
        for i in range(int(self['analyse_positions']['num_of_arrays'])):
            line = input(str(i) + '. array: ')
            coordinates += '\t    ' + line + '\n'
        self['analyse_positions']['line_edge_coordinates'] = '\n' + coordinates

    # get the start time from the first experiment image
    def get_start_time(self) -> None:
        """
        Retrieve and set the start time from the first experiment image.

        Extracts the EXIF DateTimeOriginal from the image and then computes the start time.
        Updates the 'DEFAULT' key with the 'start_time' computed.

        """
        exif_entry = get_exif_entry(self['DEFAULT']['img_directory'] + self['DEFAULT']['img_name_string'].format(
            self['DEFAULT']['first_img_num']), 'EXIF DateTimeOriginal')
        date, time_meta = exif_entry.split(' ')
        time_img = _get_datetime_from_str(date, time_meta)
        start_time = time_img - timedelta(seconds=self['DEFAULT'].getint('exif_time_infront_real_time'))
        self['DEFAULT']['start_time'] = start_time.strftime('%H:%M:%S')


def _get_datetime_from_str(date: str, time: str) -> datetime:
    """
    Convert a date and time string into a datetime object.

    The function can handle two formats:
    1. '%Y:%m:%d %H:%M:%S' - standard format with colons in the date.
    2. '%d.%m.%Y %H:%M:%S' - format with periods in the date.

    :param date: The date string.
    :type date: str
    :param time: The time string.
    :type time: str
    :return: The corresponding datetime object.
    :rtype: datetime.datetime
    """
    if date.find(":") != -1:
        date_time = datetime.strptime(date + ' ' + time, '%Y:%m:%d %H:%M:%S')
    else:
        date_time = datetime.strptime(date + ' ' + time, '%d.%m.%Y %H:%M:%S')
    return date_time


if __name__ == 'main':
    ConfigData()
