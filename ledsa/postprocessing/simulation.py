import glob
import os

import numpy as np
import pandas as pd

from ledsa.core.image_reading import read_channel_data_from_img
from ledsa.core.ConfigData import ConfigData
from ledsa.analysis.ConfigDataAnalysis import ConfigDataAnalysis


class SimData:
    def __init__(self, path_simulation: str, read_all=True, remove_duplicates=False, load_config_params=True):
        """
        Initializes SimData with simulation and image analysis settings.

        :param path_simulation: Directory containing simulation data.
        :type path_simulation: str
        :param path_images: Directory containing experimental images, defaults to None.
        :type path_images: str, optional
        :param read_all: If True, reads all data on initialization, defaults to True.
        :type read_all: bool, optional
        :param remove_duplicates: If True, removes duplicate entries from analysis, defaults to False.
        :type remove_duplicates: bool, optional
        :param load_config_params: If True, loads parameters from config files, defaults to True.
        :type load_config_params: bool, optional
        :raises ValueError: If path_images is not provided when required for image analysis.
        """
        self.solver = None
        self.average_images = False
        self.camera_channels = [0]
        self.num_ref_images = None
        self.n_layers = None
        self.search_area_radius = None
        self.path_images = None


        if load_config_params:
            os.chdir(path_simulation)
            # Read configuration from config.ini and config_analysis.ini
            self.config = ConfigData()
            self.config_analysis = ConfigDataAnalysis()

            # Get parameters from config file
            self.search_area_radius = self.config['find_search_areas']['search_area_radius']
            self.path_images = self.config['find_search_areas']['img_directory']

            # Get parameters from config_analysis file
            self.solver = self.config_analysis['DEFAULT']['solver']
            self.average_images = self.config_analysis.getboolean('DEFAULT', 'average_images')
            self.camera_channels = self.config_analysis.get_list_of_values('DEFAULT', 'camera_channels', dtype=int)
            self.num_ref_images = int(self.config_analysis['DEFAULT']['num_ref_images'])
            self.n_layers = int(self.config_analysis['model_parameters']['num_layers'])
            domain_bounds = self.config_analysis.get_list_of_values('model_parameters', 'domain_bounds', dtype=float)
            self.bottom_layer_height = domain_bounds[0]
            self.top_layer_height = domain_bounds[1]

        self.path_simulation = path_simulation
        self.led_info_path = os.path.join(self.path_simulation, 'analysis', 'led_search_areas_with_coordinates.csv')

        if self.average_images == True:
            image_infos_file = 'analysis/image_infos_analysis_avg.csv'
        else:
            image_infos_file = 'analysis/image_infos_analysis.csv'
        self.image_info_df = pd.read_csv(os.path.join(self.path_simulation, image_infos_file))

        if read_all == True:
            self.read_all()
        if remove_duplicates == True:
            self.remove_duplicate_heights()
        else:
            self.ch0_ledparams = None
            self.ch1_ledparams = None
            self.ch2_ledparams = None
            self.ch0_extcos = None
            self.ch1_extcos = None
            self.ch2_extcos = None

    height_from_layer = lambda self, layer: -1 * (
            layer / self.n_layers * (self.top_layer_height - self.bottom_layer_height) - self.top_layer_height)
    layer_from_height = lambda self, height: int(
        (self.top_layer_height - height) / (self.top_layer_height - self.bottom_layer_height) * self.n_layers)

    def set_timeshift(self, timedelta: int):
        """
        Sets the time shift for the experiment's start time.

        :param timedelta: Time shift in seconds.
        :type timedelta: int
        """
        self.all_extco_df.index += timedelta

    def _get_ledparams_df_from_path(self, channel: int) -> pd.DataFrame:
        """
        Reads experimental parameters from a binary HDF5 table based on color channel.

        :param channel: The color channel index to read the parameters for.
        :type channel: int
        :return: DataFrame with LED parameters.
        :rtype: pd.DataFrame
        """
        if self.average_images == True:
            file = os.path.join(self.path_simulation, 'analysis', f'channel{channel}', 'all_parameters_avg.h5')
        else:
            file = os.path.join(self.path_simulation, 'analysis', f'channel{channel}', 'all_parameters.h5')
        table = pd.read_hdf(file, key='channel_values')
        time = self.image_info_df['Experiment_Time[s]'].astype(int)
        table = table.merge(time, left_on='img_id', right_index=True)
        table.set_index(['Experiment_Time[s]', 'led_id'], inplace=True)
        self.led_heights = table['height']
        return table

    def _get_extco_df_from_path(self):
        """
        Read all extinction coefficients from the simulation dir and put them in the all_extco_df.
        """
        extco_list = []
        files_list = glob.glob(
            os.path.join(self.path_simulation, 'analysis', 'extinction_coefficients', self.solver, f'extinction_coefficients*.csv'))
        for file in files_list:
            file_df = pd.read_csv(file, skiprows=4)
            channel = int(file.split('channel_')[1].split('_')[0])
            led_array = int(file.split('array_')[1].split('.')[0])
            n_layers = len(file_df.columns)
            time = self.image_info_df['Experiment_Time[s]'].astype(int)
            file_df = file_df.merge(time, left_index=True, right_index=True)
            file_df.set_index('Experiment_Time[s]', inplace=True)
            iterables = [[channel], [led_array], [i for i in range(0, n_layers)]]
            file_df.columns = pd.MultiIndex.from_product(iterables, names=["Channel", "LED Array", "Layer"])
            extco_list.append(file_df)
        self.all_extco_df = pd.concat(extco_list, axis=1)
        self.all_extco_df.sort_index(ascending=True, axis=1, inplace=True)
        self.all_extco_df = self.all_extco_df[
            ~self.all_extco_df.index.duplicated(keep='first')]  # Remove duplicate times

    def read_led_params(self):
        """Read led parameters for all color channels from the simulation path"""
        self.ch0_ledparams_df = self._get_ledparams_df_from_path(0)
        self.ch1_ledparams_df = self._get_ledparams_df_from_path(1)
        self.ch2_ledparams_df = self._get_ledparams_df_from_path(2)

    def read_all(self):
        """Read led parameters and extionciton coefficients for all color channels from the simulation path"""
        self.read_led_params()
        self._get_extco_df_from_path()

    def remove_duplicate_heights(self):
        """    Removes duplicate height entries for each LED parameter DataFrame across all colorchannels."""
        self.ch0_ledparams_df = self.ch0_ledparams_df.groupby(['Experiment_Time[s]', 'height']).last()
        self.ch1_ledparams_df = self.ch1_ledparams_df.groupby(['Experiment_Time[s]', 'height']).last()
        self.ch2_ledparams_df = self.ch2_ledparams_df.groupby(['Experiment_Time[s]', 'height']).last()

    def get_extco_at_time(self, channel: int, time: int, yaxis='layer', window=1, smooth='ma') -> pd.DataFrame:
        """
        Retrieves a DataFrame containing smoothed extinction coefficients at a specific time.

        This method extracts the extinction coefficients for a specified channel and time,
        applies smoothing over time using either a moving average or median based on the specified window size.
        It restructures the DataFrame to have layers or heights as indices and LED arrays as columns.
        The method first selects the extinction coefficients for the specified channel. It then applies the specified smoothing
        operation across a defined window of timesteps. The resulting data is then pivoted to organize extinction coefficients
        by LED array and layer or height, depending on the 'yaxis' parameter, providing a structured view suitable for analysis or visualization.

        :param channel: The channel index to extract extinction coefficients for.
        :type channel: int
        :param time: The time at which to extract extinction coefficients.
        :type time: int
        :param yaxis: Determines whether the y-axis of the returned DataFrame should represent 'layer' or 'height'. Defaults to 'layer'.
        :type yaxis: str, optional
        :param window: The window size for the moving average or median smoothing. Defaults to 1.
        :type window: int, optional
        :param smooth: The smoothing method to use, either 'ma' for moving average or 'median' for median. Defaults to 'ma'.
        :type smooth: str, optional
        :return: A pandas DataFrame with the smoothed extinction coefficients. Indices represent either layers or heights, and columns represent LED arrays.
        :rtype: pd.DataFrame
        """
        ch_extco_df = self.all_extco_df.xs(channel, level=0, axis=1)
        if smooth == 'median':
            ma_ch_extco_df = ch_extco_df.iloc[::-1].rolling(window=window, closed='left').median().iloc[::-1]
        else:
            ma_ch_extco_df = ch_extco_df.iloc[::-1].rolling(window=window, closed='left').mean().iloc[::-1]
        ma_ch_extco_df = ma_ch_extco_df.loc[time, :]
        ma_ch_extco_df = ma_ch_extco_df.reset_index().pivot(columns='LED Array', index='Layer')
        ma_ch_extco_df.columns = ma_ch_extco_df.columns.droplevel()
        ma_ch_extco_df.index = range(ma_ch_extco_df.shape[0])
        if yaxis == 'layer':
            ma_ch_extco_df.index.names = ["Layer"]
        elif yaxis == 'height':
            ma_ch_extco_df.index = [round(self.height_from_layer(layer),2) for layer in ma_ch_extco_df.index]
            ma_ch_extco_df.index.names = ["Height / m"]
        return ma_ch_extco_df

    def get_extco_at_led_array(self, channel: int, led_array_id: int, yaxis='layer', window=1) -> pd.DataFrame:
        """
        Retrieves a DataFrame containing smoothed extinction coefficients for a specific LED array.

        This method extracts the extinction coefficients for a specified channel and LED array, applies a moving average smoothing over time based on the specified window size, and restructures the DataFrame to have experimental time as the index and layers (or heights) as columns.
        The method first selects the relevant extinction coefficients for the specified channel and LED array. It then applies a moving average smoothing operation across a defined window of time steps. The resulting data is organized such that the experimental time is the index, providing a structured view suitable for analysis or visualization. Depending on the 'yaxis' parameter, the columns of the resulting DataFrame are labeled as either layers or converted to heights.


        :param channel: The channel index from which to extract extinction coefficients.
        :type channel: int
        :param led_array_id: The ID for which to extract extinction coefficients.
        :type led_array_id: int
        :param yaxis: Determines whether the DataFrame's columns should represent 'layer' or 'height'. The default is 'layer', which uses the layer numbers as column names. If set to 'height', the column names are converted to the corresponding heights based on layer numbers.
        :type yaxis: str, optional
        :param window: The window size for the moving average smoothing. The default is 1, which means no smoothing is applied. A window greater than 1 will smooth the data over the specified number of time steps.
        :type window: int, optional
        :return: A pandas DataFrame with the smoothed extinction coefficients. The index represents the experimental time, and the columns represent either layers or heights, depending on the 'yaxis' parameter.
        :rtype: pd.DataFrame
        """
        ch_extco_df = self.all_extco_df.xs(channel, level=0, axis=1).xs(led_array_id, level=0, axis=1)
        ma_ch_extco_df = ch_extco_df.rolling(window=window, closed='right').mean().shift(-int(window / 2) + 1)
        # ma_ch_extco_df = ch_extco_df.iloc[::-1].rolling(window=window, closed='left').mean().iloc[::-1]

        if yaxis == 'layer':
            ma_ch_extco_df.columns.names = ["Layer"]
        elif yaxis == 'height':
            ma_ch_extco_df.columns = [self.height_from_layer(layer) for layer in ma_ch_extco_df.columns]
            ma_ch_extco_df.columns.names = ["Height / m"]
        return ma_ch_extco_df

    def get_extco_at_layer(self, channel: int, layer: int, window=1) -> pd.DataFrame:
        """
        Retrieves a DataFrame containing smoothed extinction coefficients for a specified layer.
        This method extracts the extinction coefficients for a given channel and layer, then applies a moving average smoothing over the specified window of time.
        The result is a DataFrame with experimental time as the index and LED arrays as the columns, providing a time series of extinction coefficients at the specified layer.

        :param channel: The channel index from which to extract extinction coefficients. Each channel represents a different set of measurements or sensor readings.
        :type channel: int
        :param layer: The layer number for which to extract extinction coefficients. Layers correspond to different vertical positions or depths in the experimental setup.
        :type layer: int
        :param window: The window size for the moving average smoothing. The default is 1, implying no smoothing. A larger window size will average the data over more time points, smoothing out short-term fluctuations.
        :type window: int, optional
        :return: A pandas DataFrame with smoothed extinction coefficients. The DataFrame's index represents experimental time, and its columns represent different LED arrays within the specified layer.
        :rtype: pd.DataFrame
        """
        ch_extco_df = self.all_extco_df.xs(channel, level=0, axis=1).xs(layer, level=1, axis=1)
        ma_ch_extco_df = ch_extco_df.rolling(window=window, closed='right').mean().shift(-int(window / 2) + 1)
        # ma_ch_extco_df = ch_extco_df.iloc[::-1].rolling(window=window, closed='left').mean().iloc[::-1]

        return ma_ch_extco_df

    def get_extco_at_height(self, channel: int, height: float, window=1) -> pd.DataFrame:
        """
        Retrieves a DataFrame containing smoothed extinction coefficients for a specified layer at given height.
        This method extracts the extinction coefficients for a given channel and layer, then applies a moving average smoothing over the specified window of time.
        The result is a DataFrame with experimental time as the index and LED arrays as the columns, providing a time series of extinction coefficients at the specified height.

        :param channel: The channel index from which to extract extinction coefficients. Each channel represents a different set of measurements or sensor readings.
        :type channel: int
        :param height: The height for which to extract extinction coefficients.
        :type height: float
        :param window: The window size for the moving average smoothing. The default is 1, implying no smoothing. A larger window size will average the data over more time points, smoothing out short-term fluctuations.
        :type window: int, optional
        :return: A pandas DataFrame with smoothed extinction coefficients. The DataFrame's index represents experimental time, and its columns represent different LED arrays within the specified height.
        :rtype: pd.DataFrame
        """
        layer = self.layer_from_height(height)
        return self.get_extco_at_layer(channel, layer, window)

    def get_ledparams_at_led_array(self, channel: int, led_array_id: int, param='sum_col_val', yaxis='led_id', window=1,
                                   n_ref=None) -> pd.DataFrame:
        """
        Retrieves a DataFrame containing normalized LED parameters for a specific LED array, optionally smoothed over
        time. This method selects LED parameter data for a given channel and LED array. It normalizes the data based on
        the average of the first `n_ref` entries (if `n_ref` is not False), and applies a moving average smoothing
        over the specified window of time. The result is a DataFrame with experimental time as the index and LED
        identifiers (or heights) as the columns, depending on the `yaxis` parameter.

         :param channel: The channel index from which to extract LED parameters. Channels typically represent different sensor readings or experimental conditions.
         :type channel: int
         :param led_array_id: The ID of the LED array for which to extract LED parameters. LED arrays may represent different spatial locations or orientations in the experimental setup.
         :type led_array_id: int
         :param param: The specific LED parameter to extract and analyze, such as 'sum_col_val'. Defaults to 'sum_col_val'.
         :type param: str, optional
         :param yaxis: Determines the labeling of the DataFrame's columns, either 'led_id' for LED identifiers or 'height' for physical heights. Defaults to 'led_id'.
         :type yaxis: str, optional
         :param window: The window size for the moving average smoothing. A value of 1 implies no smoothing. A larger window size averages the data over more time points, reducing short-term fluctuations.
         :type window: int, optional
         :param n_ref: The number of initial entries to average for normalization. If set to False, absolute values are returned without normalization. Defaults to 10.
         :type n_ref: int or bool, optional
         :return: A pandas DataFrame with normalized (and optionally smoothed) LED parameters. The index represents experimental time, and columns represent LED identifiers or heights.
         :rtype: pd.DataFrame
         """
        if channel == 0:
            led_params = self.ch0_ledparams_df
        elif channel == 1:
            led_params = self.ch1_ledparams_df
        elif channel == 2:
            led_params = self.ch2_ledparams_df
        index = 'height' if yaxis == 'height' else 'led_id'
        led_params = led_params.reset_index().set_index(['Experiment_Time[s]', index])
        ii = led_params[led_params['led_array_id'] == led_array_id][[param]]
        if n_ref == False:
            rel_i = ii
        else:
            # Use self.num_ref_images if n_ref is None
            n_ref = self.num_ref_images if n_ref is None else n_ref
            i0 = ii.groupby([index]).agg(lambda g: g.iloc[0:n_ref].mean())
            rel_i = ii / i0

        rel_i = rel_i.reset_index().pivot(columns=index, index='Experiment_Time[s]')
        rel_i.columns = rel_i.columns.droplevel()
        # rel_i_ma = rel_i.iloc[::-1].rolling(window=window, closed='left').mean().iloc[::-1]
        rel_i_ma = rel_i.rolling(window=window, closed='right').mean().shift(-int(window / 2) + 1)

        return rel_i_ma

    def get_ledparams_at_led_id(self, channel: int, led_id: int, param='sum_col_val', window=1,
                                n_ref=None) -> pd.DataFrame:
        """
        Retrieves a DataFrame containing normalized LED parameters for a specific LED ID, optionally smoothed over time.

        This method selects LED parameter data for a given channel and LED ID, normalizes the data based on the average of the first `n_ref` entries (if `n_ref` is not False), and applies a moving average smoothing over the specified window of time. The result is a DataFrame with experimental time as the index, providing a time series of the specified LED parameter for the selected LED.

        :param channel: The channel index from which to extract LED parameters. Channels typically represent different sets of measurements or sensor readings.
        :type channel: int
        :param led_id: The identifier of the LED for which to extract parameters. This ID is used to select the specific LED's data from the dataset.
        :type led_id: int
        :param param: The specific LED parameter to extract and analyze, such as 'sum_col_val'. This parameter determines which aspect of the LED's data is analyzed. Defaults to 'sum_col_val'.
        :type param: str, optional
        :param window: The window size for the moving average smoothing. A value of 1 implies no smoothing, while a larger window size averages the data over more time points, thereby smoothing out short-term fluctuations.
        :type window: int, optional
        :param n_ref: The number of initial entries to average for normalization. If set to False, absolute values are returned without normalization. This parameter allows for the normalization of data to account for initial conditions or baseline measurements. Defaults to 10.
        :type n_ref: int or bool, optional
        :return: A pandas DataFrame with normalized (and optionally smoothed) LED parameters for the specified LED ID. The DataFrame's index represents experimental time.
        :rtype: pd.DataFrame

        This method enables the analysis of temporal variations in specific LED parameters for an individual LED, facilitating the examination of changes or trends in the LED's behavior over the course of an experiment. By normalizing and smoothing the data, the method helps reveal underlying patterns that may not be immediately apparent from the raw data.
        """
        if channel == 0:
            led_params = self.ch0_ledparams_df
        elif channel == 1:
            led_params = self.ch1_ledparams_df
        elif channel == 2:
            led_params = self.ch2_ledparams_df
        led_params = led_params.reset_index().set_index(['Experiment_Time[s]'])
        ii = led_params[led_params['led_id'] == led_id][[param]]
        if n_ref == False:
            rel_i = ii
        else:
            # Use self.num_ref_images if n_ref is None
            n_ref = self.num_ref_images if n_ref is None else n_ref
            i0 = ii.iloc[0:n_ref].mean()
            rel_i = ii / i0
        rel_i_ma = rel_i.rolling(window=window, closed='right').mean().shift(-int(window / 2) + 1)

        return rel_i_ma

    def get_image_name_from_time(self, time: int):
        """
        Retrieves the image name corresponding to a specific times.

        :param time: The time index.
        :type time: int
        :return: The name of the image.
        :rtype: str
        """
        imagename = self.image_info_df.loc[self.image_info_df['Experiment_Time[s]'] == time]['Name'].values[0]
        return imagename

    def get_pixel_cordinates_of_LED(self, led_id: int):
        """
        Returns the pixel coordinates of a specified LED.

        :param led_id: The identifier for the LED.
        :type led_id: int
        :return: A list containing the x and y pixel coordinates of the LED.
        :rtype: list
        """
        led_info_df = pd.read_csv(self.led_info_path)
        pixel_positions = \
        led_info_df.loc[led_info_df.index == led_id][[' pixel position x', ' pixel position y']].values[0]
        return pixel_positions

    def get_pixel_values_of_led(self, led_id: int, channel: int, time: int, radius=None):
        """
        Retrieves a cropped numpy array of pixel values around a specified LED, based on its ID, for a given image channel and time.

        This method calculates the pixel values in a specified radius around an LED's position on an image. It first determines the LED's pixel coordinates, then retrieves the image corresponding to the specified time, and finally extracts a square array of pixel values centered on the LED's location.

        :param led_id: The identifier for the LED of interest. This ID is used to look up the LED's position.
        :type led_id: int
        :param channel: The image channel from which to extract pixel values. Different channels may represent different color channels or sensor readings.
        :type channel: int
        :param time: The time at which the image was taken. This corresponds to a specific moment in the experimental timeline.
        :type time: int
        :param radius: The radius around the LED's position from which to extract pixel values. If not specified, the search_area_radius from the config file is used. Defaults to None.
        :type radius: int, optional
        :return: A numpy array containing the pixel values in the specified radius around the LED. The array is cropped from the original image, centered on the LED's position.
        :rtype: numpy.ndarray
        :raises ValueError: If path_images is not available when trying to access image files.
        """
        if not self.path_images:
            raise ValueError("path_images is required for accessing image files")

        # Use the provided radius or fall back to search_area_radius from config
        radius = radius if radius is not None else self.search_area_radius

        if radius:
            pixel_positions = self.get_pixel_cordinates_of_LED(led_id)
            imagename = self.get_image_name_from_time(time)
            imagefile = os.path.join(self.path_images, imagename)
            channel_array = read_channel_data_from_img(imagefile, channel)
            x = pixel_positions[0]
            y = pixel_positions[1]
            channel_array_cropped = np.flip(channel_array[x - radius:x + radius, y - radius:y + radius], axis=0)
            return channel_array_cropped
        return None
