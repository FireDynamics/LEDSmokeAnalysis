import glob
import os
import warnings

import numpy as np
import pandas as pd

from ledsa.core.image_reading import read_channel_data_from_img
from ledsa.core.ConfigData import ConfigData
from ledsa.analysis.ConfigDataAnalysis import ConfigDataAnalysis
from ledsa.analysis.ConfigDataStacked import ConfigDataStacked


# ──────────────────────────────────────────────────────────────────────────────
# Shared base
# ──────────────────────────────────────────────────────────────────────────────

class _BaseSimData:
    """
    Shared base for :class:`SimData` and :class:`StackedSimData`.

    Holds layer geometry, the combined extinction-coefficient DataFrame, and all
    query methods.  Subclasses are responsible for populating the attributes in
    their ``__init__`` and for implementing :meth:`_get_extco_df_from_path`.
    """

    def __init__(self):
        self.all_extco_df = None
        self.solver = None
        self.camera_channels = [0]
        self.num_ref_images = None
        self.n_layers = None
        self.lower_domain_bound = None
        self.upper_domain_bound = None
        self.layer_bottom_heights = None
        self.layer_top_heights = None
        self.layer_center_heights = None

    # ------------------------------------------------------------------
    # Layer-geometry helpers
    # ------------------------------------------------------------------

    def _set_layer_geometry(self, domain_bounds, n_layers):
        """Compute and store layer border and centre arrays."""
        self.n_layers = n_layers
        self.lower_domain_bound = domain_bounds[0]
        self.upper_domain_bound = domain_bounds[1]
        borders = [
            (i / n_layers) * (domain_bounds[1] - domain_bounds[0]) + domain_bounds[0]
            for i in range(n_layers + 1)
        ]
        self.layer_bottom_heights = np.round(np.array(borders[:-1]), 2)
        self.layer_top_heights = np.round(np.array(borders[1:]), 2)
        self.layer_center_heights = np.round(
            (self.layer_bottom_heights + self.layer_top_heights) / 2, 2
        )

    def layer_from_height(self, height):
        """Return the layer index that contains *height* [m]."""
        return int(
            (height - self.lower_domain_bound)
            / (self.upper_domain_bound - self.lower_domain_bound)
            * self.n_layers
        )

    # ------------------------------------------------------------------
    # Smoothing
    # ------------------------------------------------------------------

    def _apply_moving_average(self, df, window, smooth='ma'):
        """
        Apply a centred rolling window to *df*.

        If *window* is 1, the raw data is returned unchanged.  Otherwise a
        centred rolling window is applied using either mean (``'ma'``) or
        median (``'median'``).

        :param df: Input DataFrame (time as index).
        :type df: pd.DataFrame
        :param window: Window width.  A value of 1 returns *df* unchanged.
        :type window: int
        :param smooth: Smoothing method: ``'ma'`` for moving average or
            ``'median'`` for rolling median.  Defaults to ``'ma'``.
        :type smooth: str
        :return: Smoothed DataFrame of the same shape.
        :rtype: pd.DataFrame
        """
        if window == 1:
            return df
        roller = df.rolling(window=window, closed='right')
        smoothed = roller.median() if smooth == 'median' else roller.mean()
        return smoothed.shift(-int(window / 2) + 1)

    # ------------------------------------------------------------------
    # Time helpers
    # ------------------------------------------------------------------

    def get_closest_time(self, time):
        """
        Return the index value in :attr:`all_extco_df` nearest to *time*.

        :param time: Target time [s].
        :type time: int or float
        :return: Closest matching time index value.
        """
        return self.all_extco_df.index[abs(self.all_extco_df.index - time).argmin()]

    def set_timeshift(self, timedelta):
        """
        Shift the time axis of :attr:`all_extco_df` by *timedelta* seconds.

        :param timedelta: Time offset [s].
        :type timedelta: float
        """
        if self.all_extco_df is not None:
            self.all_extco_df.index += timedelta

    # ------------------------------------------------------------------
    # Extinction-coefficient queries
    # ------------------------------------------------------------------

    def get_extco_at_time(self, channel, time, yaxis='layer', height_reference='center', window=1, smooth='ma'):
        """
        Extinction coefficients for all LED arrays at a single time step.

        :param channel: Colour channel index.
        :type channel: int
        :param time: Experiment time [s].  Use :meth:`get_closest_time` to snap
            to the nearest available time.
        :param yaxis: Index type of the returned DataFrame.
            ``'layer'`` returns integer layer indices; ``'height'`` returns
            physical heights [m].
        :type yaxis: str
        :param height_reference: Which layer border to use as the height
            reference: ``'center'``, ``'top'``, or ``'bottom'``.
        :type height_reference: str
        :param window: Rolling-window size for temporal smoothing.  1 = no
            smoothing.
        :type window: int
        :param smooth: Smoothing method: ``'ma'`` (mean) or ``'median'``.
        :type smooth: str
        :return: DataFrame indexed by layer / height; one column per LED array.
        :rtype: pd.DataFrame
        """
        ch_df = self.all_extco_df.xs(channel, level=0, axis=1)
        ma_df = self._apply_moving_average(ch_df, window, smooth)
        ma_df = ma_df.loc[time, :].reset_index().pivot(columns='LED Array', index='Layer')
        ma_df.columns = ma_df.columns.droplevel()
        ma_df.index = range(ma_df.shape[0])
        return self._set_extco_yaxis(ma_df, yaxis, height_reference)

    def get_extco_at_led_array(self, channel, led_array, yaxis='layer', height_reference='center', window=1, smooth='ma'):
        """
        Time series of extinction coefficients for one LED array.

        *led_array* is an ``int`` for :class:`SimData` (LED array ID) and a
        ``str`` for :class:`StackedSimData` (virtual array label, e.g.
        ``'array_A'``).

        :param channel: Colour channel index.
        :type channel: int
        :param led_array: LED array identifier (int for SimData, str for StackedSimData).
        :param yaxis: Column type of the returned DataFrame:
            ``'layer'`` or ``'height'``.
        :type yaxis: str
        :param height_reference: ``'center'``, ``'top'``, or ``'bottom'``.
        :type height_reference: str
        :param window: Rolling-window size for temporal smoothing.
        :type window: int
        :param smooth: Smoothing method: ``'ma'`` or ``'median'``.
        :type smooth: str
        :return: DataFrame indexed by time; one column per layer / height.
        :rtype: pd.DataFrame
        """
        ma_df = self._apply_moving_average(
            self.all_extco_df.xs(channel, level=0, axis=1).xs(led_array, level=0, axis=1),
            window, smooth,
        )
        if yaxis == 'layer':
            ma_df.columns.names = ['Layer']
        elif yaxis == 'height':
            ma_df.columns = self._height_axis(height_reference)
        return ma_df

    def get_extco_at_layer(self, channel, layer, window=1, smooth='ma'):
        """
        Time series of extinction coefficients for a single layer.

        :param channel: Colour channel index.
        :type channel: int
        :param layer: Layer index (0 = bottom-most layer).
        :type layer: int
        :param window: Rolling-window size for temporal smoothing.
        :type window: int
        :param smooth: Smoothing method: ``'ma'`` or ``'median'``.
        :type smooth: str
        :return: DataFrame indexed by time; one column per LED array.
        :rtype: pd.DataFrame
        """
        ch_df = self.all_extco_df.xs(channel, level=0, axis=1).xs(layer, level=1, axis=1)
        return self._apply_moving_average(ch_df, window, smooth)

    def get_extco_at_height(self, channel, height, window=1, smooth='ma'):
        """
        Time series of extinction coefficients at a physical height.

        Converts *height* to a layer index via :meth:`layer_from_height` and
        delegates to :meth:`get_extco_at_layer`.

        :param channel: Colour channel index.
        :type channel: int
        :param height: Target height [m].
        :type height: float
        :param window: Rolling-window size for temporal smoothing.
        :type window: int
        :param smooth: Smoothing method: ``'ma'`` or ``'median'``.
        :type smooth: str
        :return: DataFrame indexed by time; one column per LED array.
        :rtype: pd.DataFrame
        """
        return self.get_extco_at_layer(channel, self.layer_from_height(height), window, smooth)

    # ------------------------------------------------------------------
    # Internal helpers shared by query methods above
    # ------------------------------------------------------------------

    def _height_axis(self, height_reference):
        if height_reference == 'center':
            return self.layer_center_heights
        elif height_reference == 'top':
            return self.layer_top_heights
        elif height_reference == 'bottom':
            return self.layer_bottom_heights
        raise ValueError("height_reference must be 'center', 'top', or 'bottom'")

    def _set_extco_yaxis(self, df, yaxis, height_reference):
        if yaxis == 'layer':
            df.index.names = ['Layer']
        elif yaxis == 'height':
            df.index = self._height_axis(height_reference)
            df.index.names = ['Height / m']
        return df


# ──────────────────────────────────────────────────────────────────────────────
# Single-camera simulation
# ──────────────────────────────────────────────────────────────────────────────

class SimData(_BaseSimData):
    def __init__(self, path_simulation, read_all=True, remove_duplicates=False, load_config_params=True):
        """
        Initializes SimData with simulation and image analysis settings.

        :param path_simulation: Directory containing simulation data.
        :type path_simulation: str
        :param read_all: If True, reads all data on initialization, defaults to True.
        :type read_all: bool, optional
        :param remove_duplicates: If True, removes duplicate entries from analysis, defaults to False.
        :type remove_duplicates: bool, optional
        :param load_config_params: If True, loads parameters from config files, defaults to True.
        :type load_config_params: bool, optional
        """
        super().__init__()
        self.led_params_timedelta = 0
        self.ch0_ledparams_df = None
        self.ch1_ledparams_df = None
        self.ch2_ledparams_df = None
        self.ch0_extcos = None
        self.ch1_extcos = None
        self.ch2_extcos = None
        self.average_images = False
        self.search_area_radius = None
        self.path_images = None

        if load_config_params:
            os.chdir(path_simulation)
            self.config = ConfigData()
            self.config_analysis = ConfigDataAnalysis()

            self.search_area_radius = self.config['find_search_areas']['search_area_radius']
            self.path_images = self.config['find_search_areas']['img_directory']

            self.solver = self.config_analysis['DEFAULT']['solver']
            self.average_images = self.config_analysis.getboolean('DEFAULT', 'average_images')
            self.camera_channels = self.config_analysis.get_list_of_values('DEFAULT', 'camera_channels', dtype=int)
            self.num_ref_images = int(self.config_analysis['DEFAULT']['num_ref_images'])
            self._set_layer_geometry(
                self.config_analysis.get_list_of_values('model_parameters', 'domain_bounds', dtype=float),
                int(self.config_analysis['model_parameters']['num_layers']),
            )

        self.path_simulation = path_simulation
        self.led_info_path = os.path.join(self.path_simulation, 'analysis', 'led_search_areas_with_coordinates.csv')

        image_infos_file = (
            'analysis/image_infos_analysis_avg.csv'
            if self.average_images
            else 'analysis/image_infos_analysis.csv'
        )
        self.image_info_df = pd.read_csv(os.path.join(self.path_simulation, image_infos_file))

        if read_all:
            self.read_all()
        if remove_duplicates:
            self.remove_duplicate_heights()

    # set_timeshift override: also updates LED-parameter time axis
    def set_timeshift(self, timedelta):
        """
        Sets the time shift for the experiment's start time.

        :param timedelta: Time shift in seconds.
        :type timedelta: int
        """
        super().set_timeshift(timedelta)
        self.led_params_timedelta = timedelta

    # ------------------------------------------------------------------
    # Data loading
    # ------------------------------------------------------------------

    def _get_ledparams_df_from_path(self, channel):
        """
        Reads experimental parameters from a binary HDF5 table based on color channel.

        :param channel: The color channel index to read the parameters for.
        :type channel: int
        :return: DataFrame with LED parameters.
        :rtype: pd.DataFrame
        """
        if self.average_images:
            file = os.path.join(self.path_simulation, 'analysis', f'channel{channel}', 'all_parameters_avg.h5')
        else:
            file = os.path.join(self.path_simulation, 'analysis', f'channel{channel}', 'all_parameters.h5')
        table = pd.read_hdf(file, key='channel_values')
        # Not rounded to whole seconds: images taken less than a second apart would otherwise share
        # a time and be merged into one entry.
        time = pd.Series(
            self.image_info_df['Experiment_Time[s]'].values,
            index=self.image_info_df['#ID'],
            name='Experiment_Time[s]',
        )
        table = table.merge(time, left_on='img_id', right_index=True)
        table.set_index(['Experiment_Time[s]'], inplace=True)
        self.led_heights = table['height']
        return table

    def _get_extco_df_from_path(self):
        """
        Read all extinction coefficients from the simulation dir and put them in all_extco_df.
        """
        extco_list = []
        files_list = glob.glob(
            os.path.join(
                self.path_simulation, 'analysis', 'extinction_coefficients',
                self.solver, 'extinction_coefficients*.csv',
            )
        )
        for file in files_list:
            file_df = pd.read_csv(file, skiprows=7)
            channel = int(file.split('channel_')[1].split('_')[0])
            led_array = int(file.split('array_')[1].split('.')[0])
            if '# Experiment_Time[s]' not in file_df.columns:
                time = self.image_info_df['Experiment_Time[s]']
                file_df = file_df.merge(time, left_index=True, right_index=True)
            else:
                file_df.rename(columns={'# Experiment_Time[s]': 'Experiment_Time[s]'}, inplace=True)
                self._warn_on_foreign_experiment_times(file_df['Experiment_Time[s]'], file)
            file_df.set_index('Experiment_Time[s]', inplace=True)
            n_layers = len(file_df.columns)
            file_df.columns = pd.MultiIndex.from_product(
                [[channel], [led_array], list(range(n_layers))],
                names=["Channel", "LED Array", "Layer"],
            )
            extco_list.append(file_df)
        self.all_extco_df = pd.concat(extco_list, axis=1)
        self.all_extco_df.sort_index(ascending=True, axis=1, inplace=True)
        self.all_extco_df = self.all_extco_df[~self.all_extco_df.index.duplicated(keep='first')]

    def _warn_on_foreign_experiment_times(self, experiment_times, file, tolerance=1e-6):
        """
        Warn if an extinction coefficient file was computed from other image times than the current ones.

        The files carry their own time axis, while the LED parameters are put on the times read from the
        image infos. Re-running step 1 or step 3 therefore moves the LED parameters but leaves the
        coefficients on the times of the earlier run, which nothing else would notice: both data frames
        are simply indexed by times that no longer belong to the same images. Images the analysis left
        out are no mismatch, so only times without a counterpart in the image infos are reported.

        :param experiment_times: Experiment times the file was computed for.
        :type experiment_times: pandas.Series
        :param file: Path of the extinction coefficient file, used for the message.
        :type file: str
        :param tolerance: Largest difference in seconds still counted as the same time.
        :type tolerance: float
        """
        image_times = self.image_info_df['Experiment_Time[s]'].to_numpy(dtype=float)
        file_times = experiment_times.to_numpy(dtype=float)
        foreign = file_times[[not np.any(np.abs(image_times - time) <= tolerance) for time in file_times]]
        if foreign.size == 0:
            return
        warnings.warn(
            f'{os.path.basename(file)} was computed for {foreign.size} of {file_times.size} experiment '
            f'times that the image infos do not contain, the first one being {foreign[0]} s. The file is '
            f'left over from an earlier run and does not line up with the LED parameters. Run the '
            f'analysis again to recompute it.',
            UserWarning, stacklevel=3)

    def read_led_params(self):
        """Read led parameters for all color channels from the simulation path."""
        self.ch0_ledparams_df = self._get_ledparams_df_from_path(0)
        self.ch1_ledparams_df = self._get_ledparams_df_from_path(1)
        self.ch2_ledparams_df = self._get_ledparams_df_from_path(2)

    def read_all(self):
        """Read led parameters and extinction coefficients for all channels."""
        self.read_led_params()
        self._get_extco_df_from_path()

    def remove_duplicate_heights(self):
        """Remove duplicate height entries for each LED parameter DataFrame."""
        self.ch0_ledparams_df = self.ch0_ledparams_df.groupby(['Experiment_Time[s]', 'height']).last()
        self.ch1_ledparams_df = self.ch1_ledparams_df.groupby(['Experiment_Time[s]', 'height']).last()
        self.ch2_ledparams_df = self.ch2_ledparams_df.groupby(['Experiment_Time[s]', 'height']).last()

    # ------------------------------------------------------------------
    # LED parameter queries
    # ------------------------------------------------------------------

    def get_ledparams_at_led_array(self, channel, led_array_id, param='sum_col_val', yaxis='led_id', window=1, n_ref=None):
        """
        Retrieves a DataFrame containing normalized LED parameters for a specific LED array.

        :param channel: The channel index from which to extract LED parameters.
        :type channel: int
        :param led_array_id: The ID for which to extract LED parameters.
        :type led_array_id: int
        :param param: The specific LED parameter to extract and analyze. Defaults to 'sum_col_val'.
        :type param: str, optional
        :param yaxis: Column labeling: ``'led_id'`` or ``'height'``. Defaults to ``'led_id'``.
        :type yaxis: str, optional
        :param window: Smoothing window size.  1 = no smoothing.
        :type window: int, optional
        :param n_ref: Number of initial entries for normalisation.  ``False`` disables
            normalisation.  Defaults to ``None`` (uses :attr:`num_ref_images`).
        :type n_ref: int or bool, optional
        :return: DataFrame; index = time, columns = LED IDs or heights.
        :rtype: pd.DataFrame
        """
        led_params = {0: self.ch0_ledparams_df, 1: self.ch1_ledparams_df, 2: self.ch2_ledparams_df}[channel]
        index = 'height' if yaxis == 'height' else 'led_id'
        led_params = led_params.reset_index().set_index(['Experiment_Time[s]', index])
        led_params.index = led_params.index.set_levels(
            led_params.index.levels[0] + self.led_params_timedelta, level=0
        )
        ii = led_params[led_params['led_array_id'] == led_array_id][[param]]
        if n_ref is False:
            rel_i = ii
        else:
            n_ref = self.num_ref_images if n_ref is None else n_ref
            i0 = ii.groupby([index]).agg(lambda g: g.iloc[:n_ref].mean())
            rel_i = ii / i0
        rel_i = rel_i.reset_index().pivot(columns=index, index='Experiment_Time[s]')
        rel_i.columns = rel_i.columns.droplevel()
        return self._apply_moving_average(rel_i, window)

    def get_ledparams_at_led_id(self, channel, led_id, param='sum_col_val', window=1, n_ref=None):
        """
        Retrieves a DataFrame containing normalized LED parameters for a specific LED ID.

        :param channel: The channel index from which to extract LED parameters.
        :type channel: int
        :param led_id: The identifier of the LED for which to extract parameters.
        :type led_id: int
        :param param: The specific LED parameter to extract and analyze. Defaults to 'sum_col_val'.
        :type param: str, optional
        :param window: Smoothing window size.  1 = no smoothing.
        :type window: int, optional
        :param n_ref: Number of initial entries for normalisation.  ``False`` disables
            normalisation.  Defaults to ``None`` (uses :attr:`num_ref_images`).
        :type n_ref: int or bool, optional
        :return: DataFrame indexed by time with the specified LED parameter.
        :rtype: pd.DataFrame
        """
        led_params = {0: self.ch0_ledparams_df, 1: self.ch1_ledparams_df, 2: self.ch2_ledparams_df}[channel]
        led_params = led_params.reset_index().set_index(['Experiment_Time[s]'])
        ii = led_params[led_params['led_id'] == led_id][[param]]
        if n_ref is False:
            rel_i = ii
        else:
            n_ref = self.num_ref_images if n_ref is None else n_ref
            i0 = ii.iloc[:n_ref].mean()
            rel_i = ii / i0
        return self._apply_moving_average(rel_i, window)

    # ------------------------------------------------------------------
    # Image helpers
    # ------------------------------------------------------------------

    def get_image_name_from_time(self, time):
        """
        Retrieves the image name corresponding to a specific time.

        :param time: The time index.
        :type time: float
        :return: The name of the image.
        :rtype: str
        """
        return self.image_info_df.loc[self.image_info_df['Experiment_Time[s]'] == time]['Name'].values[0]

    def get_pixel_cordinates_of_LED(self, led_id):
        """
        Returns the pixel coordinates of a specified LED.

        :param led_id: The identifier for the LED.
        :type led_id: int
        :return: A list containing the x and y pixel coordinates of the LED.
        :rtype: list
        """
        led_info_df = pd.read_csv(self.led_info_path)
        return led_info_df.loc[led_info_df.index == led_id][
            [' pixel position x', ' pixel position y']
        ].values[0]

    def get_pixel_values_of_led(self, led_id, channel, time, radius=None):
        """
        Retrieves a cropped numpy array of pixel values around a specified LED.

        :param led_id: The identifier for the LED of interest.
        :type led_id: int
        :param channel: The image channel from which to extract pixel values.
        :type channel: int
        :param time: The time at which the image was taken.
        :type time: int
        :param radius: Pixel radius around the LED. If not specified, uses
            ``search_area_radius`` from the config file.
        :type radius: int, optional
        :return: A numpy array of pixel values around the LED.
        :rtype: numpy.ndarray
        :raises ValueError: If ``path_images`` is not available.
        """
        if not self.path_images:
            raise ValueError("path_images is required for accessing image files")
        radius = radius if radius is not None else self.search_area_radius
        if radius:
            pixel_positions = self.get_pixel_cordinates_of_LED(led_id)
            imagename = self.get_image_name_from_time(time)
            imagefile = os.path.join(self.path_images, imagename)
            channel_array = read_channel_data_from_img(imagefile, channel)
            x, y = pixel_positions[0], pixel_positions[1]
            return np.flip(channel_array[x - radius:x + radius, y - radius:y + radius], axis=0)
        return None


# ──────────────────────────────────────────────────────────────────────────────
# Multi-camera stacked analysis
# ──────────────────────────────────────────────────────────────────────────────

class StackedSimData(_BaseSimData):
    """
    Postprocessing interface for stacked multi-camera extinction coefficient results.

    Reads ``config_stacked.ini`` from *config_path*, discovers the stacked CSV
    output files produced by
    :class:`~ledsa.analysis.StackedExtinctionCoefficients`, and exposes the same
    :class:`_BaseSimData` query API for those results.

    LED parameter data is not available through this class because stacked results
    aggregate rays from multiple cameras.  Use individual per-camera
    :class:`SimData` objects for raw LED intensities.

    :ivar config_stacked: Parsed stacked-analysis configuration.
    :vartype config_stacked: ConfigDataStacked
    """

    def __init__(self, config_path='.', read_all=True):
        """
        :param config_path: Directory containing ``config_stacked.ini``.
            Defaults to the current working directory.
        :type config_path: str
        :param read_all: If ``True``, immediately load all stacked extinction
            coefficient CSV files.  Defaults to ``True``.
        :type read_all: bool
        """
        super().__init__()

        prev_dir = os.getcwd()
        try:
            os.chdir(config_path)
            self.config_stacked = ConfigDataStacked()
        finally:
            os.chdir(prev_dir)

        self.config_path = os.path.abspath(config_path)
        self.output_path = os.path.join(self.config_path, str(self.config_stacked.output_path))

        self.solver = self.config_stacked.solver
        self.camera_channels = self.config_stacked.camera_channels
        self.num_ref_images = self.config_stacked.num_ref_images
        self._set_layer_geometry(self.config_stacked.domain_bounds, self.config_stacked.num_layers)

        if read_all:
            self._get_extco_df_from_path()

    def _get_extco_df_from_path(self):
        """
        Discover and load all stacked extinction coefficient CSV files.

        Files are expected in::

            <output_path>/analysis/extinction_coefficients/<solver>/

        The virtual LED-array ID is stored as an integer in the
        ``all_extco_df`` MultiIndex (level ``"LED Array"``).
        """
        extco_list = []
        extco_dir = os.path.join(
            self.output_path, 'analysis', 'extinction_coefficients', self.solver,
        )
        files = glob.glob(os.path.join(extco_dir, 'extinction_coefficients*.csv'))

        if not files:
            print(
                f'No stacked extinction coefficient files found in:\n  {extco_dir}\n'
                'Run the stacked analysis first with: python -m ledsa -as'
            )
            return

        for filepath in files:
            fname = os.path.basename(filepath)
            channel = int(fname.split('channel_')[1].split('_')[0])
            led_array_id = int(fname.split('_led_array_')[1].removesuffix('.csv'))

            file_df = pd.read_csv(filepath, skiprows=4)
            file_df.rename(columns={'# Experiment_Time[s]': 'Experiment_Time[s]'}, inplace=True)
            file_df['Experiment_Time[s]'] = file_df['Experiment_Time[s]'].astype(float)
            file_df.set_index('Experiment_Time[s]', inplace=True)

            n_layers = len(file_df.columns)
            file_df.columns = pd.MultiIndex.from_product(
                [[channel], [led_array_id], list(range(n_layers))],
                names=["Channel", "LED Array", "Layer"],
            )
            extco_list.append(file_df)

        if not extco_list:
            return

        self.all_extco_df = pd.concat(extco_list, axis=1)
        self.all_extco_df.sort_index(ascending=True, axis=1, inplace=True)
        self.all_extco_df = self.all_extco_df[~self.all_extco_df.index.duplicated(keep='first')]