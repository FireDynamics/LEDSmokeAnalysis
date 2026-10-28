"""
Correction of the LED intensities for the temperature of the LEDs.

An LED loses radiant power as it heats up, so the LEDs in the hot gas layer dim over the course of
an experiment without any smoke between them and the camera. Left uncorrected, that drift enters the
relative intensities as an extinction that is not there.

The factors the intensities are reduced by are measured or modelled outside LEDSA and read from one
file per camera channel, as a table of the reduction factor over the experiment time and the height
of the LED. LEDSA interpolates that table bilinearly to the time of every image and the height of
every LED and divides the intensities by it, so that an uncorrected analysis is what it was before.

The interpolation stays between the tabulated times and heights. The table states where the
reduction of the LEDs was determined, so an image or an LED outside it is left exactly as it was
extracted rather than corrected with a factor the measurement does not cover.
"""
import warnings
from pathlib import Path
from typing import Dict, Optional, Union

import numpy as np
import pandas as pd
from scipy.interpolate import RegularGridInterpolator

# Both axes of the table are interpolated linearly, which takes at least two points per axis
MIN_POINTS_PER_AXIS = 2

# Camera channels a reduction factor file can be configured for, one per channel of a RGB camera
TEMPERATURE_CORRECTION_CHANNELS = (0, 1, 2)

# Column names that mark a file holding one row per height and time instead of a table
LONG_FORMAT_COLUMNS = ('height', 'time', 'reduction_factor')

# Factor of a time or a height the table does not cover, which leaves the intensity untouched
UNCORRECTED_FACTOR = 1.0


class TemperatureCorrection:
    """
    Reduction factors of the LED intensities over the experiment time and the height of the LED.

    :ivar times: Experiment times [s] of the table, ascending.
    :vartype times: np.ndarray
    :ivar heights: LED heights [m] of the table, ascending.
    :vartype heights: np.ndarray
    :ivar factors: Reduction factor of every time and height, indexed as [time, height].
    :vartype factors: np.ndarray
    :ivar source: Name of the file the factors were read from, for the result file header.
    :vartype source: str
    """

    def __init__(self, times: np.ndarray, heights: np.ndarray, factors: np.ndarray, source: str = ''):
        """
        :param times: Experiment times [s] of the table, ascending and without duplicates.
        :type times: np.ndarray
        :param heights: LED heights [m] of the table, ascending and without duplicates.
        :type heights: np.ndarray
        :param factors: Reduction factor of every time and height, shaped (len(times), len(heights)).
        :type factors: np.ndarray
        :param source: Name of the file the factors were read from.
        :type source: str
        """
        self.times = times
        self.heights = heights
        self.factors = factors
        self.source = source
        self._interpolate = RegularGridInterpolator((times, heights), factors, bounds_error=False,
                                                    fill_value=UNCORRECTED_FACTOR)

    def __str__(self):
        return (f'{self.source} ({self.times[0]:.1f} s to {self.times[-1]:.1f} s, '
                f'{self.heights[0]:.2f} m to {self.heights[-1]:.2f} m)')

    @classmethod
    def from_file(cls, file_path: Union[str, Path]) -> 'TemperatureCorrection':
        """
        Read the reduction factors from a CSV file, in either of two layouts.

        As a table, the first column holds the experiment time [s], the header of the other columns
        the height [m] of the LEDs the factors apply to, and every cell the factor the intensity of
        its time and height is reduced by. The name of the first column is not read, so any label can
        be used for it.

        As a list, the file has the three columns named in LONG_FORMAT_COLUMNS in any order and one
        row per pair of a height and a time, which is pivoted into the same table.

        Neither layout has to be sorted in either of its axes.

        :param file_path: Path of the reduction factor file.
        :type file_path: str or Path
        :return: The reduction factors, ready to be interpolated.
        :rtype: TemperatureCorrection
        :raises FileNotFoundError: If the file does not exist.
        :raises ValueError: If the file is not a table of numbers with at least two times and two
            heights, or if it holds a factor that is not positive.
        """
        path = Path(file_path)
        if not path.is_file():
            raise FileNotFoundError(
                f'The reduction factor file {path} of the temperature correction does not exist. '
                f'Give the path of an existing file in the [temperature_correction] section of '
                f'config_analysis.ini, or set the option to None to analyse the channel without the '
                f'correction.')
        cls._check_field_counts(path)
        contents = pd.read_csv(path)
        if cls._is_a_list_of_factors(contents):
            table = cls._table_from_list(contents, path)
        else:
            table = contents.set_index(contents.columns[0])
        # A delimiter at the end of the header leaves an empty column behind, an empty last line an
        # empty row, and neither is a point of the table
        table = table.dropna(axis=1, how='all').dropna(axis=0, how='all')
        times = cls._axis(table.index.to_numpy(), 'time', path)
        heights = cls._axis(table.columns.to_numpy(), 'height', path)
        factors = cls._factors(table, path)
        # The axes are sorted here instead of in the file, so that the heights can be written in the
        # order of the LEDs of the array, which runs from the top down
        factors = factors[np.ix_(np.argsort(times), np.argsort(heights))]
        return cls(np.sort(times), np.sort(heights), factors, source=path.name)

    @staticmethod
    def _check_field_counts(path: Path) -> None:
        """
        Check that no line holds more values than the header names.

        A decimal comma writes one factor as two fields. Pandas absorbs the extra field by moving the
        first column into the index, which shifts every factor of the line into the column of its
        neighbour instead of failing, so the file has to be checked before it is parsed.

        :param path: Path of the reduction factor file.
        :type path: Path
        :raises ValueError: If the file is empty or a line holds more values than the header.
        """
        with open(path, encoding='utf-8-sig') as factor_file:
            lines = [(number, line) for number, line in enumerate(factor_file, start=1)
                     if line.strip()]
        if not lines:
            raise ValueError(f'The reduction factor file {path} holds no lines.')
        fields_in_header = len(lines[0][1].rstrip('\n').split(','))
        for number, line in lines[1:]:
            fields = len(line.rstrip('\n').split(','))
            if fields > fields_in_header:
                raise ValueError(
                    f'Line {number} of the reduction factor file {path} holds {fields} values while '
                    f'its header names {fields_in_header}. A factor written with a decimal comma '
                    f'instead of a decimal point counts as two values and shifts the rest of the '
                    f'line into the columns of the neighbouring heights.')

    @staticmethod
    def _is_a_list_of_factors(contents: pd.DataFrame) -> bool:
        """
        Whether a file holds one row per height and time instead of a table.

        :param contents: The file as it was read.
        :type contents: pd.DataFrame
        :return: True if the columns are the ones of the list layout.
        :rtype: bool
        """
        names = {str(name).strip().lower() for name in contents.columns}
        return names == set(LONG_FORMAT_COLUMNS)

    @staticmethod
    def _table_from_list(contents: pd.DataFrame, path: Path) -> pd.DataFrame:
        """
        Turn a list of one factor per height and time into a table of the factors over time.

        :param contents: The file as it was read, with the columns of LONG_FORMAT_COLUMNS.
        :type contents: pd.DataFrame
        :param path: Path of the file, for the error messages.
        :type path: Path
        :return: The factors as a table, time over height.
        :rtype: pd.DataFrame
        :raises ValueError: If a height and time occurs in more than one row.
        """
        height, time, factor = LONG_FORMAT_COLUMNS
        contents = contents.rename(columns={name: str(name).strip().lower()
                                            for name in contents.columns})
        try:
            return contents.pivot(index=time, columns=height, values=factor)
        except ValueError as error:
            raise ValueError(f'The reduction factor file {path} holds more than one factor for the '
                             f'same height and time ({error}). Every pair of a height and a time '
                             f'takes exactly one row.') from error

    @staticmethod
    def _axis(values: np.ndarray, name: str, path: Path) -> np.ndarray:
        """
        Read one axis of the table and check that it can be interpolated.

        :param values: Labels of the axis, as they stand in the file.
        :type values: np.ndarray
        :param name: Name of the axis, for the error messages.
        :type name: str
        :param path: Path of the file, for the error messages.
        :type path: Path
        :return: The axis as numbers.
        :rtype: np.ndarray
        :raises ValueError: If the labels are not numbers, if there are fewer than
            MIN_POINTS_PER_AXIS of them or if one of them occurs twice.
        """
        try:
            axis = values.astype(float)
        except (TypeError, ValueError) as error:
            raise ValueError(
                f'The {name} values of the reduction factor file {path} are not numbers ({error}). '
                f'The file holds the experiment time [s] in its first column and the LED height [m] '
                f'in its header.') from error
        if axis.size < MIN_POINTS_PER_AXIS:
            raise ValueError(
                f'The reduction factor file {path} holds {axis.size} {name} value(s), while the '
                f'factors are interpolated between at least {MIN_POINTS_PER_AXIS}. Repeat the factors '
                f'at a second {name} to apply them independently of the {name}.')
        if np.unique(axis).size != axis.size:
            raise ValueError(f'The reduction factor file {path} holds the same {name} value twice, so '
                             f'the factors of that {name} are ambiguous.')
        return axis

    @staticmethod
    def _factors(table: pd.DataFrame, path: Path) -> np.ndarray:
        """
        Read the cells of the table and check that the intensities can be divided by them.

        :param table: The table as it was read, time over height.
        :type table: pd.DataFrame
        :param path: Path of the file, for the error messages.
        :type path: Path
        :return: The reduction factors, indexed as [time, height].
        :rtype: np.ndarray
        :raises ValueError: If a cell is not a number or holds a factor that is not positive.
        """
        try:
            factors = table.to_numpy(dtype=float)
        except (TypeError, ValueError) as error:
            raise ValueError(f'The reduction factors in {path} are not all numbers ({error}). Every '
                             f'cell holds the factor of the time of its row and the height of its '
                             f'column.') from error
        if not np.all(np.isfinite(factors)):
            raise ValueError(f'{int(np.count_nonzero(~np.isfinite(factors)))} of the reduction factors '
                             f'in {path} are empty or not finite. The table has to be filled for every '
                             f'combination of its times and heights.')
        if np.any(factors <= 0):
            raise ValueError(f'{int(np.count_nonzero(factors <= 0))} of the reduction factors in {path} '
                             f'are not positive. The LED intensities are divided by the factors, which '
                             f'a factor of zero or less does not allow.')
        return factors

    def factors_at(self, times: np.ndarray, heights: np.ndarray) -> np.ndarray:
        """
        Reduction factor of every pair of a time and a height, interpolated bilinearly.

        The factors are interpolated between the tabulated times and heights only. A time or a
        height the table does not cover gets UNCORRECTED_FACTOR, which leaves the intensity as it
        was extracted, because the table states where the reduction of the LEDs was determined and
        there is nothing outside it to extrapolate from.

        :param times: Experiment time [s] of every value to correct, counted from start_time of
            config.ini like the Experiment_Time[s] of the image infos file the times are read from.
        :type times: np.ndarray
        :param heights: Height [m] of the LED of every value to correct.
        :type heights: np.ndarray
        :return: The reduction factor of every value, in the order of the arguments.
        :rtype: np.ndarray
        """
        times = np.asarray(times, dtype=float)
        heights = np.asarray(heights, dtype=float)
        self._warn_about_values_outside_the_table(times, heights)
        return self._interpolate((times, heights))

    def _warn_about_values_outside_the_table(self, times: np.ndarray, heights: np.ndarray) -> None:
        """
        Report the times and heights the table does not cover, which are left uncorrected.

        :param times: Requested experiment times [s].
        :type times: np.ndarray
        :param heights: Requested LED heights [m].
        :type heights: np.ndarray
        """
        outside = []
        times_outside = np.count_nonzero((times < self.times[0]) | (times > self.times[-1]))
        heights_outside = np.count_nonzero((heights < self.heights[0]) | (heights > self.heights[-1]))
        if times_outside:
            outside.append(f'{times_outside} of {times.size} image times lie outside '
                           f'{self.times[0]:.1f} s to {self.times[-1]:.1f} s')
        if heights_outside:
            outside.append(f'{heights_outside} of {heights.size} LED heights lie outside '
                           f'{self.heights[0]:.2f} m to {self.heights[-1]:.2f} m')
        if outside:
            warnings.warn(f'The reduction factors of {self.source} do not cover the analysed data: '
                          f'{" and ".join(outside)}. These intensities are left uncorrected.',
                          category=UserWarning, stacklevel=3)


def load_temperature_corrections(
        file_paths_per_channel: Dict[int, Optional[str]]) -> Dict[int, TemperatureCorrection]:
    """
    Read the reduction factor file of every channel that has one configured.

    The files are read before the first coefficient is computed, so that an unreadable one stops the
    run right away instead of in the middle of it.

    :param file_paths_per_channel: Path of the reduction factor file of every channel, None for the
        channels to analyse without the correction.
    :type file_paths_per_channel: dict[int, str or None]
    :return: The reduction factors of the channels that are corrected.
    :rtype: dict[int, TemperatureCorrection]
    """
    corrections = {}
    for channel, file_path in sorted(file_paths_per_channel.items()):
        if file_path is None:
            continue
        corrections[channel] = TemperatureCorrection.from_file(file_path)
        print(f'Temperature correction of channel {channel}: reduction factors from '
              f'{corrections[channel]}.')
    return corrections
