"""Unit tests for the correction of the LED intensities for the temperature of the LEDs.

The reduction factors are computed outside LEDSA, so the tests cover reading their table, the
interpolation to the time of an image and the height of an LED, the division of the intensities by
them and the note the result file carries about the corrected channel. A channel without a
reduction factor file has to be analysed exactly as before the correction existed.
"""
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from ledsa.analysis.ConfigDataAnalysis import ConfigDataAnalysis
from ledsa.analysis.ExtinctionCoefficients import ExtinctionCoefficients
from ledsa.analysis.temperature_correction import TemperatureCorrection, load_temperature_corrections

REFERENCE_PROPERTY = 'sum_col_val'
LED_IDS = [0, 1]
LED_HEIGHTS = {0: 1.0, 1: 2.0}
IMG_TIMES = {1: 0.0, 2: 50.0, 3: 100.0}
INTENSITY = 100.0

# Factors of 1 at the start of the experiment, falling as the LEDs heat up and more so at the top
FACTOR_TABLE = """\
Time / s,1.0,2.0
0,1.0,1.0
100,0.8,0.6
"""


def write_factors(tmp_path, text=FACTOR_TABLE, name='reduction_factors.csv') -> Path:
    """Write a reduction factor file and return its path."""
    file_path = tmp_path / name
    file_path.write_text(text, encoding='utf-8')
    return file_path


class LedStub:
    def __init__(self, led_id, pos_z):
        self.id = led_id
        self.pos_z = pos_z


class ExperimentStub:
    led_array = 0
    channel = 0

    def __init__(self, path=Path('.'), leds=None):
        self.path = path
        self.leds = leds if leds is not None else [LedStub(led_id, height)
                                                   for led_id, height in LED_HEIGHTS.items()]

    def __str__(self):
        return f'channel: {self.channel}, led_array: {self.led_array}\n'


class SolverStub(ExtinctionCoefficients):
    """Smallest concrete solver, with the data and the distances it would read from disk stubbed out."""

    def calc_coefficients_of_img(self, rel_intensities: np.ndarray) -> np.ndarray:
        return rel_intensities

    def calc_distance_array(self) -> np.ndarray:
        return np.zeros((len(LED_IDS), 1))

    def load_img_data(self) -> None:
        index = pd.MultiIndex.from_product([sorted(IMG_TIMES), LED_IDS], names=['img_id', 'led_id'])
        self.calculated_img_data = pd.DataFrame({REFERENCE_PROPERTY: INTENSITY}, index=index)


def make_solver(tmp_path, correction=None, ref_img_indices=None) -> SolverStub:
    """Create a solver whose image infos name the times of the stubbed image data."""
    analysis_dir = tmp_path / 'analysis'
    analysis_dir.mkdir(exist_ok=True)
    pd.DataFrame({'#ID': list(IMG_TIMES), 'Experiment_Time[s]': list(IMG_TIMES.values())}).to_csv(
        analysis_dir / 'image_infos_analysis.csv', index=False)
    solver = SolverStub(ExperimentStub(path=tmp_path), reference_property=REFERENCE_PROPERTY,
                        num_ref_imgs=1, ref_img_indices=ref_img_indices,
                        temperature_correction=correction)
    solver.load_img_data()
    return solver


def corrected_intensity(img_id, led_id) -> float:
    """Intensity of an LED after the correction, from the factors of FACTOR_TABLE."""
    factors_at_100_s = {0: 0.8, 1: 0.6}
    reduction_at_100_s = 1.0 - factors_at_100_s[led_id]
    factor = 1.0 - reduction_at_100_s * IMG_TIMES[img_id] / 100.0
    return INTENSITY / factor


class TestReadingTheFactors:
    def test_factors_are_interpolated_between_the_points_of_the_table(self, tmp_path):
        correction = TemperatureCorrection.from_file(write_factors(tmp_path))

        # the corners of the table, the middle of its time axis and the middle of both axes
        factors = correction.factors_at([0, 100, 100, 50, 50], [1.0, 1.0, 2.0, 2.0, 1.5])

        assert factors == pytest.approx([1.0, 0.8, 0.6, 0.8, 0.85])

    def test_heights_in_the_header_may_run_from_the_top_down(self, tmp_path):
        # the heights of an LED array are naturally written in the order of its LEDs, from the top
        top_down = 'time,2.0,1.0\n0,1.0,1.0\n100,0.6,0.8\n'
        correction = TemperatureCorrection.from_file(write_factors(tmp_path, top_down))

        assert list(correction.heights) == [1.0, 2.0]
        assert correction.factors_at([100, 100], [1.0, 2.0]) == pytest.approx([0.8, 0.6])

    def test_times_in_the_file_may_be_unordered(self, tmp_path):
        unordered = 'time,1.0,2.0\n100,0.8,0.6\n0,1.0,1.0\n'
        correction = TemperatureCorrection.from_file(write_factors(tmp_path, unordered))

        assert correction.factors_at([0, 100], [1.0, 1.0]) == pytest.approx([1.0, 0.8])

    def test_a_trailing_delimiter_in_the_header_is_not_a_height(self, tmp_path):
        with_trailing_delimiter = 'time,1.0,2.0,\n0,1.0,1.0,\n100,0.8,0.6,\n'
        correction = TemperatureCorrection.from_file(write_factors(tmp_path, with_trailing_delimiter))

        assert list(correction.heights) == [1.0, 2.0]

    def test_values_outside_the_table_are_left_uncorrected(self, tmp_path):
        correction = TemperatureCorrection.from_file(write_factors(tmp_path))

        with pytest.warns(UserWarning, match='left uncorrected'):
            # before and after the tabulated times, below and above the tabulated heights
            factors = correction.factors_at([-10, 200, 100, 100], [1.0, 1.0, 0.5, 3.0])

        assert factors == pytest.approx([1.0, 1.0, 1.0, 1.0])

    def test_a_tabulated_height_outside_the_tabulated_times_is_left_uncorrected(self, tmp_path):
        # one coordinate inside the table is not enough: the factor is interpolated between the
        # points of both axes or not at all
        correction = TemperatureCorrection.from_file(write_factors(tmp_path))

        with pytest.warns(UserWarning, match='1 of 1 image times lie outside 0.0 s to 100.0 s'):
            factors = correction.factors_at([200], [2.0])

        assert factors == pytest.approx([1.0])

    def test_the_warning_counts_the_times_and_the_heights_separately(self, tmp_path):
        correction = TemperatureCorrection.from_file(write_factors(tmp_path))

        with pytest.warns(UserWarning) as warnings_raised:
            correction.factors_at([-10, 200, 50], [1.0, 1.0, 9.0])

        message = str(warnings_raised[0].message)
        assert '2 of 3 image times lie outside 0.0 s to 100.0 s' in message
        assert '1 of 3 LED heights lie outside 1.00 m to 2.00 m' in message

    def test_values_inside_the_table_do_not_warn(self, tmp_path, recwarn):
        correction = TemperatureCorrection.from_file(write_factors(tmp_path))

        correction.factors_at([0, 50, 100], [1.0, 1.5, 2.0])

        assert [str(warning.message) for warning in recwarn] == []

    def test_the_source_and_the_covered_range_are_named(self, tmp_path):
        correction = TemperatureCorrection.from_file(write_factors(tmp_path))

        assert str(correction) == 'reduction_factors.csv (0.0 s to 100.0 s, 1.00 m to 2.00 m)'


class TestReadingAListOfFactors:
    """A generator may write one row per height and time instead of a table."""

    LIST_OF_FACTORS = """\
height,time,reduction_factor
1.0,0.0,1.0
1.0,100.0,0.8
2.0,0.0,1.0
2.0,100.0,0.6
"""

    def test_a_list_of_factors_is_read_like_the_table_of_the_same_factors(self, tmp_path):
        from_list = TemperatureCorrection.from_file(
            write_factors(tmp_path, self.LIST_OF_FACTORS, 'list.csv'))
        from_table = TemperatureCorrection.from_file(write_factors(tmp_path))

        assert list(from_list.times) == list(from_table.times)
        assert list(from_list.heights) == list(from_table.heights)
        assert from_list.factors == pytest.approx(from_table.factors)

    def test_the_columns_of_a_list_may_stand_in_any_order(self, tmp_path):
        reordered = 'time,reduction_factor,height\n0.0,1.0,1.0\n100.0,0.8,1.0\n' \
                    '0.0,1.0,2.0\n100.0,0.6,2.0\n'

        correction = TemperatureCorrection.from_file(write_factors(tmp_path, reordered))

        assert correction.factors_at([100, 100], [1.0, 2.0]) == pytest.approx([0.8, 0.6])

    def test_a_repeated_height_and_time_is_rejected(self, tmp_path):
        repeated = self.LIST_OF_FACTORS + '1.0,0.0,0.9\n'

        with pytest.raises(ValueError, match='more than one factor for the same height and time'):
            TemperatureCorrection.from_file(write_factors(tmp_path, repeated))

    def test_a_missing_pair_is_rejected(self, tmp_path):
        incomplete = 'height,time,reduction_factor\n1.0,0.0,1.0\n1.0,100.0,0.8\n2.0,0.0,1.0\n'

        with pytest.raises(ValueError, match='empty or not finite'):
            TemperatureCorrection.from_file(write_factors(tmp_path, incomplete))


class TestRejectedFactorFiles:
    def test_a_decimal_comma_is_named_with_its_line(self, tmp_path):
        # 0,6 instead of 0.6 turns one factor into two fields, which pandas absorbs by shifting
        # every factor of the line into the column of its neighbour
        decimal_comma = 'time,1.0,2.0\n0,1.0,1.0\n100,0,6,0.8\n'

        with pytest.raises(ValueError, match=r'Line 3 .* holds 4 values while its header names 3'):
            TemperatureCorrection.from_file(write_factors(tmp_path, decimal_comma))

    def test_an_empty_file_is_rejected(self, tmp_path):
        with pytest.raises(ValueError, match='holds no lines'):
            TemperatureCorrection.from_file(write_factors(tmp_path, '\n\n'))

    def test_a_missing_file_is_named_in_the_error(self, tmp_path):
        with pytest.raises(FileNotFoundError, match='no_such_file.csv'):
            TemperatureCorrection.from_file(tmp_path / 'no_such_file.csv')

    def test_a_single_height_is_rejected(self, tmp_path):
        one_column = 'time,1.0\n0,1.0\n100,0.8\n'

        with pytest.raises(ValueError, match='1 height'):
            TemperatureCorrection.from_file(write_factors(tmp_path, one_column))

    def test_a_single_time_is_rejected(self, tmp_path):
        one_row = 'time,1.0,2.0\n0,1.0,1.0\n'

        with pytest.raises(ValueError, match='1 time'):
            TemperatureCorrection.from_file(write_factors(tmp_path, one_row))

    def test_a_repeated_time_is_rejected(self, tmp_path):
        repeated_time = 'time,1.0,2.0\n0,1.0,1.0\n0,0.9,0.9\n100,0.8,0.6\n'

        with pytest.raises(ValueError, match='same time value twice'):
            TemperatureCorrection.from_file(write_factors(tmp_path, repeated_time))

    def test_a_height_that_is_not_a_number_is_rejected(self, tmp_path):
        text_header = 'time,bottom,top\n0,1.0,1.0\n100,0.8,0.6\n'

        with pytest.raises(ValueError, match='height values .* are not numbers'):
            TemperatureCorrection.from_file(write_factors(tmp_path, text_header))

    def test_a_missing_factor_is_rejected(self, tmp_path):
        gap = 'time,1.0,2.0\n0,1.0,\n100,0.8,0.6\n'

        with pytest.raises(ValueError, match='empty or not finite'):
            TemperatureCorrection.from_file(write_factors(tmp_path, gap))

    def test_a_factor_of_zero_is_rejected(self, tmp_path):
        # the intensities are divided by the factors
        zero = 'time,1.0,2.0\n0,1.0,1.0\n100,0.0,0.6\n'

        with pytest.raises(ValueError, match='not positive'):
            TemperatureCorrection.from_file(write_factors(tmp_path, zero))


class TestApplyingTheCorrection:
    def test_intensities_are_divided_by_the_factor_of_their_time_and_height(self, tmp_path):
        correction = TemperatureCorrection.from_file(write_factors(tmp_path))
        solver = make_solver(tmp_path, correction)

        solver.apply_temperature_correction()

        expected = [corrected_intensity(img_id, led_id)
                    for img_id, led_id in solver.calculated_img_data.index]
        assert solver.calculated_img_data[REFERENCE_PROPERTY].to_list() == pytest.approx(expected)

    def test_an_led_outside_the_tabulated_heights_keeps_its_intensity(self, tmp_path):
        correction = TemperatureCorrection.from_file(write_factors(tmp_path))
        solver = make_solver(tmp_path, correction)
        # the first LED sits on the lower edge of the table, the second one above its top
        solver.experiment.leds = [LedStub(0, 1.0), LedStub(1, 9.0)]

        with pytest.warns(UserWarning, match='left uncorrected'):
            solver.apply_temperature_correction()

        intensities = solver.calculated_img_data[REFERENCE_PROPERTY]
        assert intensities.xs(1, level='led_id').to_list() == [INTENSITY] * len(IMG_TIMES)
        assert intensities.xs(0, level='led_id').to_list() == pytest.approx(
            [corrected_intensity(img_id, 0) for img_id in sorted(IMG_TIMES)])

    def test_the_reference_intensities_are_corrected_as_well(self, tmp_path):
        # I0 is averaged over corrected intensities, so the factors of the reference images cancel
        # out of the relative intensities and a reference image stays at a relative intensity of 1
        correction = TemperatureCorrection.from_file(write_factors(tmp_path))
        solver = make_solver(tmp_path, correction, ref_img_indices=[3])

        solver.apply_temperature_correction()
        solver.calc_and_set_ref_intensities()

        intensities_of_img_3 = solver.calculated_img_data.loc[3, REFERENCE_PROPERTY].to_numpy()
        assert intensities_of_img_3 / solver.ref_intensities == pytest.approx([1.0, 1.0])

    def test_without_a_correction_the_intensities_are_unchanged(self, tmp_path):
        solver = make_solver(tmp_path, correction=None)

        solver.set_all_member_variables(save_distances=False)

        assert solver.calculated_img_data[REFERENCE_PROPERTY].to_list() == [INTENSITY] * 6

    def test_the_factors_are_divided_out_once_when_the_variables_are_set_twice(self, tmp_path):
        correction = TemperatureCorrection.from_file(write_factors(tmp_path))
        solver = make_solver(tmp_path, correction)

        solver.set_all_member_variables(save_distances=False)
        solver.set_all_member_variables(save_distances=False)

        expected = [corrected_intensity(img_id, led_id)
                    for img_id, led_id in solver.calculated_img_data.index]
        assert solver.calculated_img_data[REFERENCE_PROPERTY].to_list() == pytest.approx(expected)

    def test_the_factors_are_divided_out_once_when_the_correction_is_applied_twice(self, tmp_path):
        correction = TemperatureCorrection.from_file(write_factors(tmp_path))
        solver = make_solver(tmp_path, correction)

        solver.apply_temperature_correction()
        solver.apply_temperature_correction()

        expected = [corrected_intensity(img_id, led_id)
                    for img_id, led_id in solver.calculated_img_data.index]
        assert solver.calculated_img_data[REFERENCE_PROPERTY].to_list() == pytest.approx(expected)

    def test_data_loaded_by_the_caller_are_corrected_as_well(self, tmp_path):
        # set_all_member_variables only loads the data if they are not there yet, so the correction
        # must not hang off the loading
        correction = TemperatureCorrection.from_file(write_factors(tmp_path))
        solver = make_solver(tmp_path, correction)
        assert not solver.calculated_img_data.empty

        solver.set_all_member_variables(save_distances=False)

        assert solver.temperature_corrected
        assert solver.calculated_img_data[REFERENCE_PROPERTY].to_list() != [INTENSITY] * 6

    def test_an_led_without_coordinates_is_named_in_the_error(self, tmp_path):
        correction = TemperatureCorrection.from_file(write_factors(tmp_path))
        solver = make_solver(tmp_path, correction)
        solver.experiment.leds = [LedStub(0, LED_HEIGHTS[0])]

        with pytest.raises(ValueError, match=r'LEDs \[1\] .* have no coordinates'):
            solver.apply_temperature_correction()

    def test_an_image_without_a_time_is_named_in_the_error(self, tmp_path):
        correction = TemperatureCorrection.from_file(write_factors(tmp_path))
        solver = make_solver(tmp_path, correction)
        pd.DataFrame({'#ID': [1, 2], 'Experiment_Time[s]': [0.0, 50.0]}).to_csv(
            tmp_path / 'analysis' / 'image_infos_analysis.csv', index=False)

        with pytest.raises(ValueError, match=r'images \[3\] .* no experiment time'):
            solver.apply_temperature_correction()


class TestTheNoteInTheResultFile:
    def test_the_note_names_the_corrected_channel_and_its_factors(self, tmp_path):
        correction = TemperatureCorrection.from_file(write_factors(tmp_path))
        solver = make_solver(tmp_path, correction)

        assert solver.temperature_correction_info() == ('channel 0 from reduction_factors.csv '
                                                        '(0.0 s to 100.0 s, 1.00 m to 2.00 m)')
        assert 'Temperature_correction: channel 0 from reduction_factors.csv' in str(solver)

    def test_the_note_states_no_correction_for_an_uncorrected_channel(self, tmp_path):
        solver = make_solver(tmp_path, correction=None)

        assert solver.temperature_correction_info() == 'None'
        assert 'Temperature_correction: None' in str(solver)


class TestLoadingTheConfiguredFiles:
    def test_only_the_channels_with_a_file_are_corrected(self, tmp_path):
        file_path = write_factors(tmp_path)

        corrections = load_temperature_corrections({0: str(file_path), 1: None})

        assert list(corrections) == [0]
        assert corrections[0].source == 'reduction_factors.csv'

    def test_a_missing_file_stops_the_run_before_the_analysis(self, tmp_path):
        with pytest.raises(FileNotFoundError, match='no_such_file.csv'):
            load_temperature_corrections({0: str(tmp_path / 'no_such_file.csv')})


class TestTheConfigOption:
    def test_the_template_asks_for_no_correction(self, tmp_path, monkeypatch):
        monkeypatch.chdir(tmp_path)
        ConfigDataAnalysis(load_config_file=False)

        config = ConfigDataAnalysis()

        assert [config.get_reduction_file(channel) for channel in (0, 1, 2)] == [None] * 3

    def test_a_configured_path_is_returned(self, tmp_path, monkeypatch):
        monkeypatch.chdir(tmp_path)
        ConfigDataAnalysis(load_config_file=False, reduction_files={1: 'factors_green.csv'})

        config = ConfigDataAnalysis()

        assert config.get_reduction_file(1) == 'factors_green.csv'
        assert config.get_reduction_file(0) is None

    def test_a_config_written_before_the_correction_asks_for_none(self, tmp_path, monkeypatch):
        # the section is missing entirely in a config file of an older LEDSA version
        monkeypatch.chdir(tmp_path)
        config_file = tmp_path / 'config_analysis.ini'
        config_file.write_text('[DEFAULT]\nreference_property = sum_col_val\n', encoding='utf-8')

        config = ConfigDataAnalysis()

        assert [config.get_reduction_file(channel) for channel in (0, 1, 2)] == [None] * 3

    def test_the_help_texts_of_the_section_survive_saving(self, tmp_path, monkeypatch):
        monkeypatch.chdir(tmp_path)
        ConfigDataAnalysis(load_config_file=False)
        created = (tmp_path / 'config_analysis.ini').read_text(encoding='utf-8')

        ConfigDataAnalysis().save()

        saved = (tmp_path / 'config_analysis.ini').read_text(encoding='utf-8')
        assert 'reduction factor file' in created
        assert [line for line in created.splitlines() if 'reduction factor file' in line] == \
               [line for line in saved.splitlines() if 'reduction factor file' in line]
