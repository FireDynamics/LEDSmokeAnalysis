"""Unit tests for analysing several quantities of step 3 in one analysis run.

Step 3 writes several intensity quantities per LED, and the reference_property option names the
ones the extinction coefficients are computed from. Every quantity is computed into a directory of
its own, so that the results of one run do not overwrite each other.
"""
import os

import numpy as np
import pytest

from ledsa.analysis.ConfigDataAnalysis import (ConfigDataAnalysis, format_reference_properties,
                                               parse_reference_properties)
from ledsa.analysis.ConfigDataStacked import ConfigDataStacked
from ledsa.analysis.Experiment import Layers
from ledsa.analysis.ExtinctionCoefficients import ExtinctionCoefficients
from ledsa.core.file_handling import get_extinction_coefficients_file_path
from ledsa.postprocessing.simulation import SimData, _BaseSimData

IMAGE_INFOS = """\
#ID,Name,Exif_Time,Experiment_Time[s]
1,test_img_1.jpg,12:00:00,0.0
2,test_img_2.jpg,12:00:01,1.0
"""


class TestParseReferenceProperties:
    def test_single_quantity_is_kept(self):
        assert parse_reference_properties('sum_col_val') == ['sum_col_val']

    def test_quantities_are_split_on_whitespace(self):
        assert parse_reference_properties('sum_col_val  bgsub_sum_col_val\tsidebg_sum_col_val') == [
            'sum_col_val', 'bgsub_sum_col_val', 'sidebg_sum_col_val']

    def test_order_of_the_option_is_kept(self):
        assert parse_reference_properties('bgsub_sum_col_val sum_col_val') == [
            'bgsub_sum_col_val', 'sum_col_val']

    def test_repeated_quantity_is_dropped(self):
        # both would be written into the same file
        assert parse_reference_properties('sum_col_val bgsub_sum_col_val sum_col_val') == [
            'sum_col_val', 'bgsub_sum_col_val']

    @pytest.mark.parametrize('value', ['', '   ', 'None', 'none'])
    def test_option_without_a_quantity_raises(self, value):
        with pytest.raises(ValueError, match='names no quantity'):
            parse_reference_properties(value)

    def test_list_is_written_as_one_value(self):
        assert format_reference_properties(['sum_col_val', 'bgsub_sum_col_val']) == \
               'sum_col_val bgsub_sum_col_val'
        assert format_reference_properties('sum_col_val') == 'sum_col_val'


class TestConfigDataAnalysis:
    def test_template_holds_the_given_quantities(self, tmp_path, monkeypatch):
        monkeypatch.chdir(tmp_path)
        ConfigDataAnalysis(load_config_file=False,
                           reference_property=['sum_col_val', 'bgsub_sum_col_val'])
        assert ConfigDataAnalysis().get_reference_properties() == ['sum_col_val', 'bgsub_sum_col_val']

    def test_single_quantity_of_an_older_config_is_read(self, tmp_path, monkeypatch):
        # configs written before several quantities could be given hold a single name
        monkeypatch.chdir(tmp_path)
        config_text = '[DEFAULT]\nreference_property = sum_col_val\n'
        (tmp_path / 'config_analysis.ini').write_text(config_text, encoding='utf-8')
        assert ConfigDataAnalysis().get_reference_properties() == ['sum_col_val']

    def test_missing_option_falls_back_to_the_default_quantity(self, tmp_path, monkeypatch):
        monkeypatch.chdir(tmp_path)
        (tmp_path / 'config_analysis.ini').write_text('[DEFAULT]\nnum_cores = 1\n', encoding='utf-8')
        assert ConfigDataAnalysis().get_reference_properties() == ['sum_col_val']


class TestConfigDataStacked:
    def write_config(self, tmp_path, options):
        text = ('[DEFAULT]\n' + options + '\n[simulation_0]\npath = /data/cam0\n'
                'camera_position = 1.0 0.0 1.5\n\n[led_arrays]\n0 = 0:0\n')
        path = tmp_path / ConfigDataStacked.DEFAULT_FILENAME
        path.write_text(text, encoding='utf-8')
        return ConfigDataStacked(filename=str(path))

    def test_several_quantities_are_read(self, tmp_path):
        config = self.write_config(tmp_path, 'reference_property = sum_col_val sidebg_sum_col_val\n')
        assert config.reference_properties == ['sum_col_val', 'sidebg_sum_col_val']
        assert config.reference_property == 'sum_col_val'

    def test_unset_option_falls_back_to_the_default_quantity(self, tmp_path):
        config = self.write_config(tmp_path, 'num_cores = 1\n')
        assert config.reference_properties == ['sum_col_val']


class SolverStub(ExtinctionCoefficients):
    """Smallest concrete solver, since only the file the results are written to is of interest."""

    def calc_coefficients_of_img(self, rel_intensities: np.ndarray) -> np.ndarray:
        return rel_intensities


class ExperimentStub:
    def __init__(self, path, channel=0, led_array=0):
        self.path = path
        self.channel = channel
        self.led_array = led_array
        self.layers = Layers(2, 0.0, 2.0)

    def __str__(self):
        return 'experiment\n'


def save_coefficients(tmp_path, reference_property, solver='linear', channel=0, led_array=0):
    """Write extinction coefficients of two images as one analysis step does."""
    coefficients = SolverStub(ExperimentStub(tmp_path, channel, led_array),
                              reference_property=reference_property)
    coefficients.solver = solver
    coefficients.analysed_img_ids = [1, 2]
    coefficients.coefficients_per_image_and_layer = [np.array([0.1, 0.2]), np.array([0.3, 0.4])]
    coefficients.save()


class TestResultPath:
    def test_file_lies_in_the_directory_of_its_solver_and_quantity(self, tmp_path):
        path = get_extinction_coefficients_file_path('linear', 1, 'sum_col_val', 2, root=tmp_path)
        assert path.parent == tmp_path / 'analysis' / 'extinction_coefficients' / 'linear' / 'sum_col_val'
        assert path.name == 'extinction_coefficients_linear_channel_1_sum_col_val_led_array_2.csv'

    def test_quantities_of_one_solver_are_told_apart(self, tmp_path):
        sum_path = get_extinction_coefficients_file_path('linear', 0, 'sum_col_val', 0, root=tmp_path)
        bgsub_path = get_extinction_coefficients_file_path('linear', 0, 'bgsub_sum_col_val', 0,
                                                           root=tmp_path)
        assert sum_path.parent != bgsub_path.parent


class TestSave:
    def setup_analysis_dir(self, tmp_path, monkeypatch):
        """The experiment times are read from the image infos of the working directory."""
        monkeypatch.chdir(tmp_path)
        analysis_dir = tmp_path / 'analysis'
        analysis_dir.mkdir(parents=True, exist_ok=True)
        (analysis_dir / 'image_infos_analysis.csv').write_text(IMAGE_INFOS, encoding='utf-8')

    def test_results_are_written_into_the_directory_of_their_quantity(self, tmp_path, monkeypatch):
        self.setup_analysis_dir(tmp_path, monkeypatch)
        save_coefficients(tmp_path, 'sum_col_val')
        path = get_extinction_coefficients_file_path('linear', 0, 'sum_col_val', 0, root=tmp_path)
        assert path.is_file()
        assert 'Reference_property: sum_col_val' in path.read_text(encoding='utf-8')

    def test_quantities_of_one_run_do_not_overwrite_each_other(self, tmp_path, monkeypatch):
        self.setup_analysis_dir(tmp_path, monkeypatch)
        quantities = ['sum_col_val', 'bgsub_sum_col_val', 'sidebg_sum_col_val']
        for quantity in quantities:
            save_coefficients(tmp_path, quantity)
        for quantity in quantities:
            path = get_extinction_coefficients_file_path('linear', 0, quantity, 0, root=tmp_path)
            assert path.is_file(), f'{path} is missing'
            assert f'Reference_property: {quantity}' in path.read_text(encoding='utf-8')

    def test_solvers_of_one_quantity_do_not_overwrite_each_other(self, tmp_path, monkeypatch):
        self.setup_analysis_dir(tmp_path, monkeypatch)
        for solver in ['linear', 'nonlinear']:
            save_coefficients(tmp_path, 'sum_col_val', solver=solver)
        for solver in ['linear', 'nonlinear']:
            assert get_extinction_coefficients_file_path(solver, 0, 'sum_col_val', 0,
                                                         root=tmp_path).is_file()


class TestPostprocessingFileDiscovery:
    """The postprocessing reads the coefficients of one quantity, also from an older result dir."""

    def write_result_file(self, directory, reference_property, channel=0, led_array=0,
                          solver='linear'):
        directory.mkdir(parents=True, exist_ok=True)
        name = (f'extinction_coefficients_{solver}_channel_{channel}_{reference_property}'
                f'_led_array_{led_array}.csv')
        (directory / name).write_text('# header\n', encoding='utf-8')
        return name

    def make_sim_data(self, reference_property='sum_col_val', solver='linear'):
        sim_data = _BaseSimData()
        sim_data.solver = solver
        sim_data.reference_property = reference_property
        return sim_data

    def test_only_the_files_of_the_own_quantity_are_read(self, tmp_path):
        solver_dir = tmp_path / 'analysis' / 'extinction_coefficients' / 'linear'
        wanted = self.write_result_file(solver_dir / 'sum_col_val', 'sum_col_val')
        self.write_result_file(solver_dir / 'bgsub_sum_col_val', 'bgsub_sum_col_val')

        files = self.make_sim_data('sum_col_val')._extco_files(str(tmp_path))

        assert [os.path.basename(file) for file in files] == [wanted]

    def test_results_of_an_earlier_version_are_still_found(self, tmp_path):
        # before every quantity got a directory, the files lay directly in the one of the solver
        solver_dir = tmp_path / 'analysis' / 'extinction_coefficients' / 'linear'
        wanted = self.write_result_file(solver_dir, 'sum_col_val')

        files = self.make_sim_data('sum_col_val')._extco_files(str(tmp_path))

        assert [os.path.basename(file) for file in files] == [wanted]

    def test_directory_of_the_quantity_wins_over_the_older_layout(self, tmp_path):
        solver_dir = tmp_path / 'analysis' / 'extinction_coefficients' / 'linear'
        self.write_result_file(solver_dir, 'sum_col_val', led_array=1)
        wanted = self.write_result_file(solver_dir / 'sum_col_val', 'sum_col_val')

        files = self.make_sim_data('sum_col_val')._extco_files(str(tmp_path))

        assert [os.path.basename(file) for file in files] == [wanted]

    def test_missing_results_of_the_quantity_are_reported(self, tmp_path):
        analysis_dir = tmp_path / 'analysis'
        analysis_dir.mkdir(parents=True)
        (analysis_dir / 'image_infos_analysis.csv').write_text(IMAGE_INFOS, encoding='utf-8')
        sim_data = SimData(str(tmp_path), read_all=False, load_config_params=False,
                           reference_property='bgsub_sum_col_val')
        sim_data.solver = 'linear'

        with pytest.raises(FileNotFoundError, match='bgsub_sum_col_val'):
            sim_data._get_extco_df_from_path()


class TestMissingReferenceProperty:
    """A quantity that is not in the binary data has to be named instead of raising a bare KeyError.

    The color corrected quantities are the realistic case: the analysis writes sum_col_val_cc into the
    binary data itself, so rebuilding them from the step 3 result files drops the column again.
    """

    @pytest.fixture
    def coefficients(self, monkeypatch, tmp_path):
        import pandas as pd

        import ledsa.analysis.ExtinctionCoefficients as extinction_coefficients
        index = pd.MultiIndex.from_product([[1, 2], [0, 1]], names=['img_id', 'led_id'])
        img_data = pd.DataFrame({'led_array_id': 0, 'sum_col_val': 100.0}, index=index)
        monkeypatch.setattr(extinction_coefficients, 'read_hdf', lambda channel, path: img_data)
        return SolverStub(ExperimentStub(tmp_path))

    def test_a_color_corrected_quantity_points_at_the_cc_flag(self, coefficients):
        coefficients.reference_property = 'sum_col_val_cc'
        with pytest.raises(ValueError, match=r'sum_col_val_cc is not among the columns .*--cc'):
            coefficients.load_img_data()

    def test_an_unknown_quantity_points_at_the_option(self, coefficients):
        coefficients.reference_property = 'sum_col_va'
        with pytest.raises(ValueError, match=r'sum_col_va is not among the columns .*reference_property'):
            coefficients.load_img_data()

    def test_the_columns_that_are_there_are_named(self, coefficients):
        # the quantity may simply have been written by a newer step 3 than the one that ran
        coefficients.reference_property = 'bgsub_sum_col_val'
        with pytest.raises(ValueError, match=r'led_array_id, sum_col_val'):
            coefficients.load_img_data()

    def test_a_quantity_that_is_there_is_loaded(self, coefficients):
        coefficients.reference_property = 'sum_col_val'
        coefficients.load_img_data()
        assert coefficients.analysed_img_ids == [1, 2]
