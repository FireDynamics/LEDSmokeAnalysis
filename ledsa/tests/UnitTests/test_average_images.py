"""Unit tests for the average_images option of config_analysis.ini.

With average_images set, the intensities of consecutive images are summarized into one averaged
image, while the reference images at the beginning of the experiment are kept as they are. The
averaged intensities go into a binary of their own and the averaged images are numbered from 1
again, so the averaged image infos have to carry the same image IDs and the same number of rows.
"""
import os

import numpy as np
import pandas as pd
import pytest

from ledsa.analysis.ConfigDataAnalysis import ConfigDataAnalysis
from ledsa.analysis.Experiment import Layers
from ledsa.analysis.ExperimentData import ExperimentData
from ledsa.analysis.ExtinctionCoefficients import ExtinctionCoefficients
from ledsa.core.ConfigData import ConfigData
from ledsa.core.file_handling import (average_all_fitpar, create_analysis_infos_avg,
                                      get_binary_file_path, read_hdf_avg)
# the fixtures that build the step 3 result files of a channel, so that the averaged data can be
# read over the real files instead of a binary written by hand
from ledsa.tests.UnitTests.test_file_handling import BinaryDataFiles

NUM_LEDS = 2
NUM_IMGS = 6
NUM_REF_IMGS = 2
N_SUMMARIZE = 2
# the reference images are kept, the remaining four are summarized two by two
NUM_AVG_IMGS = NUM_REF_IMGS + (NUM_IMGS - NUM_REF_IMGS) // N_SUMMARIZE


def write_binary_data(tmp_path, channel=0, intensity_offset=0):
    """Write the binary of the single images as the data extraction leaves it behind."""
    rows = [{'img_id': img_id, 'led_id': led_id, 'led_array_id': 0,
             'sum_col_val': 100 * img_id + intensity_offset,
             'mean_col_val': float(img_id)}
            for img_id in range(1, NUM_IMGS + 1) for led_id in range(NUM_LEDS)]
    file_path = get_binary_file_path(channel, path=tmp_path)
    os.makedirs(os.path.dirname(file_path), exist_ok=True)
    pd.DataFrame(rows).to_hdf(file_path, key='channel_values', format='table')
    return file_path


def write_image_infos(tmp_path):
    """Write the image infos of the single images as the data extraction leaves them behind."""
    analysis_dir = tmp_path / 'analysis'
    analysis_dir.mkdir(parents=True, exist_ok=True)
    lines = ['#ID,Name,Time[s],Experiment_Time[s]']
    lines += [f'{img_id},img{img_id}.jpg,{float(img_id - 1)},{float(img_id - 1)}'
              for img_id in range(1, NUM_IMGS + 1)]
    (analysis_dir / 'image_infos_analysis.csv').write_text('\n'.join(lines) + '\n', encoding='utf-8')


def _set_config_age_against_the_averaged_binary(tmp_path, seconds, channel=0):
    """Give config_analysis.ini a modification time relative to the one of the averaged binary.

    The modification times of two files written one after the other can land in the same clock
    tick, which would make the age of the config depend on the file system.
    """
    averaged_mtime = os.path.getmtime(get_binary_file_path(channel, path=tmp_path, averaged=True))
    os.utime(tmp_path / 'config_analysis.ini', (averaged_mtime + seconds, averaged_mtime + seconds))


def read_averaged(tmp_path, channel=0):
    return read_hdf_avg(channel, path=tmp_path, n_summarize=N_SUMMARIZE, num_ref_imgs=NUM_REF_IMGS)


class TestAveragedBinary:

    def test_the_averaged_binary_is_written_and_read_back(self, tmp_path):
        write_binary_data(tmp_path)
        averaged = read_averaged(tmp_path)
        assert os.path.isfile(get_binary_file_path(0, path=tmp_path, averaged=True))
        assert sorted(averaged.index.get_level_values('img_id').unique()) == \
               list(range(1, NUM_AVG_IMGS + 1))
        assert len(averaged) == NUM_AVG_IMGS * NUM_LEDS

    def test_the_reference_images_are_kept_unaveraged(self, tmp_path):
        write_binary_data(tmp_path)
        averaged = read_averaged(tmp_path)
        for img_id in range(1, NUM_REF_IMGS + 1):
            assert averaged.loc[(img_id, 0), 'mean_col_val'] == float(img_id)

    def test_the_remaining_images_are_summarized(self, tmp_path):
        write_binary_data(tmp_path)
        averaged = read_averaged(tmp_path)
        # the first averaged image holds the mean of the images 3 and 4, the second the one of 5 and 6
        assert averaged.loc[(NUM_REF_IMGS + 1, 0), 'mean_col_val'] == 3.5
        assert averaged.loc[(NUM_REF_IMGS + 2, 0), 'mean_col_val'] == 5.5

    def test_the_unaveraged_binary_is_not_returned(self, tmp_path):
        # read_hdf_avg read the binary of the single images and only averaged when it was missing,
        # so an existing binary made it return the unaveraged data without a word
        write_binary_data(tmp_path)
        averaged = read_averaged(tmp_path)
        assert len(averaged) < NUM_IMGS * NUM_LEDS
        assert max(averaged.index.get_level_values('img_id')) == NUM_AVG_IMGS

    def test_the_averaged_binary_of_an_earlier_run_is_replaced(self, tmp_path, capsys):
        # appending kept every LED of the earlier run next to the new one, so the rows piled up
        # with every run and the analysis saw each averaged image twice
        write_binary_data(tmp_path)
        average_all_fitpar(0, n_summarize=N_SUMMARIZE, num_ref_imgs=NUM_REF_IMGS, path=tmp_path)
        rows_of_the_first_run = len(read_averaged(tmp_path))
        average_all_fitpar(0, n_summarize=N_SUMMARIZE, num_ref_imgs=NUM_REF_IMGS, path=tmp_path)
        assert len(read_averaged(tmp_path)) == rows_of_the_first_run
        assert 'is replaced' in capsys.readouterr().out

    def test_a_replaced_averaged_binary_holds_the_new_values(self, tmp_path):
        write_binary_data(tmp_path)
        average_all_fitpar(0, n_summarize=N_SUMMARIZE, num_ref_imgs=NUM_REF_IMGS, path=tmp_path)
        write_binary_data(tmp_path, intensity_offset=1000)
        average_all_fitpar(0, n_summarize=N_SUMMARIZE, num_ref_imgs=NUM_REF_IMGS, path=tmp_path)
        assert read_averaged(tmp_path).loc[(1, 0), 'sum_col_val'] == 1100

    def test_the_averaged_binary_is_rebuilt_after_a_new_extraction(self, tmp_path):
        write_binary_data(tmp_path)
        read_averaged(tmp_path)
        single_img_file_path = write_binary_data(tmp_path, intensity_offset=1000)
        # the averaged binary of the earlier run is older than the data it was averaged from
        averaged_file_path = get_binary_file_path(0, path=tmp_path, averaged=True)
        older = os.path.getmtime(single_img_file_path) - 10
        os.utime(averaged_file_path, (older, older))
        assert read_averaged(tmp_path).loc[(1, 0), 'sum_col_val'] == 1100

    def test_a_changed_analysis_config_rebuilds_the_averaged_binary(self, tmp_path, capsys):
        # num_ref_images and n_summarize come out of config_analysis.ini and decide which images are
        # summarized into which, so averages computed with the counts of an earlier run must not be
        # answered out of the cache after the configuration has changed
        write_binary_data(tmp_path)
        read_averaged(tmp_path)
        config_path = tmp_path / 'config_analysis.ini'
        config_path.write_text('[DEFAULT]\n   n_summarize = 3\n', encoding='utf-8')
        _set_config_age_against_the_averaged_binary(tmp_path, 10)
        capsys.readouterr()

        read_averaged(tmp_path)

        assert 'of an earlier run is replaced' in capsys.readouterr().out

    def test_an_untouched_analysis_config_does_not_rebuild_it(self, tmp_path, capsys):
        # averaging a real experiment costs the loading time of the whole channel, so a run that
        # changed nothing is answered out of the cache
        write_binary_data(tmp_path)
        read_averaged(tmp_path)
        config_path = tmp_path / 'config_analysis.ini'
        config_path.write_text('[DEFAULT]\n   n_summarize = 2\n', encoding='utf-8')
        _set_config_age_against_the_averaged_binary(tmp_path, -10)
        capsys.readouterr()

        read_averaged(tmp_path)

        assert 'of an earlier run is replaced' not in capsys.readouterr().out

    def test_the_channels_do_not_overwrite_each_other(self, tmp_path):
        write_binary_data(tmp_path, channel=0)
        write_binary_data(tmp_path, channel=1, intensity_offset=1000)
        assert read_averaged(tmp_path, channel=0).loc[(1, 0), 'sum_col_val'] == 100
        assert read_averaged(tmp_path, channel=1).loc[(1, 0), 'sum_col_val'] == 1100


class TestAveragedImageInfos:

    def test_the_image_ids_of_the_binary_and_the_image_infos_agree(self, tmp_path):
        write_binary_data(tmp_path)
        write_image_infos(tmp_path)
        create_analysis_infos_avg(n_summarize=N_SUMMARIZE, num_ref_imgs=NUM_REF_IMGS, path=tmp_path)
        image_infos = pd.read_csv(tmp_path / 'analysis' / 'image_infos_analysis_avg.csv')
        averaged = read_averaged(tmp_path)
        assert list(image_infos['#ID']) == \
               sorted(averaged.index.get_level_values('img_id').unique())

    def test_the_summarized_images_carry_the_mean_time_and_both_names(self, tmp_path):
        write_image_infos(tmp_path)
        create_analysis_infos_avg(n_summarize=N_SUMMARIZE, num_ref_imgs=NUM_REF_IMGS, path=tmp_path)
        image_infos = pd.read_csv(tmp_path / 'analysis' / 'image_infos_analysis_avg.csv')
        assert len(image_infos) == NUM_AVG_IMGS
        reference_images = image_infos.iloc[:NUM_REF_IMGS]
        assert list(reference_images['Name']) == ['img1.jpg', 'img2.jpg']
        assert list(reference_images['Experiment_Time[s]']) == [0.0, 1.0]
        first_averaged = image_infos.iloc[NUM_REF_IMGS]
        assert first_averaged['Name'] == 'img3.jpg/img4.jpg'
        assert first_averaged['Experiment_Time[s]'] == 2.5

    def test_the_file_has_the_columns_of_the_single_images(self, tmp_path):
        write_image_infos(tmp_path)
        create_analysis_infos_avg(n_summarize=N_SUMMARIZE, num_ref_imgs=NUM_REF_IMGS, path=tmp_path)
        infos_path = tmp_path / 'analysis' / 'image_infos_analysis_avg.csv'
        header = infos_path.read_text(encoding='utf-8').splitlines()[0]
        assert header == '#ID,Name,Experiment_Time[s]'


class SolverStub(ExtinctionCoefficients):
    """Smallest concrete solver, since only the loaded image data are of interest."""

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


class TestLoadImgData:

    def make_solver(self, tmp_path, **kwargs):
        return SolverStub(ExperimentStub(tmp_path), num_ref_imgs=NUM_REF_IMGS,
                          average_images=True, **kwargs)

    def test_the_averaged_data_are_loaded(self, tmp_path, monkeypatch):
        write_binary_data(tmp_path)
        write_image_infos(tmp_path)
        monkeypatch.chdir(tmp_path)
        solver = self.make_solver(tmp_path)
        solver.load_img_data()
        assert solver.analysed_img_ids == list(range(1, NUM_AVG_IMGS + 1))

    def test_the_image_infos_are_written_for_the_same_images(self, tmp_path, monkeypatch):
        write_binary_data(tmp_path)
        write_image_infos(tmp_path)
        monkeypatch.chdir(tmp_path)
        solver = self.make_solver(tmp_path)
        solver.load_img_data()
        image_infos = pd.read_csv(tmp_path / 'analysis' / 'image_infos_analysis_avg.csv')
        assert list(image_infos['#ID']) == solver.analysed_img_ids

    def test_the_number_of_summarized_images_reaches_the_averaging(self, tmp_path, monkeypatch):
        # load_img_data handed on the reference image count only, so n_summarize stayed at the two
        # images per group the averaging was written with, whatever the configuration asked for
        write_binary_data(tmp_path)
        write_image_infos(tmp_path)
        monkeypatch.chdir(tmp_path)
        solver = self.make_solver(tmp_path, n_summarize=NUM_IMGS - NUM_REF_IMGS)

        solver.load_img_data()

        # the images after the reference images now form a single averaged image
        assert solver.analysed_img_ids == list(range(1, NUM_REF_IMGS + 2))
        image_infos = pd.read_csv(tmp_path / 'analysis' / 'image_infos_analysis_avg.csv')
        assert list(image_infos['#ID']) == solver.analysed_img_ids

    def test_explicit_reference_image_indices_are_refused(self, tmp_path, monkeypatch):
        # the averaging numbers the images from 1 again, so the indices would name other images
        write_binary_data(tmp_path)
        write_image_infos(tmp_path)
        monkeypatch.chdir(tmp_path)
        solver = self.make_solver(tmp_path, ref_img_indices=[5, 6])
        with pytest.raises(ValueError, match='average_images and ref_img_indices'):
            solver.load_img_data()


class TestAveragedBinaryOverTheResultFiles(BinaryDataFiles):
    """The averaged data are a cache of a cache, so a re-run of step 3 has to reach them too.

    The step 3 result files reach the averaged binary through the binary of the single images, and
    that one is rebuilt only when it is read. Comparing the two binaries with each other alone
    would therefore miss a data extraction run that nothing has read back yet.
    """

    # the fixture writes the images 1 and 2, so one reference image leaves one summarized image
    NUM_REF_IMGS_OF_THE_FIXTURE = 1

    def _read_averaged(self, tmp_path):
        return read_hdf_avg(0, path=tmp_path, n_summarize=N_SUMMARIZE,
                            num_ref_imgs=self.NUM_REF_IMGS_OF_THE_FIXTURE)

    def test_a_rerun_of_step_3_is_picked_up(self, analysis_dir):
        for img_id in (1, 2):
            self._write_result_file(analysis_dir, img_id, self.NUM_COLUMNS)
        # the offset of the re-run shifts every column of the result files, led_id included, so the
        # values are compared instead of the row of one LED
        assert sorted(self._read_averaged(analysis_dir)['sum_col_val']) == [2, 2, 12, 12]

        rerun_paths = [self._write_result_file(analysis_dir, img_id, self.NUM_COLUMNS, offset=100)
                       for img_id in (1, 2)]
        # only the result files moved; both binaries are untouched and the averaged one is the newer
        for path in rerun_paths:
            self._set_age_against_the_binary(analysis_dir, path, 10)

        assert sorted(self._read_averaged(analysis_dir)['sum_col_val']) == [102, 102, 112, 112]

    def test_an_untouched_averaged_binary_is_not_rebuilt(self, analysis_dir, capsys):
        for img_id in (1, 2):
            self._write_result_file(analysis_dir, img_id, self.NUM_COLUMNS)
        self._read_averaged(analysis_dir)
        self._make_all_sources_older(analysis_dir)
        capsys.readouterr()

        self._read_averaged(analysis_dir)

        assert 'is replaced' not in capsys.readouterr().out


class TestAveragedBinaryOfAnEarlierVersion:
    """Cases the averaged binary can be in that the comparison of the modification times misses."""

    def test_a_binary_written_under_the_old_key_is_recomputed(self, tmp_path):
        # Until the key was named, the averaged binary was written under 'table', the key read_hdf
        # still accepts for the binary of the single images. Asking for 'channel_values' ended the
        # whole analysis with a KeyError instead of computing the averages again.
        binary_path = write_binary_data(tmp_path)
        averaged_path = get_binary_file_path(0, path=tmp_path, averaged=True)
        pd.DataFrame({'img_id': [1], 'led_id': [0]}).to_hdf(averaged_path, key='table',
                                                            format='table')
        # a file older than the binary would be rebuilt for being stale, which is not the case here
        legacy_mtime = os.path.getmtime(binary_path) + 10
        os.utime(averaged_path, (legacy_mtime, legacy_mtime))
        assert len(read_averaged(tmp_path)) == NUM_AVG_IMGS * NUM_LEDS

    def test_a_binary_rebuilt_on_its_own_is_picked_up(self, tmp_path):
        # A run with average_images = False rebuilds the binary of the single images and leaves the
        # averaged one alone, so the averages have to follow it without a new extraction run.
        write_binary_data(tmp_path)
        values_of_the_first_run = read_averaged(tmp_path)['sum_col_val'].tolist()
        binary_path = write_binary_data(tmp_path, intensity_offset=100)
        averaged_path = get_binary_file_path(0, path=tmp_path, averaged=True)
        # the two writes can land in the same clock tick, so the age is set explicitly
        rebuilt_mtime = os.path.getmtime(averaged_path) + 10
        os.utime(binary_path, (rebuilt_mtime, rebuilt_mtime))
        assert read_averaged(tmp_path)['sum_col_val'].tolist() == [value + 100 for value
                                                                   in values_of_the_first_run]


def write_configs(**analysis_options):
    """Write a config.ini and a config_analysis.ini that ExperimentData can be loaded from.

    Only the options the loading asks for are filled in, the rest keeps the defaults of the
    template, so that nothing is requested from the user on the console.
    """
    ConfigData(load_config_file=False, merge_led_array_indices=None).save()
    config_analysis = ConfigDataAnalysis(load_config_file=False, **analysis_options)
    config_analysis.set('experiment_geometry', '   camera_position', '0 0 2')
    config_analysis.set('model_parameters', '   domain_bounds', '0 3')
    config_analysis.set('model_parameters', '   led_array_indices', '0')
    config_analysis.save()


class TestNSummarizeOption:
    """n_summarize is an option of config_analysis.ini and arrives at the analysis from there.

    Both halves of the averaging took the number of summarized images as a parameter, but no config
    key named it, so the two images per group they default to could not be changed at all.
    """

    def test_the_option_is_written_and_read_back(self, tmp_path, monkeypatch):
        monkeypatch.chdir(tmp_path)
        write_configs(n_summarize=5)
        assert ConfigDataAnalysis(load_config_file=True)['DEFAULT']['n_summarize'] == '5'

    def test_the_option_reaches_the_experiment_data(self, tmp_path, monkeypatch):
        monkeypatch.chdir(tmp_path)
        write_configs(num_ref_images=20, average_images=True, n_summarize=5)
        experiment_data = ExperimentData()
        assert (experiment_data.n_summarize, experiment_data.average_images,
                experiment_data.num_ref_images) == (5, True, 20)

    def test_a_config_written_before_the_option_keeps_the_former_behaviour(self, tmp_path, monkeypatch):
        # a config_analysis.ini of an earlier version names no count, so the analysis summarizes the
        # two images per group the averaging was hardcoded to
        monkeypatch.chdir(tmp_path)
        write_configs(average_images=True)
        config_path = tmp_path / 'config_analysis.ini'
        lines = [line for line in config_path.read_text(encoding='utf-8').splitlines()
                 if 'n_summarize' not in line]
        config_path.write_text('\n'.join(lines) + '\n', encoding='utf-8')
        assert ExperimentData().n_summarize == 2
