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

from ledsa.analysis.Experiment import Layers
from ledsa.analysis.ExtinctionCoefficients import ExtinctionCoefficients
from ledsa.core.file_handling import (average_all_fitpar, create_analysis_infos_avg,
                                      get_binary_file_path, read_hdf_avg)

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
        header = (tmp_path / 'analysis' / 'image_infos_analysis_avg.csv').read_text(
            encoding='utf-8').splitlines()[0]
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

    def test_explicit_reference_image_indices_are_refused(self, tmp_path, monkeypatch):
        # the averaging numbers the images from 1 again, so the indices would name other images
        write_binary_data(tmp_path)
        write_image_infos(tmp_path)
        monkeypatch.chdir(tmp_path)
        solver = self.make_solver(tmp_path, ref_img_indices=[5, 6])
        with pytest.raises(ValueError, match='average_images and ref_img_indices'):
            solver.load_img_data()
