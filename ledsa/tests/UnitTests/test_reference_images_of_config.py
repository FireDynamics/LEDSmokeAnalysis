"""Unit tests for the reference images config.ini names for the normalization.

ref_img_experiment_ids names the images the LED intensities are normalized with by the ID their
file name is built from, so that the data extraction knows them before the analysis IDs exist.
They are analysed whatever num_skip_imgs leaves out, and the analysis translates them into the
analysis IDs of the same images.
"""
import pytest

from ledsa.core.ConfigData import ConfigData
from ledsa.core.file_handling import _number_of_analysed_images
from ledsa.core.image_handling import get_analysis_img_ids
from ledsa.data_extraction.init_functions import _insert_ref_img_ids

CONFIG_WITHOUT_REFERENCE_IMAGES = """\
[DEFAULT]
img_name_string = img{:04d}.jpg
num_img_overflow = None

[analyse_photo]
first_img_analysis_id = 1
last_img_analysis_id = 7
num_skip_imgs = 2
"""


def write_config(tmp_path, monkeypatch, ref_img_experiment_ids=None) -> ConfigData:
    """Write a config by hand in an empty working directory and load it."""
    text = CONFIG_WITHOUT_REFERENCE_IMAGES
    if ref_img_experiment_ids is not None:
        text += f'ref_img_experiment_ids = {ref_img_experiment_ids}\n'
    monkeypatch.chdir(tmp_path)
    (tmp_path / 'config.ini').write_text(text, encoding='utf-8')
    return ConfigData()


def write_image_infos(tmp_path, img_ids) -> None:
    """Write the image infos of the analysed images as the data extraction leaves them behind."""
    analysis_dir = tmp_path / 'analysis'
    analysis_dir.mkdir(parents=True, exist_ok=True)
    lines = ['#ID,Name,Time[s],Experiment_Time[s]']
    lines += [f'{analysis_id},img{img_id:04d}.jpg,{float(analysis_id)},{float(analysis_id)}'
              for analysis_id, img_id in enumerate(img_ids, start=1)]
    (analysis_dir / 'image_infos_analysis.csv').write_text('\n'.join(lines) + '\n', encoding='utf-8')


class TestConfigOption:

    def test_the_named_images_are_read_as_a_list(self, tmp_path, monkeypatch):
        config = write_config(tmp_path, monkeypatch, '385 386 387')
        assert config.get_ref_img_experiment_ids() == [385, 386, 387]

    def test_none_names_no_images(self, tmp_path, monkeypatch):
        config = write_config(tmp_path, monkeypatch, 'None')
        assert config.get_ref_img_experiment_ids() is None

    def test_a_config_written_before_the_option_names_no_images(self, tmp_path, monkeypatch):
        config = write_config(tmp_path, monkeypatch)
        assert config.get_ref_img_experiment_ids() is None

    def test_something_other_than_image_ids_raises(self, tmp_path, monkeypatch):
        config = write_config(tmp_path, monkeypatch, '385, 386')
        with pytest.raises(ValueError, match='ref_img_experiment_ids'):
            config.get_ref_img_experiment_ids()

    def test_the_template_carries_the_option(self, tmp_path, monkeypatch):
        monkeypatch.chdir(tmp_path)
        ConfigData(load_config_file=False)
        assert ConfigData().get_ref_img_experiment_ids() is None


class TestInsertIntoTheAnalysedImages:
    """num_skip_imgs thins out the analysed images, the named ones are analysed all the same."""

    def test_a_skipped_reference_image_is_analysed(self):
        assert _insert_ref_img_ids(['0001', '0004', '0007'], [5, 6], first=1) == \
               ['0001', '0004', '0005', '0006', '0007']

    def test_a_reference_image_of_the_increment_is_not_analysed_twice(self):
        assert _insert_ref_img_ids(['0001', '0004', '0007'], [4], first=1) == \
               ['0001', '0004', '0007']

    def test_images_without_a_reference_image_are_left_alone(self):
        assert _insert_ref_img_ids(['0001', '0003'], None, first=1) == ['0001', '0003']
        assert _insert_ref_img_ids(['0001', '0003'], [], first=1) == ['0001', '0003']

    def test_a_reference_image_outside_the_analysed_range_raises(self):
        with pytest.raises(ValueError, match=r'\[9\]'):
            _insert_ref_img_ids(['0001', '0004', '0007'], [9], first=1)

    def test_image_ids_that_start_at_one_again_keep_their_order(self):
        # the IDs of the camera run over 9999 and start at 1 again in the middle of the experiment
        assert _insert_ref_img_ids(['9998', '0001'], [9999], first=9998) == \
               ['9998', '9999', '0001']


class TestTranslationIntoAnalysisIds:

    def test_the_analysis_ids_of_the_named_images_are_found(self, tmp_path, monkeypatch):
        # num_skip_imgs = 2 left out the images 2, 3, 5 and 6, which the reference images 5 and 6
        # were added to again
        write_image_infos(tmp_path, [1, 4, 5, 6, 7])
        monkeypatch.chdir(tmp_path)
        assert get_analysis_img_ids([5, 6], 'img{:04d}.jpg') == [3, 4]

    def test_an_image_that_was_not_analysed_raises(self, tmp_path, monkeypatch):
        write_image_infos(tmp_path, [1, 4, 7])
        monkeypatch.chdir(tmp_path)
        with pytest.raises(ValueError, match=r'\[5\]'):
            get_analysis_img_ids([5], 'img{:04d}.jpg')


class TestNumberOfAnalysedImages:
    """The binary data are built over the analysed images, the added ones included."""

    def test_the_image_infos_name_the_analysed_images(self, tmp_path, monkeypatch):
        config = write_config(tmp_path, monkeypatch, '5 6')
        # the increment leaves the images 1, 4 and 7, the reference images 5 and 6 are added
        write_image_infos(tmp_path, [1, 4, 5, 6, 7])
        assert _number_of_analysed_images(config) == 5

    def test_the_image_range_is_used_without_the_image_infos(self, tmp_path, monkeypatch):
        # every version before the image infos were read computed the number from the range and
        # the increment, which counts the images 1, 4 and 7 of the IDs 1 to 7
        config = write_config(tmp_path, monkeypatch)
        assert _number_of_analysed_images(config) == 3
