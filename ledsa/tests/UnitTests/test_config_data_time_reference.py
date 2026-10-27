"""Unit tests for the time reference image settings of the experiment configuration."""
import os
from datetime import datetime

import pytest

from ledsa.core.ConfigData import ConfigData
from ledsa.data_extraction.init_functions import request_config_parameters

IMG_TIME = datetime(2026, 10, 27, 12, 0, 5)
TIME_REF_IMG_TIME = '12:00:00'
TEMPLATE_DEFAULTS = dict(img_directory='imgs/', img_name_string='img{:04d}.jpg',
                         time_ref_img_time=TIME_REF_IMG_TIME, first_img_experiment_id=1,
                         last_img_experiment_id=4, ref_img_id=1, max_num_leds=10, num_arrays=1)


def create_config(tmp_path, monkeypatch, **options) -> ConfigData:
    """Write a config template in an empty working directory and return it loaded from the file.

    Loading it back is what the tests are about: the template writes the option names, every
    other part of LEDSA reads them.

    :param options: Template arguments overriding the defaults, e.g. time_ref_img_id.
    """
    monkeypatch.chdir(tmp_path)
    ConfigData(load_config_file=False, **dict(TEMPLATE_DEFAULTS, **options))
    return ConfigData()


def write_config(tmp_path, monkeypatch, default_options='') -> ConfigData:
    """Write a config carrying only the given options of the ``[DEFAULT]`` section and load it.

    Unlike the template, this leaves out every option not named, which is how a config edited
    by hand can look.
    """
    monkeypatch.chdir(tmp_path)
    (tmp_path / 'config.ini').write_text('[DEFAULT]\n' + default_options, encoding='utf-8')
    return ConfigData()


def patch_exif_datetime(monkeypatch, img_time=IMG_TIME) -> list:
    """Replace the EXIF reader with one that records which file it was asked for.

    :return: The list the requested file paths are appended to.
    """
    read_paths = []

    def fake_get_exif_datetime(filename, tag='DateTimeOriginal'):
        read_paths.append(filename)
        return img_time

    monkeypatch.setattr('ledsa.core.ConfigData.get_exif_datetime', fake_get_exif_datetime)
    return read_paths


def fail_on_input(monkeypatch) -> None:
    """Let a prompt fail the test, so an option already in the config cannot be asked for again."""
    def fail(prompt=''):
        raise AssertionError(f'asked for an option that is set in the config: {prompt}')

    monkeypatch.setattr('builtins.input', fail)


class TestTemplate:
    def test_the_id_is_written_under_the_name_the_code_reads(self, tmp_path, monkeypatch):
        config = create_config(tmp_path, monkeypatch, time_ref_img_id=5)

        assert config.get('DEFAULT', 'time_ref_img_id') == '5'

    def test_the_id_is_not_written_under_any_other_name(self, tmp_path, monkeypatch):
        config = create_config(tmp_path, monkeypatch, time_ref_img_id=5)

        assert 'time_img_id' not in config['DEFAULT']

    def test_a_config_with_the_id_is_not_asked_for_it_again(self, tmp_path, monkeypatch):
        config = create_config(tmp_path, monkeypatch, time_ref_img_id=5)
        patch_exif_datetime(monkeypatch)
        fail_on_input(monkeypatch)

        request_config_parameters(config)

        assert config.get('DEFAULT', 'time_ref_img_id') == '5'

    def test_a_config_with_the_file_is_not_asked_for_the_id(self, tmp_path, monkeypatch):
        config = create_config(tmp_path, monkeypatch, time_ref_img_file='clock.jpg')
        patch_exif_datetime(monkeypatch)
        fail_on_input(monkeypatch)

        request_config_parameters(config)

        assert config.get('DEFAULT', 'time_ref_img_file') == 'clock.jpg'


class TestTimeDiffFromTheReferenceImage:
    def test_the_id_names_the_image_through_the_image_name_string(self, tmp_path, monkeypatch):
        config = create_config(tmp_path, monkeypatch, time_ref_img_id=5)
        read_paths = patch_exif_datetime(monkeypatch)

        config.in_time_diff_to_img_time()

        assert read_paths == [os.path.abspath(os.path.join('imgs/', 'img0005.jpg'))]
        assert float(config['DEFAULT']['exif_time_infront_real_time']) == 5.0
        assert config['DEFAULT']['date'] == '2026:10:27'

    def test_the_file_names_the_image_without_an_id_in_the_config(self, tmp_path, monkeypatch):
        config = create_config(tmp_path, monkeypatch, time_ref_img_file='clock.jpg')
        read_paths = patch_exif_datetime(monkeypatch)

        config.in_time_diff_to_img_time()

        assert read_paths == [os.path.abspath('clock.jpg')]
        assert float(config['DEFAULT']['exif_time_infront_real_time']) == 5.0

    def test_the_file_wins_over_a_differing_id_and_is_reported(self, tmp_path, monkeypatch):
        config = create_config(tmp_path, monkeypatch, time_ref_img_id=5, time_ref_img_file='clock.jpg')
        read_paths = patch_exif_datetime(monkeypatch)

        with pytest.warns(UserWarning, match='time_ref_img_id'):
            config.in_time_diff_to_img_time()

        assert read_paths == [os.path.abspath('clock.jpg')]

    def test_an_option_missing_from_the_config_counts_as_unset(self, tmp_path, monkeypatch):
        config = write_config(tmp_path, monkeypatch,
                              'img_directory = imgs/\n'
                              'img_name_string = img{:04d}.jpg\n'
                              f'time_ref_img_time = {TIME_REF_IMG_TIME}\n')

        with pytest.raises(ValueError, match='time_ref_img_id'):
            config.in_time_diff_to_img_time()
