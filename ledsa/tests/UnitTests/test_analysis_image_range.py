"""Unit tests for the range of images the analysis covers.

The image list of the analysis used to be built from first_img_experiment_id, while the number
of result files to load was computed from first_img_analysis_id. Asking for the images 500 to
800 therefore extracted the intensities of the images 1 to 800 and then inverted the first 301
of them, with nothing in the output showing which images had been used.
"""
from datetime import datetime

from ledsa.core.ConfigData import ConfigData
from ledsa.data_extraction.init_functions import generate_image_infos_csv

FIRST_EXPERIMENT_ID = 1
LAST_EXPERIMENT_ID = 800
FIRST_ANALYSIS_ID = 500
LAST_ANALYSIS_ID = 800
TEMPLATE_DEFAULTS = dict(img_directory='imgs/', img_name_string='img{:04d}.jpg',
                         date='2026:10:27', start_time='12:00:00', time_diff_to_image_time=0,
                         first_img_experiment_id=FIRST_EXPERIMENT_ID,
                         last_img_experiment_id=LAST_EXPERIMENT_ID,
                         first_img_analysis_id=FIRST_ANALYSIS_ID,
                         last_img_analysis_id=LAST_ANALYSIS_ID,
                         ref_img_id=1, max_num_leds=10, num_arrays=1)


def create_config(tmp_path, monkeypatch, **options) -> ConfigData:
    """Write a config template in an empty working directory and return it loaded from the file.

    :param options: Template arguments overriding the defaults, e.g. num_skip_imgs.
    """
    monkeypatch.chdir(tmp_path)
    ConfigData(load_config_file=False, **dict(TEMPLATE_DEFAULTS, **options))
    (tmp_path / 'analysis').mkdir()
    return ConfigData()


def patch_image_times(monkeypatch) -> None:
    """Replace the reader of the image times, so that no image file is needed."""
    def fake_experiment_and_real_time(build_type, config, tag, img_number):
        return 0.0, datetime(2026, 10, 27, 12, 0, 0)

    monkeypatch.setattr('ledsa.data_extraction.init_functions._calc_experiment_and_real_time',
                        fake_experiment_and_real_time)


def fail_on_input(monkeypatch) -> None:
    """Let a prompt fail the test, so an option already in the config cannot be asked for again."""
    def fail(prompt=''):
        raise AssertionError(f'asked for an option that is set in the config: {prompt}')

    monkeypatch.setattr('builtins.input', fail)


def image_names(path) -> list:
    """Return the image names listed in an image infos file, in the order they are analysed."""
    lines = path.read_text(encoding='utf-8').splitlines()
    return [line.split(',')[1] for line in lines[1:]]


def analysed_image_names(tmp_path) -> list:
    return image_names(tmp_path / 'analysis' / 'image_infos_analysis.csv')


def test_the_analysis_starts_at_the_first_analysis_image(tmp_path, monkeypatch):
    config = create_config(tmp_path, monkeypatch)
    patch_image_times(monkeypatch)
    fail_on_input(monkeypatch)

    generate_image_infos_csv(config, build_analysis_infos=True)

    names = analysed_image_names(tmp_path)
    assert names[0] == f'img{FIRST_ANALYSIS_ID:04d}.jpg'
    assert names[-1] == f'img{LAST_ANALYSIS_ID:04d}.jpg'
    assert len(names) == LAST_ANALYSIS_ID - FIRST_ANALYSIS_ID + 1


def test_skipped_images_are_counted_from_the_first_analysis_image(tmp_path, monkeypatch):
    config = create_config(tmp_path, monkeypatch, num_skip_imgs=1)
    patch_image_times(monkeypatch)
    fail_on_input(monkeypatch)

    generate_image_infos_csv(config, build_analysis_infos=True)

    names = analysed_image_names(tmp_path)
    assert names[:3] == [f'img{FIRST_ANALYSIS_ID + offset:04d}.jpg' for offset in (0, 2, 4)]


def test_the_experiment_infos_still_cover_the_whole_experiment(tmp_path, monkeypatch):
    config = create_config(tmp_path, monkeypatch)
    patch_image_times(monkeypatch)
    fail_on_input(monkeypatch)

    generate_image_infos_csv(config, build_experiment_infos=True)

    names = image_names(tmp_path / 'image_infos.csv')
    assert names[0] == f'img{FIRST_EXPERIMENT_ID:04d}.jpg'
    assert names[-1] == f'img{LAST_EXPERIMENT_ID:04d}.jpg'
