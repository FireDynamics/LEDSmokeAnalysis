"""Unit tests for the clock offset in the experiment time of the image infos files.

start_time is on the real clock, the EXIF times of the images are on the camera clock. The
experiment time used to be measured from the uncorrected EXIF time, so every experiment time
was off by exif_time_infront_real_time, and cameras with different offsets stayed apart.
"""
import os
from datetime import datetime, timedelta

import pandas as pd

from ledsa.core.ConfigData import ConfigData
from ledsa.data_extraction.init_functions import generate_image_infos_csv

CAMERA_AHEAD_OF_REAL_TIME = 30.0
FIRST_REAL_TIME = datetime(2026, 10, 27, 12, 0, 0)
IMG_INTERVAL = 10
NUM_IMGS = 3


def create_config(tmp_path, monkeypatch, time_diff, start_time=None) -> ConfigData:
    monkeypatch.chdir(tmp_path)
    ConfigData(load_config_file=False, img_directory='imgs/', img_name_string='img{:04d}.jpg',
               date='2026:10:27', start_time=start_time, time_diff_to_image_time=time_diff,
               first_img_experiment_id=1, last_img_experiment_id=NUM_IMGS,
               first_img_analysis_id=1, last_img_analysis_id=NUM_IMGS,
               ref_img_id=1, max_num_leds=10, num_arrays=1)
    (tmp_path / 'analysis').mkdir()
    return ConfigData()


def patch_exif_times(monkeypatch, time_diff) -> None:
    """Let image n be taken IMG_INTERVAL * (n - 1) seconds after FIRST_REAL_TIME on a camera clock running time_diff ahead."""
    def fake_exif_datetime(path, tag):
        img_id = int(os.path.basename(path)[3:7])
        return FIRST_REAL_TIME + timedelta(seconds=time_diff + IMG_INTERVAL * (img_id - 1))

    monkeypatch.setattr('ledsa.core.ConfigData.get_exif_datetime', fake_exif_datetime)
    monkeypatch.setattr('ledsa.data_extraction.init_functions.get_exif_datetime', fake_exif_datetime)


def experiment_times(tmp_path) -> list:
    return list(pd.read_csv(tmp_path / 'image_infos.csv')['Experiment_Time[s]'])


def test_the_first_image_is_at_zero_when_start_time_is_taken_from_it(tmp_path, monkeypatch):
    config = create_config(tmp_path, monkeypatch, CAMERA_AHEAD_OF_REAL_TIME)
    patch_exif_times(monkeypatch, CAMERA_AHEAD_OF_REAL_TIME)

    generate_image_infos_csv(config, build_experiment_infos=True)

    assert config['DEFAULT']['start_time'] == '12:00:00'
    assert experiment_times(tmp_path) == [0.0, 10.0, 20.0]


def test_cameras_with_different_offsets_share_the_time_axis(tmp_path, monkeypatch):
    times = []
    for time_diff in (CAMERA_AHEAD_OF_REAL_TIME, -5.0):
        camera_path = tmp_path / f'camera_{time_diff}'
        camera_path.mkdir()
        config = create_config(camera_path, monkeypatch, time_diff, start_time='12:00:00')
        patch_exif_times(monkeypatch, time_diff)
        generate_image_infos_csv(config, build_experiment_infos=True)
        times.append(experiment_times(camera_path))

    assert times[0] == times[1] == [0.0, 10.0, 20.0]
