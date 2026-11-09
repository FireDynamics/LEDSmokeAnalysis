"""A restart of step 3 analyses an image again unless every channel of the restart has its result file."""
from pathlib import Path

import pytest

from ledsa.data_extraction.DataExtractor import DataExtractor


def write_image_infos(num_imgs):
    analysis_dir = Path('analysis')
    analysis_dir.mkdir(exist_ok=True)
    lines = ['#ID,Name,Time[s],Experiment_Time[s]']
    lines += [f'{img_id},IMG_{img_id:04d}.CR2,12:00:{img_id:02d}.000,{float(img_id)}' for img_id in
              range(1, num_imgs + 1)]
    (analysis_dir / 'image_infos_analysis.csv').write_text('\n'.join(lines) + '\n', encoding='utf-8')


def write_result_files(channel, img_ids):
    """Create the result files of the images, the restart looks at their names only."""
    channel_dir = Path('analysis', f'channel{channel}')
    channel_dir.mkdir(parents=True, exist_ok=True)
    for img_id in img_ids:
        (channel_dir / f'{img_id}_led_positions.csv').touch()


def restart(channels):
    """Set up the restart of the channels and return the images it analyses."""
    # the configuration that the constructor reads is not needed to find the images
    extractor = DataExtractor.__new__(DataExtractor)
    extractor.channels = channels
    extractor.setup_restart()
    return Path('images_to_process.csv').read_text(encoding='utf-8').split()


@pytest.mark.parametrize('analysed_img_ids, remaining_img_ids', [
    # interrupted in image 3 after its red channel was written
    ({0: [1, 2, 3], 1: [1, 2], 2: [1, 2]}, [3, 4, 5]),
    # several cores finish the images out of order and each image one channel after the other
    ({0: [1, 2, 4], 1: [1, 4], 2: [1]}, [2, 3, 4, 5]),
])
def test_restart_analyses_an_image_again_that_misses_one_of_the_channels(tmp_path, monkeypatch, analysed_img_ids,
                                                                          remaining_img_ids):
    monkeypatch.chdir(tmp_path)
    write_image_infos(5)
    for channel, img_ids in analysed_img_ids.items():
        write_result_files(channel, img_ids)

    assert restart([0, 1, 2]) == [f'IMG_{img_id:04d}.CR2' for img_id in remaining_img_ids]


def test_restart_of_one_channel_ignores_the_others(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    write_image_infos(5)
    write_result_files(0, [1, 2, 3])
    write_result_files(1, [1])

    assert restart([0]) == ['IMG_0004.CR2', 'IMG_0005.CR2']
