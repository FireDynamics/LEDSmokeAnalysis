"""Channel flags select the requested step 3 mode without a second analysis, and a restart resumes the mode of the
interrupted run."""
import argparse
from pathlib import Path
from unittest.mock import Mock

import pytest

from ledsa.core.parser_arguments_declaration import add_parser_arguments_data_extraction
from ledsa.core import parser_arguments_run
from ledsa.data_extraction.step_3_functions import _create_header


@pytest.mark.parametrize('flags, channels', [
    (['-r'], [0]),
    (['-g'], [1]),
    (['-b'], [2]),
    (['-rgb'], [0, 1, 2]),
    (['-r', '-b'], [0, 2]),
])
@pytest.mark.parametrize('fast', [False, True])
def test_channel_flags_run_step_three_once(tmp_path, monkeypatch, flags, channels, fast):
    monkeypatch.chdir(tmp_path)
    (tmp_path / 'config.ini').touch()
    extractor = Mock()
    monkeypatch.setattr(parser_arguments_run, 'DataExtractor', extractor)
    parser = add_parser_arguments_data_extraction(argparse.ArgumentParser())
    args = parser.parse_args(flags + (['-s3_fast'] if fast else []))

    parser_arguments_run.run_data_extraction_arguments(args)

    kwargs = {'build_experiment_infos': True, 'channels': channels}
    if fast:
        kwargs['fit_leds'] = False
    extractor.assert_called_once_with(**kwargs)
    extractor.return_value.setup_step3.assert_called_once_with()
    extractor.return_value.process_image_data.assert_called_once_with()


@pytest.mark.parametrize('flags, channels', [
    ([], [0]),
    (['-r'], [0]),
    (['-g'], [1]),
    (['-b'], [2]),
    (['-rgb'], [0, 1, 2]),
    (['-g', '-b'], [1, 2]),
])
def test_restart_resumes_the_selected_channels_without_step_three(tmp_path, monkeypatch, flags, channels):
    monkeypatch.chdir(tmp_path)
    (tmp_path / 'config.ini').touch()
    extractor = Mock()
    monkeypatch.setattr(parser_arguments_run, 'DataExtractor', extractor)
    parser = add_parser_arguments_data_extraction(argparse.ArgumentParser())
    args = parser.parse_args(['-re'] + flags)

    parser_arguments_run.run_data_extraction_arguments(args)

    extractor.assert_called_once_with(build_experiment_infos=False, channels=channels, fit_leds=False)
    extractor.return_value.setup_restart.assert_called_once_with()
    extractor.return_value.setup_step3.assert_not_called()
    extractor.return_value.process_image_data.assert_called_once_with()


# Header lines of the result files as LEDSA wrote them before the background-corrected sums were added, with fewer
# columns in both modes than the current version
HEADER_WITH_FEWER_COLUMNS = ('# image root = experiment, photo file name = IMG_0001.CR2, channel = 0, time[s] = 0.0\n'
                             '# id,line,sum_col_value,average_col_value,max_col_value')
FIT_COLUMNS_OF_HEADER = (',led_center_x, led_center_y,x,y,dx,dy,A,alpha,wx,wy,fit_success,fit_fun,fit_nfev,fit_time'
                         '// all spatial quantities in pixel coordinates\n')


@pytest.fixture
def extractor(tmp_path, monkeypatch):
    """Mock of the DataExtractor, run in an experiment directory with a config.ini."""
    monkeypatch.chdir(tmp_path)
    (tmp_path / 'config.ini').touch()
    extractor = Mock()
    monkeypatch.setattr(parser_arguments_run, 'DataExtractor', extractor)
    return extractor


def run_data_extraction(flags):
    parser = add_parser_arguments_data_extraction(argparse.ArgumentParser())
    parser_arguments_run.run_data_extraction_arguments(parser.parse_args(flags))


def step_3_header(channel, img_id, fit_leds):
    """Header that step 3 writes into the result file of an image."""
    img_infos = [['', '', '', '0.0']] * img_id
    return _create_header(channel, img_id, f'IMG_{img_id:04d}.CR2', img_infos, 'experiment', fit_leds)


def write_result_file(channel, img_id, header):
    """Write the result file of step 3 of an image with a single LED, all of its values 0."""
    channel_dir = Path('analysis', f'channel{channel}')
    channel_dir.mkdir(parents=True, exist_ok=True)
    num_columns = header.splitlines()[-1].count(',') + 1
    row = ','.join(['0'] * num_columns)
    (channel_dir / f'{img_id}_led_positions.csv').write_text(f'{header}{row}\n', encoding='utf-8')


@pytest.mark.parametrize('flags, channels', [
    ([], [0]),
    (['-g'], [1]),
    (['-rgb'], [0, 1, 2]),
])
@pytest.mark.parametrize('fit_leds', [True, False])
def test_restart_resumes_in_the_mode_of_the_interrupted_run(extractor, flags, channels, fit_leds):
    for channel in channels:
        for img_id in (1, 2):
            write_result_file(channel, img_id, step_3_header(channel, img_id, fit_leds))

    run_data_extraction(['-re'] + flags)

    extractor.assert_called_once_with(build_experiment_infos=False, channels=channels, fit_leds=fit_leds)
    extractor.return_value.setup_restart.assert_called_once_with()
    extractor.return_value.process_image_data.assert_called_once_with()


@pytest.mark.parametrize('fit_leds', [True, False])
def test_restart_reads_the_mode_from_the_column_names_not_their_number(extractor, fit_leds):
    write_result_file(0, 1, HEADER_WITH_FEWER_COLUMNS + (FIT_COLUMNS_OF_HEADER if fit_leds else '\n'))

    run_data_extraction(['-re'])

    extractor.assert_called_once_with(build_experiment_infos=False, channels=[0], fit_leds=fit_leds)


def test_restart_takes_the_mode_from_the_channels_with_result_files(extractor):
    # interrupted in the first image, after its red channel was written
    write_result_file(0, 1, step_3_header(0, 1, fit_leds=True))

    run_data_extraction(['-re', '-rgb'])

    extractor.assert_called_once_with(build_experiment_infos=False, channels=[0, 1, 2], fit_leds=True)


def test_restart_ignores_an_empty_result_file(extractor):
    # killed after the file of image 2 was created, before its buffered contents reached it
    write_result_file(0, 1, step_3_header(0, 1, fit_leds=True))
    Path('analysis', 'channel0', '2_led_positions.csv').touch()

    run_data_extraction(['-re'])

    extractor.assert_called_once_with(build_experiment_infos=False, channels=[0], fit_leds=True)


def test_restart_stops_if_the_result_files_of_a_channel_disagree(extractor):
    write_result_file(0, 1, step_3_header(0, 1, fit_leds=True))
    write_result_file(0, 2, step_3_header(0, 2, fit_leds=False))

    with pytest.raises(ValueError, match='channel 0 were written both with the LED fits'):
        run_data_extraction(['-re'])

    extractor.assert_not_called()


def test_restart_stops_if_the_channels_disagree(extractor):
    write_result_file(0, 1, step_3_header(0, 1, fit_leds=True))
    write_result_file(1, 1, step_3_header(1, 1, fit_leds=False))

    with pytest.raises(ValueError, match='resumes all of its channels in one mode'):
        run_data_extraction(['-re', '-r', '-g'])

    extractor.assert_not_called()
