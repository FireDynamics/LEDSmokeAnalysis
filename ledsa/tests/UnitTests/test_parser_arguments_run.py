"""Channel flags select the requested step 3 mode without a second analysis."""
import argparse
from unittest.mock import Mock

import pytest

from ledsa.core.parser_arguments_declaration import add_parser_arguments_data_extraction
from ledsa.core import parser_arguments_run


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
