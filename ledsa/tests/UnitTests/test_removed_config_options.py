"""Unit tests for the note on config options that LEDSA no longer has.

A config file written by an older version keeps setting the options a release has removed. The
parser reads them without complaint and LEDSA ignores them, so without the note an analysis would
change without the user noticing.
"""
import configparser
import importlib

import pytest

from ledsa.analysis.ConfigDataAnalysis import ConfigDataAnalysis
from ledsa.core import config_file
from ledsa.core.ConfigData import ConfigData

# the modules, not the classes of the same name
config_data_module = importlib.import_module('ledsa.core.ConfigData')
config_data_analysis_module = importlib.import_module('ledsa.analysis.ConfigDataAnalysis')


@pytest.fixture(autouse=True)
def fresh_process(monkeypatch):
    """Every test starts like a new process, with no option named yet."""
    monkeypatch.setattr(config_file, '_reported_removed_options', set())


def parser_of(text: str) -> configparser.ConfigParser:
    parser = configparser.ConfigParser(allow_no_value=True)
    parser.read_string(text)
    return parser


class TestReportRemovedOptions:
    def test_an_option_the_file_still_sets_is_named_with_what_became_of_it(self, capsys):
        parser = parser_of('[analyse_photo]\n   old_option = 3\n   num_skip_leds = 0\n')
        found = config_file.report_removed_options(parser, 'config.ini', {'old_option': 'It is gone.'})
        assert found == ['old_option']
        assert 'config.ini sets old_option, which is ignored. It is gone.' in capsys.readouterr().out

    def test_an_option_of_the_default_section_is_found(self):
        parser = parser_of('[DEFAULT]\nold_option = 3\n')
        assert config_file.report_removed_options(parser, 'x.ini', {'old_option': 'Gone.'}) == ['old_option']

    def test_a_file_without_removed_options_gives_no_note(self, capsys):
        parser = parser_of('[analyse_photo]\nnum_skip_leds = 0\n')
        assert config_file.report_removed_options(parser, 'config.ini', {'old_option': 'Gone.'}) == []
        assert capsys.readouterr().out == ''

    def test_the_note_is_given_once_per_process(self, capsys):
        parser = parser_of('[DEFAULT]\nold_option = 3\n')
        config_file.report_removed_options(parser, 'x.ini', {'old_option': 'Gone.'})
        config_file.report_removed_options(parser, 'x.ini', {'old_option': 'Gone.'})
        assert capsys.readouterr().out.count('old_option') == 1


class TestLoadingAConfigFile:
    def test_config_ini_names_its_removed_options(self, tmp_path, monkeypatch, capsys):
        monkeypatch.chdir(tmp_path)
        monkeypatch.setitem(config_data_module.REMOVED_OPTIONS, 'old_option', 'Gone.')
        (tmp_path / 'config.ini').write_text('[analyse_photo]\n   old_option = 3\n', encoding='utf-8')
        ConfigData()
        assert 'config.ini sets old_option, which is ignored. Gone.' in capsys.readouterr().out

    def test_config_analysis_ini_names_its_removed_options(self, tmp_path, monkeypatch, capsys):
        monkeypatch.chdir(tmp_path)
        monkeypatch.setitem(config_data_analysis_module.REMOVED_OPTIONS, 'old_option', 'Gone.')
        (tmp_path / 'config_analysis.ini').write_text('[DEFAULT]\n   old_option = 3\n', encoding='utf-8')
        ConfigDataAnalysis()
        assert 'config_analysis.ini sets old_option, which is ignored. Gone.' in capsys.readouterr().out

    @pytest.mark.parametrize('module, file_name', [(config_data_module, 'config.ini'),
                                                   (config_data_analysis_module, 'config_analysis.ini')])
    def test_every_removed_option_is_named(self, module, file_name, capsys):
        """The options the releases have removed are all found where the old templates wrote them."""
        if not module.REMOVED_OPTIONS:
            pytest.skip(f'no option of {file_name} has been removed')
        section = 'analyse_photo' if file_name == 'config.ini' else 'DEFAULT'
        parser = parser_of(f'[{section}]\n' + ''.join(f'   {name} = 1\n' for name in module.REMOVED_OPTIONS))
        assert config_file.report_removed_options(parser, file_name, module.REMOVED_OPTIONS) == \
               sorted(module.REMOVED_OPTIONS)
