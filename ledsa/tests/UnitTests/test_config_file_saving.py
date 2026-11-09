"""Unit tests for saving a configuration without dropping the help texts of its options.

LEDSA writes the config back whenever it fills in a value it asked for or read from an image,
so a run must leave the comments of the file, which are the only documentation of the options,
where the user wrote them.
"""
from ledsa.analysis.ConfigDataAnalysis import ConfigDataAnalysis
from ledsa.analysis.ConfigDataStacked import ConfigDataStacked
from ledsa.core.ConfigData import ConfigData

HAND_WRITTEN = """\
[DEFAULT]
# Where the images of the experiment are stored
img_directory = imgs/
img_name_string = img{:04d}.jpg

[analyse_photo]
# Only images with an ID dividable by this + 1 are analysed
num_skip_imgs = 0
; a comment in the other notation
num_skip_leds = 0
"""


def comment_lines(text: str) -> list:
    """Return the comment lines of a config file, i.e. the help texts of its options."""
    return [line for line in text.splitlines() if line.strip().startswith(('#', ';'))]


def create_template(tmp_path, monkeypatch, **options) -> str:
    """Write the config template in an empty working directory and return its text.

    :param options: Template arguments, e.g. img_directory.
    """
    monkeypatch.chdir(tmp_path)
    ConfigData(load_config_file=False, **options)
    return (tmp_path / 'config.ini').read_text(encoding='utf-8')


def write_config(tmp_path, monkeypatch, text=HAND_WRITTEN) -> ConfigData:
    """Write a config by hand in an empty working directory and load it."""
    monkeypatch.chdir(tmp_path)
    (tmp_path / 'config.ini').write_text(text, encoding='utf-8')
    return ConfigData()


class TestHelpTexts:
    def test_template_help_texts_survive_saving(self, tmp_path, monkeypatch):
        created = create_template(tmp_path, monkeypatch)
        ConfigData().save()
        saved = (tmp_path / 'config.ini').read_text(encoding='utf-8')
        assert comment_lines(saved) == comment_lines(created)
        assert comment_lines(saved) != []

    def test_hand_written_config_only_changes_in_the_value(self, tmp_path, monkeypatch):
        config = write_config(tmp_path, monkeypatch)
        config['analyse_photo']['num_skip_imgs'] = '3'
        config.save()
        saved = (tmp_path / 'config.ini').read_text(encoding='utf-8')
        assert saved == HAND_WRITTEN.replace('num_skip_imgs = 0', 'num_skip_imgs = 3')

    def test_saving_twice_changes_nothing(self, tmp_path, monkeypatch):
        create_template(tmp_path, monkeypatch)
        ConfigData().save()
        once = (tmp_path / 'config.ini').read_text(encoding='utf-8')
        ConfigData().save()
        assert (tmp_path / 'config.ini').read_text(encoding='utf-8') == once

    def test_analysis_template_help_texts_survive_saving(self, tmp_path, monkeypatch):
        monkeypatch.chdir(tmp_path)
        ConfigDataAnalysis(load_config_file=False)
        created = (tmp_path / 'config_analysis.ini').read_text(encoding='utf-8')
        config = ConfigDataAnalysis()
        config['model_parameters']['domain_bounds'] = '0 3'
        config.save()
        saved = (tmp_path / 'config_analysis.ini').read_text(encoding='utf-8')
        assert comment_lines(saved) == comment_lines(created)
        assert 'domain_bounds = 0 3' in saved

    def test_stacked_template_help_texts_survive_saving(self, tmp_path):
        path = tmp_path / ConfigDataStacked.DEFAULT_FILENAME
        ConfigDataStacked.create_template(filename=str(path))
        created = path.read_text(encoding='utf-8')
        config = ConfigDataStacked(filename=str(path))
        config['simulation_0']['path'] = '/data/cam0'
        config.save()
        saved = path.read_text(encoding='utf-8')
        assert comment_lines(saved) == comment_lines(created)
        assert ConfigDataStacked(filename=str(path)).get_simulation_path(0).as_posix() == '/data/cam0'


class TestValues:
    def test_changed_value_is_read_back(self, tmp_path, monkeypatch):
        create_template(tmp_path, monkeypatch)
        config = ConfigData()
        config['DEFAULT']['start_time'] = '15:36:07'
        config['find_search_areas']['ref_img_id'] = '1'
        config.save()
        reloaded = ConfigData()
        assert reloaded['DEFAULT']['start_time'] == '15:36:07'
        assert reloaded['find_search_areas']['ref_img_id'] == '1'

    def test_every_option_of_the_template_is_kept(self, tmp_path, monkeypatch):
        create_template(tmp_path, monkeypatch, img_directory='imgs/')
        before = ConfigData()
        before.save()
        after = ConfigData()
        for section in [before.default_section] + before.sections():
            assert dict(after[section]) == dict(before[section])

    def test_multiline_value_survives_repeated_saving(self, tmp_path, monkeypatch):
        create_template(tmp_path, monkeypatch)
        config = ConfigData()
        config['analyse_positions']['led_array_edge_indices'] = '\n457 246\n12 347'
        config.save()
        for _ in range(2):
            config = ConfigData()
            config.save()
        assert ConfigData().get2dnparray('analyse_positions',
                                         'led_array_edge_indices').tolist() == [[457, 246], [12, 347]]

    def test_option_added_to_the_parser_is_appended_to_its_section(self, tmp_path, monkeypatch):
        config = write_config(tmp_path, monkeypatch)
        config['analyse_photo']['background_floor'] = '0.05'
        config.save()
        saved = (tmp_path / 'config.ini').read_text(encoding='utf-8')
        assert saved.splitlines()[-1] == 'background_floor = 0.05'
        assert comment_lines(saved) == comment_lines(HAND_WRITTEN)

    def test_option_overriding_a_default_is_written_into_its_section(self, tmp_path, monkeypatch):
        config = write_config(tmp_path, monkeypatch)
        config['analyse_photo']['img_directory'] = 'other_imgs/'
        config.save()
        reloaded = ConfigData()
        assert reloaded['analyse_photo']['img_directory'] == 'other_imgs/'
        assert reloaded['DEFAULT']['img_directory'] == 'imgs/'

    def test_section_added_to_the_parser_is_appended(self, tmp_path, monkeypatch):
        """A section the file does not hold yet is written with the help texts it carries.

        A help text is stored as an option without a value, which ConfigParser lowercases, so
        the comparison ignores the case.
        """
        config = write_config(tmp_path, monkeypatch)
        config['analyse_positions'] = {}
        config.set('analyse_positions', '# Number of LED arrays')
        config['analyse_positions']['num_arrays'] = '7'
        config.save()
        saved = (tmp_path / 'config.ini').read_text(encoding='utf-8')
        assert '# number of led arrays' in saved.lower()
        assert saved.startswith(HAND_WRITTEN)
        assert ConfigData()['analyse_positions']['num_arrays'] == '7'

    def test_removed_option_is_dropped(self, tmp_path, monkeypatch):
        config = write_config(tmp_path, monkeypatch)
        config.remove_option('analyse_photo', 'num_skip_leds')
        config.save()
        saved = (tmp_path / 'config.ini').read_text(encoding='utf-8')
        assert 'num_skip_leds' not in saved
        assert comment_lines(saved) == comment_lines(HAND_WRITTEN)

    def test_config_file_is_created_when_it_is_missing(self, tmp_path, monkeypatch):
        monkeypatch.chdir(tmp_path)
        config = ConfigData(load_config_file=False)
        (tmp_path / 'config.ini').unlink()
        config.save()
        assert comment_lines((tmp_path / 'config.ini').read_text(encoding='utf-8')) != []

    def test_values_set_on_a_fresh_template_are_written(self, tmp_path, monkeypatch):
        """The demo and the acceptance tests fill a template in memory before saving it.

        Their option names carry the indentation the template gives them, which is part of the
        name the parser holds but not of the one the file names.
        """
        monkeypatch.chdir(tmp_path)
        config = ConfigData(load_config_file=False)
        config.set('analyse_positions', '   led_array_edge_indices', '49 0')
        config.set('DEFAULT', '   date', '2018:11:27')
        config.save()
        saved = (tmp_path / 'config.ini').read_text(encoding='utf-8')
        assert saved.count('led_array_edge_indices =') == 1
        assert saved.count('date =') == 1
        reloaded = ConfigData()
        assert reloaded['analyse_positions']['led_array_edge_indices'] == '49 0'
        assert reloaded['DEFAULT']['date'] == '2018:11:27'
