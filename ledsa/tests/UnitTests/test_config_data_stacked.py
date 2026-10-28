"""Unit tests for the reference image settings of the stacked analysis configuration."""
from ledsa.analysis.ConfigDataStacked import ConfigDataStacked

SIMULATIONS = """
[simulation_0]
path = /data/cam0
camera_position = 1.0 0.0 1.5

[simulation_1]
path = /data/cam1
camera_position = -1.0 0.0 1.5
{sim_1_options}
[led_arrays]
0 = 0:0 1:0
"""


def load_config(tmp_path, default_options='', sim_1_options=''):
    """Write a stacked config with the given options and load it.

    :param default_options: Lines of the ``[DEFAULT]`` section.
    :param sim_1_options: Lines of the ``[simulation_1]`` section, i.e. of the second camera.
    """
    text = ('[DEFAULT]\n' + default_options
            + SIMULATIONS.format(sim_1_options=sim_1_options))
    path = tmp_path / ConfigDataStacked.DEFAULT_FILENAME
    path.write_text(text, encoding='utf-8')
    return ConfigDataStacked(filename=str(path))


class TestRefImgIndices:
    def test_section_value_overrides_default(self, tmp_path):
        config = load_config(tmp_path, 'ref_img_indices = 1 2 3\n',
                             'ref_img_indices = 385 386\n')
        assert config.get_ref_img_indices(0) == [1, 2, 3]
        assert config.get_ref_img_indices(1) == [385, 386]

    def test_missing_value_is_inherited_from_default(self, tmp_path):
        config = load_config(tmp_path, 'ref_img_indices = 1 2 3\n')
        assert config.get_ref_img_indices(0) == [1, 2, 3]
        assert config.get_ref_img_indices(1) == [1, 2, 3]

    def test_none_in_section_overrides_the_list_of_default(self, tmp_path):
        config = load_config(tmp_path, 'ref_img_indices = 1 2 3\n', 'ref_img_indices = None\n')
        assert config.get_ref_img_indices(0) == [1, 2, 3]
        assert config.get_ref_img_indices(1) is None

    def test_none_in_default_leaves_every_camera_without_indices(self, tmp_path):
        config = load_config(tmp_path, 'ref_img_indices = None\n')
        assert config.get_ref_img_indices(0) is None
        assert config.get_ref_img_indices(1) is None

    def test_unset_option_leaves_every_camera_without_indices(self, tmp_path):
        config = load_config(tmp_path)
        assert config.get_ref_img_indices(0) is None
        assert config.get_ref_img_indices(1) is None


class TestNumRefImages:
    def test_section_value_overrides_default(self, tmp_path):
        config = load_config(tmp_path, 'num_ref_images = 10\n', 'num_ref_images = 3\n')
        assert config.get_num_ref_images(0) == 10
        assert config.get_num_ref_images(1) == 3

    def test_missing_value_is_inherited_from_default(self, tmp_path):
        config = load_config(tmp_path, 'num_ref_images = 7\n')
        assert config.get_num_ref_images(0) == 7
        assert config.get_num_ref_images(1) == 7

    def test_unset_option_falls_back_to_ten(self, tmp_path):
        config = load_config(tmp_path)
        assert config.get_num_ref_images(0) == 10


class TestCompatibility:
    def test_default_properties_still_read_the_default_section(self, tmp_path):
        config = load_config(tmp_path, 'num_ref_images = 10\nref_img_indices = 1 2 3\n',
                             'num_ref_images = 3\nref_img_indices = 385 386\n')
        assert config.num_ref_images == 10
        assert config.ref_img_indices == [1, 2, 3]

    def test_template_leaves_the_commented_example_out_of_the_values(self, tmp_path):
        path = tmp_path / ConfigDataStacked.DEFAULT_FILENAME
        ConfigDataStacked.create_template(filename=str(path), num_simulations=2)

        config = ConfigDataStacked(filename=str(path))
        for sim_idx in range(2):
            assert config.get_ref_img_indices(sim_idx) is None
            assert config.get_num_ref_images(sim_idx) == 10


class TestNSummarize:
    """Number of consecutive images summarized into one when average_images is set."""

    def test_the_default_section_value_is_read(self, tmp_path):
        config = load_config(tmp_path, 'average_images = True\nn_summarize = 5\n')
        assert config.n_summarize == 5

    def test_a_config_written_before_the_option_keeps_the_former_behaviour(self, tmp_path):
        # the averaging was hardcoded to summarize two images, so a config of an earlier version is
        # read with that count
        config = load_config(tmp_path)
        assert config.n_summarize == 2
