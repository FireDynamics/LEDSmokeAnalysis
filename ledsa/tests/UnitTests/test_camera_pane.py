"""Unit tests for a camera that looks through a window pane into the smoke.

The part of a line of sight between the camera and the pane is free of smoke, so only its part
behind the pane is distributed over the layers. Without a pane, the light paths reach the camera
as before.
"""
import configparser

import numpy as np
import pytest

import ledsa.analysis.StackedExtinctionCoefficients as stacked_module
from ledsa.analysis.ConfigDataAnalysis import ConfigDataAnalysis
from ledsa.analysis.ConfigDataStacked import ConfigDataStacked
from ledsa.analysis.Experiment import (Camera, Experiment, InconsistentDistanceError, LED, Layers, Pane,
                                       PaneGeometryError, read_pane)
from ledsa.analysis.ExtinctionCoefficientsNonLinear import ExtinctionCoefficientsNonLinear
from ledsa.analysis.StackedExtinctionCoefficients import StackedExtinctionCoefficients

NUM_LAYERS = 30
BOTTOM_BORDER = 0.0
TOP_BORDER = 3.0
CAMERA_POSITION = (0.0, 0.0, 2.0)
LED_DISTANCE = 4.0  # the LEDs stand at y = 4, in front of the camera at y = 0
HEIGHTS = [0.5, 1.5, 2.5]
# the pane at y = 2 lies halfway between the camera and the LEDs
HALFWAY_PANE = Pane((0.0, 2.0, 0.0), (0.0, 1.0, 0.0))


class ExperimentStub(Experiment):
    """Experiment whose LEDs are given by their heights instead of read from the analysis files."""

    def __init__(self, heights, pane=None, camera_position=CAMERA_POSITION):
        self.heights = heights
        super().__init__(layers=Layers(NUM_LAYERS, BOTTOM_BORDER, TOP_BORDER), led_array=0,
                         camera=Camera(*camera_position, pane=pane))

    def set_leds(self) -> None:
        self.leds = [LED(led_id, 0.0, LED_DISTANCE, height) for led_id, height in enumerate(self.heights)]
        self.num_leds = len(self.leds)


def distances(heights, pane=None, camera_position=CAMERA_POSITION):
    """Distances traversed in the layers, one row per LED in the order of the heights."""
    experiment = ExperimentStub(heights, pane, camera_position)
    return np.array([experiment.calc_traversed_dist_per_layer(led) for led in experiment.leds])


def distance_to_camera(height, camera_position=CAMERA_POSITION):
    return np.hypot(LED_DISTANCE - camera_position[1], height - camera_position[2])


class TestLightPathsBehindAPane:
    def test_without_a_pane_the_light_paths_reach_the_camera(self):
        np.testing.assert_allclose(distances(HEIGHTS).sum(axis=1),
                                   [distance_to_camera(height) for height in HEIGHTS])

    def test_the_light_path_starts_where_the_line_of_sight_crosses_the_pane(self):
        experiment = ExperimentStub([1.0], HALFWAY_PANE)

        start = experiment.calc_smoke_path_start(experiment.leds[0])

        np.testing.assert_allclose(start, (0.0, 2.0, 1.5))

    def test_a_pane_halfway_halves_the_light_paths(self):
        np.testing.assert_allclose(distances(HEIGHTS, HALFWAY_PANE).sum(axis=1),
                                   [distance_to_camera(height) / 2 for height in HEIGHTS])

    def test_only_the_layers_between_the_pane_and_the_led_hold_the_light_path(self):
        # the line of sight of the LED at 0.5 m crosses the pane at 1.25 m
        full, = distances([0.5])
        behind, = distances([0.5], HALFWAY_PANE)
        borders = np.linspace(BOTTOM_BORDER, TOP_BORDER, NUM_LAYERS + 1)
        below_the_crossing = borders[1:] <= 1.25
        above_the_crossing = borders[:-1] >= 1.25

        np.testing.assert_allclose(behind[below_the_crossing], full[below_the_crossing])
        assert not behind[above_the_crossing].any()

    def test_the_direction_and_length_of_the_normal_do_not_matter(self):
        flipped = Pane((0.0, 2.0, 0.0), (0.0, -3.0, 0.0))

        np.testing.assert_allclose(distances(HEIGHTS, flipped), distances(HEIGHTS, HALFWAY_PANE))

    def test_an_oblique_pane(self):
        # the plane y + z / 2 = 2 crosses the line of sight to the LED at 1 m at 2/7 of its length
        oblique = Pane((0.0, 2.0, 0.0), (0.0, 1.0, 0.5))

        np.testing.assert_allclose(distances([1.0], oblique).sum(), 5 / 7 * distance_to_camera(1.0))

    def test_a_level_line_of_sight_lies_in_the_layer_of_the_camera(self):
        behind, = distances([CAMERA_POSITION[2]], HALFWAY_PANE)

        assert behind.sum() == pytest.approx(LED_DISTANCE / 2)
        assert np.count_nonzero(behind) == 1

    def test_a_camera_on_the_pane_sees_the_smoke_from_the_camera_on(self):
        pane = Pane(CAMERA_POSITION, (0.0, 1.0, 0.0))

        np.testing.assert_allclose(distances(HEIGHTS, pane), distances(HEIGHTS))

    def test_an_led_on_the_camera_side_of_the_pane_is_reported(self):
        behind_the_leds = Pane((0.0, LED_DISTANCE + 1, 0.0), (0.0, 1.0, 0.0))

        with pytest.raises(PaneGeometryError, match='LED 0 '):
            distances([1.0], behind_the_leds)

    def test_an_led_on_the_pane_is_reported(self):
        through_the_leds = Pane((0.0, LED_DISTANCE, 0.0), (0.0, 1.0, 0.0))

        with pytest.raises(PaneGeometryError, match='pane_normal'):
            distances([1.0], through_the_leds)

    def test_a_camera_above_the_layers_is_accepted_if_the_light_paths_behind_the_pane_lie_inside(self):
        # the lines of sight cross the pane at 2.0 m and 2.5 m, inside the layers
        camera_position = (0.0, 0.0, TOP_BORDER + 0.5)
        heights = [0.5, 1.5]

        np.testing.assert_allclose(distances(heights, HALFWAY_PANE, camera_position).sum(axis=1),
                                   [distance_to_camera(height, camera_position) / 2 for height in heights])
        with pytest.raises(InconsistentDistanceError, match='domain_bounds'):
            distances(heights, None, camera_position)

    def test_a_crossing_above_the_layers_is_reported(self):
        # the line of sight crosses the pane at 4.25 m, above the layers
        with pytest.raises(InconsistentDistanceError, match='cross the pane'):
            distances([2.5], HALFWAY_PANE, (0.0, 0.0, 6.0))

    def test_the_distance_array_holds_the_light_paths_behind_the_pane(self):
        solver = ExtinctionCoefficientsNonLinear(ExperimentStub(HEIGHTS, HALFWAY_PANE))

        distance_array = solver.calc_distance_array()

        # the distance array is built from the top LED down, the reverse of the LED order
        np.testing.assert_allclose(distance_array.sum(axis=1),
                                   [distance_to_camera(height) / 2 for height in reversed(HEIGHTS)])


class TestPane:
    def test_a_zero_normal_is_refused(self):
        with pytest.raises(ValueError, match='zero vector'):
            Pane((0.0, 2.0, 0.0), (0.0, 0.0, 0.0))

    def test_two_coordinates_are_refused(self):
        with pytest.raises(ValueError, match='three finite coordinates'):
            Pane((0.0, 2.0), (0.0, 1.0, 0.0))

    def test_a_coordinate_that_is_not_finite_is_refused(self):
        with pytest.raises(ValueError, match='three finite coordinates'):
            Pane((0.0, np.nan, 0.0), (0.0, 1.0, 0.0))

    def test_the_camera_of_the_result_header_names_its_pane(self):
        assert str(Camera(0, 0, 2)) == 'Camera: (0, 0, 2)\n'
        assert str(Camera(0, 0, 2, pane=HALFWAY_PANE)) == \
            'Camera: (0, 0, 2), behind the pane through (0, 2, 0) with the normal (0, 1, 0)\n'


def config_with(**options):
    config = configparser.ConfigParser()
    config['experiment_geometry'] = {'camera_position': '0 0 2', **options}
    return config


class TestReadPane:
    def test_a_config_without_the_options_has_no_pane(self):
        assert read_pane(config_with(), 'experiment_geometry', 'config_analysis.ini') is None

    def test_none_means_no_pane(self):
        config = config_with(pane_point='None', pane_normal='None')

        assert read_pane(config, 'experiment_geometry', 'config_analysis.ini') is None

    def test_a_section_that_is_missing_has_no_pane(self):
        assert read_pane(config_with(), 'simulation_3', 'config_stacked.ini') is None

    def test_both_options_give_the_pane(self):
        config = config_with(pane_point='0 2 0', pane_normal='0 1 0')

        assert read_pane(config, 'experiment_geometry', 'config_analysis.ini') == HALFWAY_PANE

    def test_a_point_without_a_normal_is_reported(self):
        config = config_with(pane_point='0 2 0')

        with pytest.raises(ValueError, match='sets pane_point but not pane_normal'):
            read_pane(config, 'experiment_geometry', 'config_analysis.ini')

    def test_a_normal_without_a_point_is_reported(self):
        config = config_with(pane_normal='0 1 0', pane_point='None')

        with pytest.raises(ValueError, match='sets pane_normal but not pane_point'):
            read_pane(config, 'experiment_geometry', 'config_analysis.ini')

    @pytest.mark.parametrize('value', ['0 2', '0 2 0 1', 'x 2 0'])
    def test_a_value_that_is_not_three_numbers_is_reported(self, value):
        config = config_with(pane_point=value, pane_normal='0 1 0')

        with pytest.raises(ValueError, match='pane_point = .* is not three numbers'):
            read_pane(config, 'experiment_geometry', 'config_analysis.ini')

    def test_a_zero_normal_is_reported_with_its_section(self):
        config = config_with(pane_point='0 2 0', pane_normal='0 0 0')

        with pytest.raises(ValueError, match=r'\[experiment_geometry\] of config_analysis.ini: .*zero'):
            read_pane(config, 'experiment_geometry', 'config_analysis.ini')


class TestAnalysisConfig:
    def test_the_template_writes_no_pane(self, tmp_path, monkeypatch):
        monkeypatch.chdir(tmp_path)
        ConfigDataAnalysis(load_config_file=False)

        config = ConfigDataAnalysis()

        assert config['experiment_geometry']['pane_point'] == 'None'
        assert config['experiment_geometry']['pane_normal'] == 'None'
        assert config.get_pane() is None

    def test_the_template_writes_a_given_pane(self, tmp_path, monkeypatch):
        monkeypatch.chdir(tmp_path)
        ConfigDataAnalysis(load_config_file=False, pane_point=[0, 2, 0], pane_normal='0 1 0')

        assert ConfigDataAnalysis().get_pane() == HALFWAY_PANE

    def test_a_config_written_before_the_options_has_no_pane(self, tmp_path, monkeypatch):
        monkeypatch.chdir(tmp_path)
        (tmp_path / 'config_analysis.ini').write_text(
            '[experiment_geometry]\ncamera_position = 0 0 2\n', encoding='utf-8')

        assert ConfigDataAnalysis().get_pane() is None


STACKED_CONFIG = """
[DEFAULT]
solver = linear
num_layers = 10
domain_bounds = 0 3
reference_property = sum_col_val

[simulation_0]
path = {tmp_path}/cam0
camera_position = 0 0 2
pane_point = 0 2 0
pane_normal = 0 1 0

[simulation_1]
path = {tmp_path}/cam1
camera_position = 0 0 0.5

[led_arrays]
0 = 0:0 1:0
"""


def load_stacked_config(tmp_path):
    path = tmp_path / ConfigDataStacked.DEFAULT_FILENAME
    path.write_text(STACKED_CONFIG.format(tmp_path=tmp_path.as_posix()), encoding='utf-8')
    return ConfigDataStacked(filename=str(path))


class ExperimentSpy:
    """Stands in for the Experiment of a camera, which would read the LEDs of its run."""

    def __init__(self, layers, led_array, camera, path, channel, merge_led_arrays):
        self.camera = camera


class TestStackedAnalysis:
    def test_every_camera_has_its_own_pane(self, tmp_path):
        config = load_stacked_config(tmp_path)

        assert config.get_pane(0) == HALFWAY_PANE
        assert config.get_pane(1) is None

    def test_the_template_leaves_the_pane_commented_out(self, tmp_path):
        path = tmp_path / ConfigDataStacked.DEFAULT_FILENAME
        ConfigDataStacked.create_template(filename=str(path), num_simulations=2)

        config = ConfigDataStacked(filename=str(path))

        assert [config.get_pane(sim_idx) for sim_idx in range(2)] == [None, None]

    @pytest.fixture
    def stacked_solver(self, tmp_path, monkeypatch):
        monkeypatch.setattr(stacked_module, 'Experiment', ExperimentSpy)
        monkeypatch.setattr(StackedExtinctionCoefficients, '_build_sub_solver', lambda self, exp, sim_idx: None)
        return StackedExtinctionCoefficients(load_stacked_config(tmp_path), led_array_id=0, channel=0)

    def test_the_analysis_puts_every_camera_behind_its_own_pane(self, stacked_solver):
        assert [experiment.camera.pane for experiment in stacked_solver.sub_experiments] == [HALFWAY_PANE, None]

    def test_the_header_names_only_the_cameras_behind_a_pane(self, stacked_solver):
        assert stacked_solver._pane_info() == 'sim0 behind the pane through (0, 2, 0) with the normal (0, 1, 0) | '
