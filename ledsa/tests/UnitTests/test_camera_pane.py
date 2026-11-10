"""Unit tests for a camera that looks through a window pane into the smoke.

The pane is pane_distance away from the camera and taken as normal to the line of sight, so the
first pane_distance of every line of sight is free of smoke and only the part behind the pane is
distributed over the layers. Without a pane, the light paths reach the camera as before.
"""
import configparser

import numpy as np
import pytest

import ledsa.analysis.StackedExtinctionCoefficients as stacked_module
from ledsa.analysis.ConfigDataAnalysis import ConfigDataAnalysis
from ledsa.analysis.ConfigDataStacked import ConfigDataStacked
from ledsa.analysis.Experiment import (Camera, Experiment, InconsistentDistanceError, LED, Layers,
                                       PaneGeometryError, read_pane_distance)
from ledsa.analysis.ExtinctionCoefficientsNonLinear import ExtinctionCoefficientsNonLinear
from ledsa.analysis.StackedExtinctionCoefficients import StackedExtinctionCoefficients

NUM_LAYERS = 30
BOTTOM_BORDER = 0.0
TOP_BORDER = 3.0
CAMERA_POSITION = (0.0, 0.0, 2.0)
LED_DISTANCE = 4.0  # the LEDs stand at y = 4, in front of the camera at y = 0
HEIGHTS = [0.5, 1.5, 2.5]


class ExperimentStub(Experiment):
    """Experiment whose LEDs are given by their heights instead of read from the analysis files."""

    def __init__(self, heights, pane_distance=None, camera_position=CAMERA_POSITION):
        self.heights = heights
        super().__init__(layers=Layers(NUM_LAYERS, BOTTOM_BORDER, TOP_BORDER), led_array=0,
                         camera=Camera(*camera_position, pane_distance=pane_distance))

    def set_leds(self) -> None:
        self.leds = [LED(led_id, 0.0, LED_DISTANCE, height) for led_id, height in enumerate(self.heights)]
        self.num_leds = len(self.leds)


def distances(heights, pane_distance=None, camera_position=CAMERA_POSITION):
    """Distances traversed in the layers, one row per LED in the order of the heights."""
    experiment = ExperimentStub(heights, pane_distance, camera_position)
    return np.array([experiment.calc_traversed_dist_per_layer(led) for led in experiment.leds])


def distance_to_camera(height, camera_position=CAMERA_POSITION):
    return np.hypot(LED_DISTANCE - camera_position[1], height - camera_position[2])


class TestLightPathsBehindAPane:
    def test_without_a_pane_the_light_paths_reach_the_camera(self):
        np.testing.assert_allclose(distances(HEIGHTS).sum(axis=1),
                                   [distance_to_camera(height) for height in HEIGHTS])

    def test_the_light_path_starts_pane_distance_away_from_the_camera(self):
        # the line of sight from the camera at 3.5 m to the LED at 0.5 m is 5 m long
        experiment = ExperimentStub([0.5], 2.5, camera_position=(0.0, 0.0, 3.5))

        start = experiment.calc_smoke_path_start(experiment.leds[0])

        np.testing.assert_allclose(start, (0.0, 2.0, 2.0))

    def test_the_pane_shortens_every_light_path_by_its_distance(self):
        np.testing.assert_allclose(distances(HEIGHTS, 1.0).sum(axis=1),
                                   [distance_to_camera(height) - 1.0 for height in HEIGHTS])

    def test_only_the_layers_between_the_pane_and_the_led_hold_the_light_path(self):
        # a pane halfway along the line of sight of the LED at 0.5 m is crossed at 1.25 m
        halfway = distance_to_camera(0.5) / 2
        full, = distances([0.5])
        behind, = distances([0.5], halfway)
        borders = np.linspace(BOTTOM_BORDER, TOP_BORDER, NUM_LAYERS + 1)
        below_the_crossing = borders[1:] <= 1.25
        above_the_crossing = borders[:-1] >= 1.25

        np.testing.assert_allclose(behind[below_the_crossing], full[below_the_crossing])
        assert not behind[above_the_crossing].any()

    def test_a_level_line_of_sight_lies_in_the_layer_of_the_camera(self):
        behind, = distances([CAMERA_POSITION[2]], 1.5)

        assert behind.sum() == pytest.approx(LED_DISTANCE - 1.5)
        assert np.count_nonzero(behind) == 1

    def test_a_pane_at_the_camera_changes_nothing(self):
        np.testing.assert_array_equal(distances(HEIGHTS, 0.0), distances(HEIGHTS))

    def test_an_led_nearer_than_the_pane_is_reported(self):
        with pytest.raises(PaneGeometryError, match='LED 0 '):
            distances([1.0], LED_DISTANCE + 1)

    def test_an_led_on_the_pane_is_reported(self):
        with pytest.raises(PaneGeometryError, match='pane_distance'):
            distances([CAMERA_POSITION[2]], LED_DISTANCE)

    def test_a_camera_above_the_layers_is_accepted_if_the_light_paths_behind_the_pane_lie_inside(self):
        # the lines of sight cross the pane at 2.0 m and 2.38 m, inside the layers
        camera_position = (0.0, 0.0, TOP_BORDER + 0.5)
        heights = [0.5, 1.5]

        np.testing.assert_allclose(distances(heights, 2.5, camera_position).sum(axis=1),
                                   [distance_to_camera(height, camera_position) - 2.5 for height in heights])
        with pytest.raises(InconsistentDistanceError, match='domain_bounds'):
            distances(heights, None, camera_position)

    def test_a_crossing_above_the_layers_is_reported(self):
        # the 5 m long line of sight from the camera at 5.5 m to the LED at 2.5 m crosses the pane at
        # 4 m, above the layers
        with pytest.raises(InconsistentDistanceError,
                           match=r'point \(0, 2, 4\) where its line of sight crosses the pane'):
            distances([2.5], 2.5, (0.0, 0.0, 5.5))

    def test_the_distance_array_holds_the_light_paths_behind_the_pane(self):
        solver = ExtinctionCoefficientsNonLinear(ExperimentStub(HEIGHTS, 1.0))

        distance_array = solver.calc_distance_array()

        # the distance array is built from the top LED down, the reverse of the LED order
        np.testing.assert_allclose(distance_array.sum(axis=1),
                                   [distance_to_camera(height) - 1.0 for height in reversed(HEIGHTS)])


class TestCamera:
    def test_a_negative_distance_is_refused(self):
        with pytest.raises(ValueError, match='at least 0'):
            Camera(0.0, 0.0, 2.0, pane_distance=-0.5)

    @pytest.mark.parametrize('distance', [np.nan, np.inf])
    def test_a_distance_that_is_not_finite_is_refused(self, distance):
        with pytest.raises(ValueError, match='finite'):
            Camera(0.0, 0.0, 2.0, pane_distance=distance)

    def test_the_camera_of_the_result_header_names_the_pane_distance(self):
        assert str(Camera(0, 0, 2)) == 'Camera: (0, 0, 2)\n'
        assert str(Camera(0, 0, 2, pane_distance=0.5)) == 'Camera: (0, 0, 2), pane_distance: 0.5\n'


def config_with(**options):
    config = configparser.ConfigParser()
    config['experiment_geometry'] = {'camera_position': '0 0 2', **options}
    return config


class TestReadPaneDistance:
    def test_a_config_without_the_option_has_no_pane(self):
        assert read_pane_distance(config_with(), 'experiment_geometry', 'config_analysis.ini') is None

    @pytest.mark.parametrize('value', ['None', 'none', ''])
    def test_none_or_no_value_means_no_pane(self, value):
        config = config_with(pane_distance=value)

        assert read_pane_distance(config, 'experiment_geometry', 'config_analysis.ini') is None

    def test_a_section_that_is_missing_has_no_pane(self):
        assert read_pane_distance(config_with(), 'simulation_3', 'config_stacked.ini') is None

    def test_the_option_gives_the_distance(self):
        config = config_with(pane_distance='0.5')

        assert read_pane_distance(config, 'experiment_geometry', 'config_analysis.ini') == 0.5

    @pytest.mark.parametrize('value', ['x', '0 2 0', '-1', 'nan', 'inf'])
    def test_a_value_that_is_not_a_distance_is_reported_with_its_section(self, value):
        config = config_with(pane_distance=value)

        with pytest.raises(ValueError, match=r'pane_distance = .* in \[experiment_geometry\] of '
                                             r'config_analysis.ini is not a distance'):
            read_pane_distance(config, 'experiment_geometry', 'config_analysis.ini')


class TestAnalysisConfig:
    def test_the_template_writes_no_pane(self, tmp_path, monkeypatch):
        monkeypatch.chdir(tmp_path)
        ConfigDataAnalysis(load_config_file=False)

        config = ConfigDataAnalysis()

        assert config['experiment_geometry']['pane_distance'] == 'None'
        assert config.get_pane_distance() is None

    def test_the_template_writes_a_given_pane_distance(self, tmp_path, monkeypatch):
        monkeypatch.chdir(tmp_path)
        ConfigDataAnalysis(load_config_file=False, pane_distance=0.5)

        assert ConfigDataAnalysis().get_pane_distance() == 0.5

    def test_a_config_written_before_the_option_has_no_pane(self, tmp_path, monkeypatch):
        monkeypatch.chdir(tmp_path)
        config_text = '[experiment_geometry]\ncamera_position = 0 0 2\n'
        (tmp_path / 'config_analysis.ini').write_text(config_text, encoding='utf-8')

        assert ConfigDataAnalysis().get_pane_distance() is None


STACKED_CONFIG = """
[DEFAULT]
solver = linear
num_layers = 10
domain_bounds = 0 3
reference_property = sum_col_val

[simulation_0]
path = {tmp_path}/cam0
camera_position = 0 0 2
pane_distance = 2.0

[simulation_1]
path = {tmp_path}/cam1
camera_position = 0 0 0.5

[led_arrays]
0 = 0:0 1:0
"""
STACKED_CONFIG_WITHOUT_PANES = STACKED_CONFIG.replace('pane_distance = 2.0\n', '')


def load_stacked_config(tmp_path, config_text=STACKED_CONFIG):
    path = tmp_path / ConfigDataStacked.DEFAULT_FILENAME
    path.write_text(config_text.format(tmp_path=tmp_path.as_posix()), encoding='utf-8')
    return ConfigDataStacked(filename=str(path))


class ExperimentSpy:
    """Stands in for the Experiment of a camera, which would read the LEDs of its run."""

    def __init__(self, layers, led_array, camera, path, channel, merge_led_arrays):
        self.camera = camera


def build_stacked_solver(tmp_path, monkeypatch, config_text=STACKED_CONFIG):
    monkeypatch.setattr(stacked_module, 'Experiment', ExperimentSpy)
    monkeypatch.setattr(StackedExtinctionCoefficients, '_build_sub_solver', lambda self, exp, sim_idx: None)
    return StackedExtinctionCoefficients(load_stacked_config(tmp_path, config_text), led_array_id=0, channel=0)


class TestStackedAnalysis:
    def test_every_camera_has_its_own_pane_distance(self, tmp_path):
        config = load_stacked_config(tmp_path)

        assert config.get_pane_distance(0) == 2.0
        assert config.get_pane_distance(1) is None

    def test_the_template_leaves_the_pane_distance_commented_out(self, tmp_path):
        path = tmp_path / ConfigDataStacked.DEFAULT_FILENAME
        ConfigDataStacked.create_template(filename=str(path), num_simulations=2)

        config = ConfigDataStacked(filename=str(path))

        assert [config.get_pane_distance(sim_idx) for sim_idx in range(2)] == [None, None]

    def test_the_analysis_puts_every_camera_behind_its_own_pane(self, tmp_path, monkeypatch):
        solver = build_stacked_solver(tmp_path, monkeypatch)

        assert [experiment.camera.pane_distance for experiment in solver.sub_experiments] == [2.0, None]

    def test_the_header_names_only_the_cameras_behind_a_pane(self, tmp_path, monkeypatch):
        solver = build_stacked_solver(tmp_path, monkeypatch)

        assert solver._pane_info() == 'pane_distance: sim0 2.0 | '

    def test_without_a_pane_the_header_stays_as_it_was(self, tmp_path, monkeypatch):
        solver = build_stacked_solver(tmp_path, monkeypatch, STACKED_CONFIG_WITHOUT_PANES)

        assert solver._pane_info() == ''
