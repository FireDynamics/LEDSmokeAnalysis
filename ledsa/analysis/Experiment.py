import configparser as cp
import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Optional, Tuple

import numpy as np


class InconsistentDistanceError(ValueError):
    """The light path of an LED does not add up to its distance from the camera.

    The traversed distances are accumulated per layer, so the part of a light path that lies
    outside the layers is lost. Computing extinction coefficients from such a path would
    attribute the dimming of the whole path to the layers covering a part of it.
    """


class PaneGeometryError(ValueError):
    """The window pane in front of the camera does not lie between the camera and an LED.

    The light path of an LED through the smoke starts where its line of sight crosses the pane,
    pane_distance away from the camera. An LED that is not farther away from the camera than the
    pane has no such crossing.
    """


@dataclass
class Layer:
    """
    Represents a spatial layer with a bottom and top border.

    :ivar bottom_border: The z-coordinate of the bottom border of the layer
    :vartype bottom_border: float
    :ivar top_border: The z-coordinate of the top border of the layer
    :vartype top_border: float
    """
    bottom_border: float
    top_border: float

    def __contains__(self, obj):
        """
        Check if a given object's z-coordinate falls within this layer.

        :param obj: The object to check
        :type obj: object with attribute 'pos_z'
        :return: True if object is contained within the layer, False otherwise
        :rtype: bool
        """
        if self.bottom_border <= obj.pos_z < self.top_border:
            return True
        return False


@dataclass
class Layers:
    """
    Represents a collection of spatial layers.

    :ivar amount: The number of layers
    :vartype amount: int
    :ivar bottom_border: The z-coordinate of the bottom-most border of the layers
    :vartype bottom_border: float
    :ivar top_border: The z-coordinate of the top-most border of the layers
    :vartype top_border: float
    """
    amount: int
    bottom_border: float
    top_border: float
    layers: [Layer] = field(init=False)
    borders: np.ndarray = field(init=False)

    def __post_init__(self):
        self.layers = []
        self.borders = np.linspace(self.bottom_border, self.top_border, self.amount + 1)
        for i in range(self.amount):
            self.layers.append(Layer(self.borders[i], self.borders[i + 1]))

    def __getitem__(self, layer):
        return self.layers[layer]

    def __str__(self):
        return f'num_of_layers: {self.amount}, bottom_border: {self.bottom_border}, top_border: {self.top_border}\n'

    def __repr__(self):
        return f'Layers(amount={self.amount}, bottom_border={self.bottom_border}, top_border={self.top_border})'


@dataclass
class Camera:
    """
    Contains a camera's position in 3D space.

    :ivar pos_x: The x-coordinate of the camera
    :vartype pos_x: float
    :ivar pos_y: The y-coordinate of the camera
    :vartype pos_y: float
    :ivar pos_z: The z-coordinate of the camera
    :vartype pos_z: float
    :ivar pane_distance: Distance [m] between the camera and the window pane it looks through into
        the smoke, None if the smoke reaches the camera. The pane is taken as normal to the line of
        sight, so the first pane_distance of every line of sight is free of smoke. The transmission
        of the pane itself is the same in the reference images and cancels in I/I0, and the
        refraction in a pane with parallel faces shifts the line of sight by less than the thickness
        of the pane, so neither is taken into account.
    :vartype pane_distance: float or None
    """
    pos_x: float
    pos_y: float
    pos_z: float
    pane_distance: Optional[float] = None

    def __post_init__(self):
        if self.pane_distance is not None:
            distance = float(self.pane_distance)
            if not np.isfinite(distance) or distance < 0:
                raise ValueError(f'The distance between the camera and the pane has to be a finite number of '
                                 f'at least 0, got {self.pane_distance}.')
            self.pane_distance = distance

    def __str__(self):
        if self.pane_distance is None:
            return f'Camera: ({self.pos_x}, {self.pos_y}, {self.pos_z})\n'
        return f'Camera: ({self.pos_x}, {self.pos_y}, {self.pos_z}), pane_distance: {self.pane_distance}\n'

    def __repr__(self):
        pane = '' if self.pane_distance is None else f', pane_distance={self.pane_distance}'
        return f'Camera(pos_x={self.pos_x}, pos_y={self.pos_y}, pos_z={self.pos_z}{pane})'


@dataclass
class LED:
    """
    Represents a LED light source in 3D space.

    :ivar id: The identifier for the LED
    :vartype id: int
    :ivar pos_x: The x-coordinate of the LED
    :vartype pos_x: float
    :ivar pos_y: The y-coordinate of the LED
    :vartype pos_y: float
    :ivar pos_z: The z-coordinate of the LED
    :vartype pos_z: float
    """
    id: int
    pos_x: float
    pos_y: float
    pos_z: float


class Experiment:
    """
    Represents an experimental setup involving layers, an LED array, and a camera.

    :ivar layers: The spatial layers involved in the experiment.
    :vartype layers: Layers
    :ivar led_array: The identifier for the LED array.
    :vartype led_array: int
    :ivar camera: The camera involved in the experiment.
    :vartype camera: Camera
    :ivar leds: List of LED light sources.
    :vartype leds: List[LED]
    :ivar led_number: Number of LEDs on LED array.
    :vartype led_number: int
    :ivar path: File path for experiment data.
    :vartype path: Path
    :ivar channel: The camera channel to be analysed.
    :vartype channel: int
    :ivar merge_led_arrays: Whether to merge LED arrays.
    :vartype merge_led_arrays: bool
    """
    def __init__(self, layers: Layers, led_array: int, camera: Camera, path=Path('.'), channel=0,
                 merge_led_arrays=False):
        """
        :param layers: The spatial layers involved in the experiment.
        :type layers: Layers
        :param led_array: The identifier for the LED array.
        :type led_array: int
        :param camera: The camera involved in the experiment.
        :type camera: Camera
        :param path: The file path for experiment data, defaults to current directory.
        :type path: Path, optional
        :param channel: The camera channel to be analysed, defaults to 0.
        :type channel: int, optional
        :param merge_led_arrays: Whether to merge LED arrays, defaults to False.
        :type merge_led_arrays: bool, optional
        """
        self.num_leds = None
        self.layers = layers
        self.led_array = led_array
        self.camera = camera
        self.leds = []
        self.path = path
        self.channel = channel
        self.merge_led_arrays = merge_led_arrays

        try:
            self.set_leds()
        except IOError as err:
            print(err)

    def __str__(self):
        out = f'channel: {self.channel}, led_array: {self.led_array}\n' + \
              str(self.layers) + \
              str(self.camera)
        return out

    def calc_smoke_path_start(self, led: LED) -> Tuple[float, float, float]:
        """
        Find the point where the light path between an LED and the camera enters the smoke.

        Without a pane, the smoke reaches the camera and the light path starts there. Behind a pane,
        it starts where the line of sight of the LED crosses the pane, pane_distance away from the
        camera, because the part between the camera and the pane is free of smoke. The pane is taken
        as normal to the line of sight. A flat pane normal to the optical axis of the camera is
        crossed by a line of sight at an angle theta to the axis pane_distance / cos(theta) away
        from the camera, at 15 degrees 3.5 % farther; this is negligible as long as the pane is
        close to the camera compared to the LEDs.

        :param led: The LED light source
        :type led: LED
        :return: Global X, Y, Z coordinates of the start of the light path through the smoke
        :rtype: Tuple[float, float, float]
        :raises PaneGeometryError: If the LED is not farther away from the camera than the pane.
        """
        camera = (self.camera.pos_x, self.camera.pos_y, self.camera.pos_z)
        pane_distance = self.camera.pane_distance
        if pane_distance is None:
            return camera
        camera_pos = np.array(camera, dtype=float)
        led_pos = np.array([led.pos_x, led.pos_y, led.pos_z], dtype=float)
        led_distance = np.linalg.norm(led_pos - camera_pos)
        if pane_distance >= led_distance:
            raise PaneGeometryError(
                f'LED {led.id} of LED array {self.led_array} at ({led.pos_x}, {led.pos_y}, {led.pos_z}) '
                f'is {led_distance:.6g} away from the camera at {camera}, so it is not behind the pane '
                f'{pane_distance:g} away from the camera and its line of sight does not pass the pane '
                f'into the smoke. Check pane_distance and camera_position in the analysis '
                f'configuration: the pane has to lie between the camera and every LED.')
        return tuple(camera_pos + pane_distance / led_distance * (led_pos - camera_pos))

    def calc_traversed_dist_per_layer(self, led: LED) -> np.ndarray:
        """
        Calculate the distance traversed by light from an LED through each layer to the camera.

        Behind a pane, only the part of the light path between the pane and the LED is traversed in
        the smoke, see calc_smoke_path_start.

        :param led: The LED light source
        :type led: LED
        :return: Array of distances traversed in each layer
        :rtype: np.ndarray
        :raises InconsistentDistanceError: If the distances do not add up to the length of the light
                                           path through the smoke.
        :raises PaneGeometryError: If the LED is not farther away from the camera than the pane.
        """
        start = self.calc_smoke_path_start(led)
        start_x, start_y, start_z = start
        horizontal_dist = np.sqrt((start_x - led.pos_x) ** 2 + (start_y - led.pos_y) ** 2)
        alpha = np.arctan((led.pos_z - start_z) / horizontal_dist)
        if alpha == 0:
            distance_per_layer = self.calc_traversed_dist_in_plane(led, start)
        else:
            distance_per_layer = self.calc_traversed_dist_per_layer_with_nonzero_alpha(alpha, led, start)

        if not self.distance_calculation_is_consistent(distance_per_layer, led, start=start):
            path_length = np.sqrt((start_x - led.pos_x) ** 2 +
                                  (start_y - led.pos_y) ** 2 +
                                  (start_z - led.pos_z) ** 2)
            if self.camera.pane_distance is None:
                path_end = f'while the LED is {path_length:.6g} away from the camera'
                range_end = 'the camera'
            else:
                path_end = (f'while the LED is {path_length:.6g} away from the point ({start_x:.6g}, '
                            f'{start_y:.6g}, {start_z:.6g}) where its line of sight crosses the pane, '
                            f'{self.camera.pane_distance:g} away from the camera')
                range_end = 'the points where the lines of sight cross the pane'
            raise InconsistentDistanceError(
                f'The light path of LED {led.id} of LED array {self.led_array} traverses the layers '
                f'over {np.sum(distance_per_layer):.6g}, {path_end}. LED position: ({led.pos_x}, '
                f'{led.pos_y}, {led.pos_z}), camera position: ({self.camera.pos_x}, {self.camera.pos_y}, '
                f'{self.camera.pos_z}), layers: {self.layers.bottom_border} to {self.layers.top_border}. '
                f'The part of the light path outside the layers belongs to no layer, so set domain_bounds '
                f'in the analysis configuration to a range that holds {range_end} and every LED of the '
                f'array.')
        return distance_per_layer

    def calc_traversed_dist_in_plane(self, led: LED,
                                     start: Optional[Tuple[float, float, float]] = None) -> np.ndarray:
        """
        Calculate the distance traversed by light from an LED in a plane to the camera.

        :param led: The LED light source
        :type led: LED
        :param start: Start of the light path through the smoke, the camera position if None
        :type start: Tuple[float, float, float] or None
        :return: Array of distances traversed in each layer within the plane
        :rtype: np.ndarray
        """
        start_x, start_y, start_z = self._camera_position_if_none(start)
        distance_per_layer = np.zeros(self.layers.amount)
        horizontal_dist = np.sqrt((start_x - led.pos_x) ** 2 + (start_y - led.pos_y) ** 2)
        for layer in range(self.layers.amount):
            layer_bottom = self.layers.borders[layer]
            layer_top = self.layers.borders[layer + 1]
            if layer_bottom <= start_z < layer_top:
                distance_per_layer[layer] = horizontal_dist
        return distance_per_layer

    def calc_traversed_dist_per_layer_with_nonzero_alpha(
            self, alpha: float, led: LED, start: Optional[Tuple[float, float, float]] = None) -> np.ndarray:
        """
        Calculate the distance traversed by light from an LED through each layer to the camera,
        taking into account a non-zero angle of incidence.

        :param alpha: The angle of incidence
        :type alpha: float
        :param led: The LED light source
        :type led: LED
        :param start: Start of the light path through the smoke, the camera position if None
        :type start: Tuple[float, float, float] or None
        :return: Array of distances traversed in each layer considering the angle
        :rtype: np.ndarray
        """
        start_z = self._camera_position_if_none(start)[2]
        distance_per_layer = np.zeros(self.layers.amount)
        for layer in range(self.layers.amount):
            layer_bottom = self.layers.borders[layer]
            layer_top = self.layers.borders[layer + 1]
            th = self.calc_traversed_height_in_layer(led.pos_z, layer_bottom, layer_top, start_z)
            distance_per_layer[layer] = np.abs(th / np.sin(alpha))
        return distance_per_layer

    def calc_traversed_height_in_layer(self, led_height: float, layer_bot: float, layer_top: float,
                                       start_height: Optional[float] = None) -> float:
        """
        Calculate the vertical distance (height) traversed by light from an LED within a layer.

        :param led_height: The z-coordinate of the LED
        :type led_height: float
        :param layer_bot: The z-coordinate of the bottom of the layer
        :type layer_bot: float
        :param layer_top: The z-coordinate of the top of the layer
        :type layer_top: float
        :param start_height: The z-coordinate of the start of the light path through the smoke, that
            of the camera if None
        :type start_height: float or None
        :return: The vertical distance traversed within the layer
        :rtype: float
        """
        if start_height is None:
            start_height = self.camera.pos_z
        top_end = max(start_height, led_height)
        bot_end = min(start_height, led_height)
        bot = max(bot_end, layer_bot)
        top = min(top_end, layer_top)
        h = top - bot
        if h < 0:
            h = 0
        return h

    def distance_calculation_is_consistent(self, distance_per_layer: np.ndarray, led: LED, silent=True,
                                           start: Optional[Tuple[float, float, float]] = None) -> bool:
        """
        Check if the computed distance values for each layer are consistent with the Euclidean distance.

        :param distance_per_layer: Array of distances traversed in each layer
        :type distance_per_layer: np.ndarray
        :param led: The LED light source
        :type led: LED
        :param silent: If set to False, prints out debugging information, defaults to True
        :type silent: bool, optional
        :param start: Start of the light path through the smoke, the camera position if None
        :type start: Tuple[float, float, float] or None
        :return: True if computations are consistent, False otherwise
        :rtype: bool
        """
        start_x, start_y, start_z = self._camera_position_if_none(start)
        if np.abs(np.sum(distance_per_layer) - np.sqrt((start_x - led.pos_x) ** 2 +
                                                       (start_y - led.pos_y) ** 2 +
                                                       (start_z - led.pos_z) ** 2)) > 1e-6:
            if not silent:
                print("error in distance computation, start_x: {}, start_y: {} start_z: {}, led_x: {}, led_y: {}, "
                      "led_z: {}".format(start_x, start_y, start_z, led.pos_x, led.pos_y, led.pos_z))
            return False
        return True

    def _camera_position_if_none(self, start: Optional[Tuple[float, float, float]]) -> Tuple[float, float, float]:
        """Return start, or the position of the camera if start is None."""
        if start is None:
            return self.camera.pos_x, self.camera.pos_y, self.camera.pos_z
        return start

    def set_leds(self) -> None:
        """
        Initialize the LED instances involved in the experiment based on loaded data.

        """
        ids = self.get_led_ids()
        x, y, z = self.get_led_positions(ids)
        for i in range(len(ids)):
            self.leds.append(LED(ids[i], x[i], y[i], z[i]))
        self.num_leds = len(ids)

    def get_led_ids(self) -> np.ndarray:
        """
        Retrieve the IDs for the LEDs involved in the experiment.

        :return: Array of LED IDs
        :rtype: np.ndarray
        """
        file_name_extension = '_merge' if self.merge_led_arrays != 'None' else ''
        file_name = f'led_array_indices_{self.led_array:03d}{file_name_extension}.csv'
        file_path = os.path.join(self.path, 'analysis', file_name)

        led_array_indices = np.loadtxt(file_path, dtype=int)
        return led_array_indices

    def get_led_positions(self, ids: np.ndarray) -> List[np.ndarray]:
        """
        Retrieve the 3D positions for a set of LEDs.

        :param ids: Array of LED identifiers
        :type ids: np.ndarray
        :return: List of arrays containing x, y, and z coordinates for each LED
        :rtype: List[np.ndarray]
        """
        file_path = os.path.join(self.path, 'analysis', 'led_search_areas_with_coordinates.csv')
        search_areas_all = np.loadtxt(file_path, delimiter=',')
        search_areas_led_array = []
        for led_id in ids:
            search_areas_led_array.append(search_areas_all[led_id])
        search_areas_led_array = np.array(search_areas_led_array)
        return [search_areas_led_array[:, 3], search_areas_led_array[:, 4], search_areas_led_array[:, 5]]


def read_pane_distance(config: cp.ConfigParser, section: str, file_name: str) -> Optional[float]:
    """
    Read the distance between a camera and the window pane it looks through into the smoke.

    The option pane_distance is optional, so a config file written before it, or one that sets it to
    None, keeps the light paths reaching the camera through the smoke.

    :param config: The loaded config file
    :type config: configparser.ConfigParser
    :param section: Section holding the options of the camera
    :type section: str
    :param file_name: Name of the config file, for the error messages
    :type file_name: str
    :return: The distance [m], None if the option is not set
    :rtype: Optional[float]
    :raises ValueError: If the value is not a finite number of at least 0.
    """
    raw = config.get(section, 'pane_distance', fallback=None)
    raw = '' if raw is None else raw.strip()
    if raw.lower() in ('', 'none'):
        return None
    try:
        distance = float(raw)
    except ValueError:
        distance = np.nan
    if not np.isfinite(distance) or distance < 0:
        raise ValueError(f'pane_distance = {raw} in [{section}] of {file_name} is not a distance: give the '
                         f'distance [m] between the camera and the window pane as a number of at least 0, or '
                         f'None for a camera without a pane.')
    return distance
