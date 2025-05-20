import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import List

import numpy as np


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
    """
    pos_x: float
    pos_y: float
    pos_z: float

    def __str__(self):
        return f'Camera: ({self.pos_x}, {self.pos_y}, {self.pos_z})\n'

    def __repr__(self):
        return f'Camera(pos_x={self.pos_x}, pos_y={self.pos_y}, pos_z={self.pos_z})'


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

    def calc_traversed_dist_per_layer(self, led: LED) -> np.ndarray:
        """
        Calculate the distance traversed by light from an LED through each layer to the camera.

        :param led: The LED light source
        :type led: LED
        :return: Array of distances traversed in each layer
        :rtype: np.ndarray
        """
        horizontal_dist = np.sqrt((self.camera.pos_x - led.pos_x) ** 2 + (self.camera.pos_y - led.pos_y) ** 2)
        alpha = np.arctan((led.pos_z - self.camera.pos_z) / horizontal_dist)
        if alpha == 0:
            distance_per_layer = self.calc_traversed_dist_in_plane(led)
        else:
            distance_per_layer = self.calc_traversed_dist_per_layer_with_nonzero_alpha(alpha, led)

        if not self.distance_calculation_is_consistent(distance_per_layer, led):
            distance_per_layer = None
        return distance_per_layer

    def calc_traversed_dist_in_plane(self, led: LED) -> np.ndarray:
        """
        Calculate the distance traversed by light from an LED in a plane to the camera.

        :param led: The LED light source
        :type led: LED
        :return: Array of distances traversed in each layer within the plane
        :rtype: np.ndarray
        """
        distance_per_layer = np.zeros(self.layers.amount)
        horizontal_dist = np.sqrt((self.camera.pos_x - led.pos_x) ** 2 + (self.camera.pos_y - led.pos_y) ** 2)
        for layer in range(self.layers.amount):
            layer_bottom = self.layers.borders[layer]
            layer_top = self.layers.borders[layer + 1]
            if layer_bottom <= self.camera.pos_z < layer_top:
                distance_per_layer[layer] = horizontal_dist
        return distance_per_layer

    def calc_traversed_dist_per_layer_with_nonzero_alpha(self, alpha: float, led: LED) -> np.ndarray:
        """
        Calculate the distance traversed by light from an LED through each layer to the camera,
        taking into account a non-zero angle of incidence.

        :param alpha: The angle of incidence
        :type alpha: float
        :param led: The LED light source
        :type led: LED
        :return: Array of distances traversed in each layer considering the angle
        :rtype: np.ndarray
        """
        distance_per_layer = np.zeros(self.layers.amount)
        for layer in range(self.layers.amount):
            layer_bottom = self.layers.borders[layer]
            layer_top = self.layers.borders[layer + 1]
            th = self.calc_traversed_height_in_layer(led.pos_z, layer_bottom, layer_top)
            distance_per_layer[layer] = np.abs(th / np.sin(alpha))
        return distance_per_layer

    def calc_traversed_height_in_layer(self, led_height: float, layer_bot: float, layer_top: float) -> float:
        """
        Calculate the vertical distance (height) traversed by light from an LED within a layer.

        :param led_height: The z-coordinate of the LED
        :type led_height: float
        :param layer_bot: The z-coordinate of the bottom of the layer
        :type layer_bot: float
        :param layer_top: The z-coordinate of the top of the layer
        :type layer_top: float
        :return: The vertical distance traversed within the layer
        :rtype: float
        """
        top_end = max(self.camera.pos_z, led_height)
        bot_end = min(self.camera.pos_z, led_height)
        bot = max(bot_end, layer_bot)
        top = min(top_end, layer_top)
        h = top - bot
        if h < 0:
            h = 0
        return h

    def distance_calculation_is_consistent(self, distance_per_layer: np.ndarray, led: LED, silent=True) -> bool:
        """
        Check if the computed distance values for each layer are consistent with the Euclidean distance.

        :param distance_per_layer: Array of distances traversed in each layer
        :type distance_per_layer: np.ndarray
        :param led: The LED light source
        :type led: LED
        :param silent: If set to False, prints out debugging information, defaults to True
        :type silent: bool, optional
        :return: True if computations are consistent, False otherwise
        :rtype: bool
        """
        if np.abs(np.sum(distance_per_layer) - np.sqrt((self.camera.pos_x - led.pos_x) ** 2 +
                                                       (self.camera.pos_y - led.pos_y) ** 2 +
                                                       (self.camera.pos_z - led.pos_z) ** 2)) > 1e-6:
            if not silent:
                print("error in distance computation, camera_x: {}, camera_y: {} camera_z: {}, led_x: {}, led_y: {}, "
                      "led_z: {}".format(
                    self.camera.pos_x, self.camera.pos_y, self.camera.pos_z, led.pos_x, led.pos_y, led.pos_z))
            return False
        return True

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
