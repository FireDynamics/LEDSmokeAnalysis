from dataclasses import dataclass, field
import numpy as np
from typing import List
from pathlib import Path


@dataclass
class Layer:
    bottom_border: float
    top_border: float

    def __contains__(self, obj):
        if self.bottom_border <= obj.pos_z < self.top_border:
            return True
        return False

@dataclass
class Layers:
    amount: int
    bottom_border: float
    top_border: float
    layers: [Layer] = field(init=False)
    borders: np.ndarray = field(init=False)

    def __post_init__(self):
        self.layers = []
        self.borders = np.linspace(self.bottom_border, self.top_border, self.amount + 1)
        for i in range(self.amount):
            self.layers.append(Layer(self.borders[i], self.borders[i+1]))

    def __getitem__(self, layer):
        return self.layers[layer]

    def __str__(self):
        return f'num_of_layers: {self.amount}, bottom_border: {self.bottom_border}, top_border: {self.top_border}\n'



@dataclass
class Camera:
    pos_x: float
    pos_y: float
    pos_z: float

    def __str__(self):
        return f'Camera: ({self.pos_x}, {self.pos_y}, {self.pos_z})\n'


@dataclass
class LED:
    id: int
    pos_x: float
    pos_y: float
    pos_z: float


class Experiment:
    """Physical structure info about the experiment and distance calculations."""
    def __init__(self, layers: Layers, led_array: int, camera: Camera, path=Path('.'), channel=0):
        self.layers = layers
        self.led_array = led_array
        self.camera = camera
        self.leds = []
        self.led_number = 0
        self.path = path
        self.channel = channel

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
        horizontal_dist = np.sqrt((self.camera.pos_x-led.pos_x)**2 + (self.camera.pos_y-led.pos_y)**2)
        alpha = np.arctan((led.pos_z - self.camera.pos_z) / horizontal_dist)
        if alpha == 0:
            distance_per_layer = self.calc_traversed_dist_in_plane(led)
        else:
            distance_per_layer = self.calc_traversed_dist_per_layer_with_nonzero_alpha(alpha, led)

        if not self.distance_calculation_is_consistent(distance_per_layer, led):
            distance_per_layer = None
        return distance_per_layer

    def calc_traversed_dist_in_plane(self, led: LED) -> np.ndarray:
        distance_per_layer = np.zeros(self.layers.amount)
        horizontal_dist = np.sqrt((self.camera.pos_x - led.pos_x) ** 2 + (self.camera.pos_y - led.pos_y) ** 2)
        for layer in range(self.layers.amount):
            layer_bottom = self.layers.borders[layer]
            layer_top = self.layers.borders[layer + 1]
            if layer_bottom <= self.camera.pos_z < layer_top:
                distance_per_layer[layer] = horizontal_dist
        return distance_per_layer

    def calc_traversed_dist_per_layer_with_nonzero_alpha(self, alpha: float, led: LED) -> np.ndarray:
        distance_per_layer = np.zeros(self.layers.amount)
        for layer in range(self.layers.amount):
            layer_bottom = self.layers.borders[layer]
            layer_top = self.layers.borders[layer + 1]
            th = self.calc_traversed_height_in_layer(led.pos_z, layer_bottom, layer_top)
            distance_per_layer[layer] = np.abs(th / np.sin(alpha))
        return distance_per_layer

    def calc_traversed_height_in_layer(self, led_height: float, layer_bot: float, layer_top: float) -> float:
        top_end = max(self.camera.pos_z, led_height)
        bot_end = min(self.camera.pos_z, led_height)
        bot = max(bot_end, layer_bot)
        top = min(top_end, layer_top)
        h = top - bot
        if h < 0:
            h = 0
        return h

    def distance_calculation_is_consistent(self, distance_per_layer: np.ndarray, led: LED, silent=True) -> bool:
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
        ids = self.get_led_ids()
        x, y, z = self.get_led_positions(ids)
        for i in range(len(ids)):
            self.leds.append(LED(ids[i], x[i], y[i], z[i]))
        self.led_number = len(ids)
        return

    def get_led_ids(self) -> np.ndarray:
        line_indices = np.loadtxt(self.path / 'analysis' / f'line_indices_{self.led_array:03d}.csv', dtype=int)
        return line_indices

    def get_led_positions(self, ids: np.ndarray) -> List[np.ndarray]:
        search_areas_all = np.loadtxt(self.path / 'analysis' / 'led_search_areas_with_coordinates.csv', delimiter=',')
        search_areas_led_array = []
        for led_id in ids:
            search_areas_led_array.append(search_areas_all[led_id])
        search_areas_led_array = np.array(search_areas_led_array)
        return [search_areas_led_array[:, 3], search_areas_led_array[:, 4], search_areas_led_array[:, 5]]
