from dataclasses import dataclass, field

import numpy as np


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
            self.layers.append(Layer(self.borders[i], self.borders[i + 1]))

    def __getitem__(self, layer):
        return self.layers[layer]

    def __str__(self):
        return f'num_of_layers: {self.amount}, bottom_border: {self.bottom_border}, top_border: {self.top_border}\n'

    def __repr__(self):
        return f'Layers(amount={self.amount}, bottom_border={self.bottom_border}, top_border={self.top_border})'


@dataclass
class Camera:
    pos_x: float
    pos_y: float
    pos_z: float

    def __str__(self):
        return f'Camera: ({self.pos_x}, {self.pos_y}, {self.pos_z})\n'

    def __repr__(self):
        return f'Camera(pos_x={self.pos_x}, pos_y={self.pos_y}, pos_z={self.pos_z})'


@dataclass
class LED:
    id: int
    pos_x: float
    pos_y: float
    pos_z: float


class TestExperiment:
    def __init__(self, layers: Layers, camera: Camera):
        self.layers = layers
        self.camera = camera
        self.leds = []
        self.extinction_coefficients = np.zeros(layers.amount)

    def add_led(self, pos_x, pos_y, pos_z) -> None:
        led_id = len(self.leds)
        self.leds.append(LED(led_id, pos_x, pos_y, pos_z))

    def set_extinction_coefficients(self, extinction_coefficients_array):
        self.extinction_coefficients = extinction_coefficients_array

    def calc_traversed_dist_per_layer(self, led: LED) -> np.ndarray:
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

    def calc_led_transmission(self, led):
        dist_per_layer = self.calc_traversed_dist_per_layer(led)
        transmission = np.exp(- sum(dist_per_layer * self.extinction_coefficients))
        return transmission

    def calc_all_led_transmissions(self):
        transmissions = []
        for led in self.leds:
            transmissions.append(self.calc_led_transmission(led))
        return transmissions

# layers = Layers(10, 0, 1.2)
# camera = Camera(0, 0, 0)
# ex = TestExperiment(layers=layers, camera=camera)
# ex.add_led(0, 0, 0.7)
# ex.add_led(0, 0, 0.3)
# print(ex.calc_traversed_dist_per_layer(ex.leds[0]))
# ex.set_extinction_coefficients(np.ones(10)*0.01)
# print(ex.extinction_coefficients)
#
# print(ex.calc_led_transmission(ex.leds[0]))
