from ..core import _led_helper as ledh
from ..core import ledsa_conf as lc
import os
# from math import *
from scipy import linalg
import numpy as np

# os path separator
sep = os.path.sep


class LED:
    def __init__(self, id=None, pos=None, pix_pos=None):
        self.id = id
        self.pos = pos
        self.pix_pos = pix_pos

    def conversion_matrix(self, led2):
        a = np.array([self.pix_pos, led2.pix_pos])
        b = np.array([self.pos, led2.pos])
        x = linalg.solve(a, b)
        return np.transpose(x)

    def get_line(self, led2):
        return led2.pix_pos - self.pix_pos


# projects a point orthogonal onto a line
# the arguments are numpy arrays with two elements with point and point_on_line containing the pixel coordinates of the
# point to project and one point on the line respectively and line containing the direction of the line
# (point B - point A)
def orth_projection(point, line, point_on_line):
    # normalized direction vector of line
    line_hat = (line / np.linalg.norm(line)).flatten()
    # vector between the line and the normalized direction vector of the line
    line_pos = point_on_line.flatten() - point_on_line.flatten().dot(line_hat)*line_hat
    # projection of the point onto the line
    projection = point.flatten().dot(line_hat)*line_hat + line_pos
    return projection


# calculates from the measured room coordinates of two points per led array the room coordinates of each other point by
# calculating the linear transformation between pixel and room coordinates and applying it to the projection of each led
# onto the corresponding line
def calculate_coordinates():
    conf = lc.ConfigData(load_config_file=True)
    search_areas = ledh.load_file('.{}analysis{}led_search_areas.csv'.format(sep, sep), delim=',')
    search_areas = np.pad(search_areas, ((0, 0), (0, 3)), constant_values=(-1, -1))
    led_coordinates = conf.get2dnparray('analyse_positions', 'line_edge_coordinates', 6, float)
    print("Loaded coordinates from config.ini:")
    print(led_coordinates)
    edge_leds = conf.get2dnparray('analyse_positions', 'line_edge_indices')

    # loop over the led-arrays
    for ledarray in range(int(conf['DEFAULT']['num_of_arrays'])):
        line_indices = ledh.load_file('.{}analysis{}line_indices_{:03d}.csv'.format(sep, sep, ledarray))

        # get the edge leds of an array to calculate from them the conversion matrix for this array
        idx = np.where(search_areas[:, 0] == edge_leds[ledarray, 0])[0]
        pos = led_coordinates[ledarray][0:3]
        pix_pos = np.array([search_areas[idx, 1], search_areas[idx, 2]])
        top_led = LED(line_indices[0], pos, pix_pos)

        idx = np.where(search_areas[:, 0] == edge_leds[ledarray, 1])[0]
        pos = led_coordinates[ledarray, 3:6]
        pix_pos = np.array([search_areas[idx, 1], search_areas[idx, 2]])
        bot_led = LED(line_indices[-1], pos, pix_pos)

        x = top_led.conversion_matrix(bot_led)
        line = top_led.get_line(bot_led)

        # loop over all leds in the array
        for led in line_indices:
            idx = np.where(search_areas[:, 0] == led)[0]
            pix_pos = np.array([search_areas[idx, 1], search_areas[idx, 2]])
            pix_pos = orth_projection(pix_pos, line, top_led.pix_pos)
            pos = np.transpose(x @ pix_pos)
            search_areas[idx, -3:] = pos

    np.savetxt('.{}analysis{}led_search_areas_with_coordinates.csv'.format(sep, sep), search_areas,
               header='LED id, pixel position x, pixel position y, x, y, z', fmt='%d,%d,%d,%f,%f,%f')
    print("\nCoordinates successfully saved in analysis{}led_search_areas_with_coordinates.csv".format(sep))
