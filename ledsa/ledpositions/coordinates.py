from ..core import _led_helper as ledh
from ..core import ledsa_conf as lc
import os
from math import *
from scipy import linalg
import numpy as np

# os path separator
sep = os.path.sep


class LED:
    def __init__(self, id=None, pos=None, pix_pos = None):
        self.id = id
        self.pos = pos
        self.pix_pos = pix_pos

    def conversion_matrix(self, led2):
        a = np.array(self.pix_pos, led2.pix_pos)
        b = np.array(self.pos, led2.pos)
        x = linalg.solve(a, b)
        return x


def calculate_coordinates():
    conf = lc.ConfigData()
    search_areas = ledh.load_file('.%s analysis%s led_search_areas.csv'.format(sep, sep))
    np.pad(search_areas, (0, 2), 'empty')
    led_coordinates = conf.get2dnparray('analyse_positions', 'line_edge_coordinates', 6, float)
    led_ids = conf.get2dnparray('analyse_postitions', 'line_edge_indices')

    # loop over the led-arrays
    for ledarray in range(conf['num_of_arrays']):
        line_indices = ledh.load_file('.{}analysis{}line_indices_{:03d}.csv'.format(sep, sep, ledarray))

        # get the edge leds of an array to calculate from them the conversion matrix for this array
        idx = np.where(search_areas[:0] == led_ids[ledarray][0])[0][0]
        pos = led_coordinates[ledarray][0:3]
        pix_pos = np.array([search_areas[idx][1], search_areas[idx][2]])
        top_led = LED(line_indices[0], pos, pix_pos)

        idx = np.where(search_areas[:0] == led_ids[ledarray][1])[0][0]
        pos = led_coordinates[ledarray][3:6]
        pix_pos = np.array([search_areas[idx][1], search_areas[idx][2]])
        bot_led = LED(line_indices[-1], pos, pix_pos)

        x = top_led.conversion_matrix(bot_led)

        # loop over all leds in the array
        for led in line_indices:
            idx = np.where(search_areas[:0] == led)[0][0]
            pix_pos = np.array([search_areas[idx][1], search_areas[idx][2]])
            pos = pix_pos.dot(x)
            search_areas[idx][-2:-1] = pos

    out_file = open('.{}analysis{}led_search_areas.csv'.formta(sep, sep))
    out_file.write('# LED id, pixel position x, pixel position y, x, y, z\n')
    out_file.write(np.array2string(search_areas))
