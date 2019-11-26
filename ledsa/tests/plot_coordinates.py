import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import numpy as np
from ..core._led_helper import load_file
from os import sep


def plot_coordinates():
    leds = load_file('analysis{}led_search_areas_with_coordinates.csv'.format(sep), delim=',')
    print(np.shape(leds))
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')

    ax.scatter(leds[:, 3], leds[:, 4], leds[:, 5])
    ax.set_xbound(0,7)
    ax.set_ybound(0,7)
    ax.set_zbound(0,7)

    plt.show()
