import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import numpy as np
from ..core.led_helper import load_file
from os import sep


def plot_coordinates():
    leds = load_file('analysis{}led_search_areas_with_coordinates.csv'.format(sep), delim=',')
    print(np.shape(leds))
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')

    ax.scatter(leds[:, 3], leds[:, 4], leds[:, 5])
    c1 = [7.29, 6.46, 2.3]
    c2 = [1.28, 4.88, 2.3]
    f = [4.5, 5.25, 0]
    x = [7.29, 1.28, 4.5]
    y = [6.46, 4.88, 5.25]
    z = [2.3, 2.3, 0]
    ax.scatter(x, y, z, s=30)
    ax.set_xbound(0, 7)
    ax.set_ybound(0, 7)
    ax.set_zbound(0, 7)

    plt.show()


def compare_coordinates():
    leds1 = load_file(f'Cam1{sep}analysis{sep}led_search_areas_with_coordinates.csv', delim=',')
    leds2 = load_file(f'Cam2{sep}analysis{sep}led_search_areas_with_coordinates.csv', delim=',')
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')

    ax.scatter(leds1[:, 3], leds1[:, 4], leds1[:, 5])
    ax.scatter(leds2[:, 3], leds2[:, 4], leds2[:, 5])
    ax.set_xbound(0, 7)
    ax.set_ybound(0, 7)
    ax.set_zbound(0, 7)

    plt.show()
