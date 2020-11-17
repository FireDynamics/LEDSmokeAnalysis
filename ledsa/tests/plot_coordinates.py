import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import numpy as np
from ..core.led_helper import load_file
from os import sep
import pandas as pd


def plot_coordinates():
    leds = load_file('analysis{}led_search_areas_with_coordinates.csv'.format(sep), delim=',')
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')

    ax.scatter(leds[:, 3], leds[:, 4], leds[:, 5], c='b', s=10, marker='.')

    try:
        fire = pd.read_csv('experiment_structure.csv', usecols=lambda x: x.upper() in ['FIRE']).values.tolist()
        cameras = pd.read_csv('experiment_structure.csv',
                              usecols=lambda x: x.upper() in ['CAM1', 'CAM2']).values.tolist()
        corners = pd.read_csv('experiment_structure.csv',
                              usecols=lambda x: x.upper() in ['CORNER1', 'CORNER2', 'CORNER3', 'CORNER4', 'CORNER5',
                                                              'CORNER6', 'CORNER7', 'CORNER8']).values.tolist()
        ax.scatter(fire[0], fire[1], fire[2], s=150, c='r', marker='^')
        ax.scatter(cameras[0][0], cameras[1][0], cameras[2][0], s=300, c='k', marker='$Cam1$')
        ax.scatter(cameras[0][1], cameras[1][1], cameras[2][1], s=300, c='k', marker='$Cam2$')
        ax.scatter(corners[0], corners[1], corners[2], s=40, c='k', marker='+')
    except IOError:
        print('experiment_structure.csv not found.')
        print('Camera position, fire and corners are not plotted.')
    ax.set_xbound(0, 10)
    ax.set_ybound(0, 10)
    ax.set_zbound(0, 4)

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
