import os
import warnings

import numpy as np
from scipy import linalg
from scipy.optimize import curve_fit

from ledsa.core.ConfigData import ConfigData
from ledsa.core.file_handling import read_table

warnings.filterwarnings("ignore",
                        message="Covariance of the parameters could not be estimated")  # TODO: Find a better workaround for not letting tests crash


class LED:
    """
    Represents an LED with its physical position and pixel position.

    :ivar led_id: The LED's identifier.
    :vartype led_id: float
    :ivar pos: The LED's 3D position.
    :vartype pos: np.ndarray
    :ivar pix_pos: The LED's pixel position in 2D space.
    :vartype pix_pos: np.ndarray
    """
    def __init__(self, led_id=None, pos=None, pix_pos=None):
        self.id = led_id
        self.pos = pos
        self.pix_pos = pix_pos

    def conversion_matrix(self, led2) -> np.ndarray:
        """
        Compute a conversion matrix between the current LED and another LED.

        :param led2: Another LED instance.
        :type led2: LED
        :return: The conversion matrix between the two LEDs.
        :rtype: np.ndarray
        """
        a = np.atleast_2d(np.array([self.pix_pos, led2.pix_pos]))
        b = np.atleast_2d(np.array([self.pos, led2.pos]))
        x = linalg.solve(a, b, assume_a='gen')
        return np.transpose(x)

    def get_led_array(self, led2) -> np.ndarray:
        """
        Compute the X and Y pixel delta between the current LED and another LED.

        :param led2: Another LED instance.
        :type led2: LED
        :return: The X and Y pixel delta between the two LEDs.
        :rtype: np.ndarray
        """
        return led2.pix_pos - self.pix_pos


def calculate_coordinates() -> None:
    """Calculate and save the 3D and 2D coordinates of LEDs."""
    coordinates_3d = _calculate_3d_coordinates()
    coordinates_2d = _calculate_2d_coordinates(coordinates_3d[0:, 3:6])
    coord = np.append(coordinates_3d, coordinates_2d.T, axis=1)

    file_path = os.path.join('analysis', 'led_search_areas_with_coordinates.csv')
    np.savetxt(file_path, coord, header='LED id, pixel position x, pixel position y, x, y, z, width, height',
               fmt='%d,%d,%d,%f,%f,%f,%f,%f')
    print(f"\nCoordinates successfully saved in {file_path}")


# calculates from the measured room coordinates of two points per led array the room coordinates of each other point by
# calculating the linear transformation between pixel and room coordinates and applying it to the projection of each led
# onto the corresponding LED array
def _calculate_3d_coordinates() -> np.ndarray:
    """
    Calculate the 3D coordinates of LEDs using a configuration and search areas.

    :return: An array containing the LED IDs, the pixel positions and the 3D coordinates of the LEDs.
    :rtype: np.ndarray
    """
    conf = ConfigData(load_config_file=True)
    file_path = os.path.join('analysis', 'led_search_areas.csv')
    search_areas = read_table(file_path, delim=',')
    search_areas = np.pad(search_areas, ((0, 0), (0, 3)), constant_values=(-1, -1))
    if conf['analyse_positions']['led_array_edge_coordinates'] == 'None':
        conf.in_led_array_edge_coordinates()
        conf.save()
    led_coordinates = conf.get2dnparray('analyse_positions', 'led_array_edge_coordinates', 6, float)
    print("Loaded coordinates from config.ini:")
    print(led_coordinates)

    # loop over the led-arrays
    for ledarray in range(int(conf['analyse_positions']['num_arrays'])):
        file_path = os.path.join('analysis', f'led_array_indices_{ledarray:03d}.csv')
        led_array_indices = read_table(file_path)

        # get the edge leds of an array to calculate from them the conversion matrix for this array
        # Use the first LED in the LED array indices file as the top edge LED
        idx = np.where(search_areas[:, 0] == led_array_indices[-1])[0]
        pos = led_coordinates[ledarray][0:3]
        pix_pos = np.array([search_areas[idx, 1], search_areas[idx, 2]])
        top_led = LED(led_array_indices[0], pos, pix_pos)

        # Use the last LED in the LED array indices file as the bottom edge LED
        idx = np.where(search_areas[:, 0] == led_array_indices[0])[0]
        pos = led_coordinates[ledarray, 3:6]
        pix_pos = np.array([search_areas[idx, 1], search_areas[idx, 2]])
        bot_led = LED(led_array_indices[-1], pos, pix_pos)

        x = top_led.conversion_matrix(bot_led)
        led_array = top_led.get_led_array(bot_led)

        # loop over all leds in the array
        for led in led_array_indices:
            idx = np.where(search_areas[:, 0] == led)[0]
            pix_pos = np.array([search_areas[idx, 1], search_areas[idx, 2]])
            pix_pos = _orth_projection(pix_pos, led_array, top_led.pix_pos)
            pos = np.transpose(x @ pix_pos)
            search_areas[idx, -3:] = pos
    return search_areas


# uses least squares to fit a plane through the points, projects the points onto the plane and changes the coordinate
# system such that there is a width axis in [0,inf) and a height axis which stays the same as the z axis
def _calculate_2d_coordinates(points: np.ndarray) -> np.ndarray:
    """
    Calculate 2D coordinates by projecting 3D points onto a plane.

    :param points: An array of 3D points.
    :type points: np.ndarray
    :return: An array of 2D coordinates.
    :rtype: np.ndarray
    """
    if points.shape[1] == 3 and points.shape[0] != 3:
        points = points.T
    plane = _fit_plane(points)
    projections = _project_points_to_plane(points, plane)
    return _get_plane_coordinates(projections, plane)


def _orth_projection(point: np.ndarray, led_array, point_on_led_array: np.ndarray) -> np.ndarray:
    """
    Project a point orthogonally onto a LED array.

    :param point: The point to project.
    :type point: np.ndarray
    :param led_array: The LED array's direction vector.
    :type led_array: np.ndarray
    :param point_on_led_array: A point on the LED array.
    :type point_on_led_array: np.ndarray
    :return: The orthogonal projection of the point onto the LED array.
    :rtype: np.ndarray
    """
    # normalized direction vector of LED array
    led_array_hat = (led_array / np.linalg.norm(led_array)).flatten()

    # vector between the LED array and the normalized direction vector of the LED array
    led_array_pos = point_on_led_array.flatten() - point_on_led_array.flatten().dot(led_array_hat) * led_array_hat

    # projection of the point onto the LED array
    projection = point.flatten().dot(led_array_hat) * led_array_hat + led_array_pos
    return projection

def _fit_plane(points: np.ndarray) -> np.ndarray:
    """
    Fit a plane through the given points using the least squares method.

    :param points: An array of points with 3d physical coordinates to fit the plane through.
    :type points: np.ndarray
    :return: The optimization coefficients of the fitted plane.
    :rtype: np.ndarray
    """
    def plane_func(point, a, b, d):
        return -1. / b * (a * point[0] + d)

    popt, pcov = curve_fit(plane_func, points, points[1])
    popt = np.insert(popt, 2, 0)
    return popt


def _project_points_to_plane(points: np.ndarray, plane: np.ndarray) -> np.ndarray:
    """
    Project points onto a specified plane.

    :param points: An array of points to project with 3d physical coordinates to fit the plane through.
    :type points: np.ndarray
    :param plane: The coefficients of the plane as normal vector.
    :type plane: np.ndarray
    :return: An array of the projected points onto the plane with 3d physical coordinates.
    :rtype: np.ndarray
    """
    t = -(plane[0] * points[0] + plane[1] * points[1] + plane[3]) / (plane[0] ** 2 + plane[1] ** 2)
    t = np.atleast_2d(t)
    plane = np.atleast_2d(plane[0:3])
    print(points.shape, plane.T.shape, t.shape)
    projection = points + plane.T * t
    return projection


# Transforms the coordinate system of points on a plane orthogonal to the xy-plane from 3D to 2D.
def _get_plane_coordinates(points: np.ndarray, plane: np.ndarray) -> np.ndarray:
    """
    Convert 3D points on a plane to 2D plane coordinates.

    :param points:An array of points to project with 3d physical coordinates on a plane.
    :type points: np.ndarray
    :param plane: The coefficients of the plane as normal vector.
    :type plane: np.ndarray
    :return: An array of the 2D physical coordinates of the points on the plane.
    :rtype: np.ndarray
    """
    plane_coordinates = np.ndarray((2, points.shape[1]))

    # move the plane so it goes through the origin
    y = points[1] + plane[3] / plane[1]

    # coordinate transformation x -> width
    plane_coordinates[0] = np.sqrt(points[0] ** 2 + y ** 2) / (np.dot([0, 1], plane[0:2]) / np.linalg.norm(plane[0:2]))

    # transform so width is in [0,inf)
    plane_coordinates[0] = plane_coordinates[0] - np.min(plane_coordinates[0])

    plane_coordinates[1] = points[2]
    return plane_coordinates
