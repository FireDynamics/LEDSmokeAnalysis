import numpy as np
from scipy import linalg
from scipy.optimize import curve_fit

from ledsa.core.ConfigData import ConfigData
from ledsa.core.file_handling import read_table, sep


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


def calculate_coordinates():

    coordinates_3d = calculate_3d_coordinates()
    coordinates_2d = calculate_2d_coordinates(coordinates_3d[0:, 3:6])
    coord = np.append(coordinates_3d, coordinates_2d.T, axis=1)
    np.savetxt('.{}analysis{}led_search_areas_with_coordinates.csv'.format(sep, sep), coord,
               header='LED id, pixel position x, pixel position y, x, y, z, width, height',
               fmt='%d,%d,%d,%f,%f,%f,%f,%f')
    print("\nCoordinates successfully saved in analysis{}led_search_areas_with_coordinates.csv".format(sep))


# calculates from the measured room coordinates of two points per led array the room coordinates of each other point by
# calculating the linear transformation between pixel and room coordinates and applying it to the projection of each led
# onto the corresponding line
def calculate_3d_coordinates():
    conf = ConfigData(load_config_file=True)
    search_areas = read_table('.{}analysis{}led_search_areas.csv'.format(sep, sep), delim=',')
    search_areas = np.pad(search_areas, ((0, 0), (0, 3)), constant_values=(-1, -1))
    if conf['analyse_positions']['line_edge_coordinates'] == 'None':
        conf.in_line_edge_coordinates()
        conf.save()
    led_coordinates = conf.get2dnparray('analyse_positions', 'line_edge_coordinates', 6, float)
    print("Loaded coordinates from config.ini:")
    print(led_coordinates)

    if conf['analyse_positions']['line_edge_indices'] == 'None':
        conf.in_line_edge_indices()
        conf.save()
    edge_leds = conf.get2dnparray('analyse_positions', 'line_edge_indices')
    print("Loaded line edge indices from config.ini:")
    print(edge_leds)
    if edge_leds.shape[0] != led_coordinates.shape[0]:
        exit("The number of coordinate sets does not match the number of LED line edge indices!")
    # loop over the led-arrays
    for ledarray in range(int(conf['analyse_positions']['num_of_arrays'])):
        line_indices = read_table('.{}analysis{}line_indices_{:03d}.csv'.format(sep, sep, ledarray))

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
            pix_pos = _orth_projection(pix_pos, line, top_led.pix_pos)
            pos = np.transpose(x @ pix_pos)
            search_areas[idx, -3:] = pos

    return search_areas


# uses least squares to fit a plane through the points, projects the points onto the plane and changes the coordinate
# system such that there is a width axis in [0,inf) and a height axis which stays the same as the z axis
def calculate_2d_coordinates(points):
    if points.shape[1] == 3 and points.shape[0] != 3:
        points = points.T
    plane = _fit_plane(points)
    projections = _project_points_to_plane(points, plane)
    return _get_plane_coordinates(projections, plane)


# projects a point orthogonal onto a line
# the arguments are numpy arrays with two elements with point and point_on_line containing the pixel coordinates of the
# point to project and one point on the line respectively and line containing the direction of the line
# (point B - point A)
def _orth_projection(point, line, point_on_line):
    # normalized direction vector of line
    line_hat = (line / np.linalg.norm(line)).flatten()

    # vector between the line and the normalized direction vector of the line
    line_pos = point_on_line.flatten() - point_on_line.flatten().dot(line_hat)*line_hat

    # projection of the point onto the line
    projection = point.flatten().dot(line_hat)*line_hat + line_pos
    return projection


# Uses least squares to fit a plane through an array of points. The fitted plane is orthogonal to the xy-plane.
def _fit_plane(points: np.ndarray):
    def plane_func(point, a, b, d):
        return -1./b * (a*point[0]+d)

    popt, pcov = curve_fit(plane_func, points, points[1])
    popt = np.insert(popt, 2, 0)
    return popt


def _project_points_to_plane(points: np.ndarray, plane: np.ndarray):
    t = -(plane[0] * points[0] + plane[1] * points[1] + plane[3]) / (plane[0]**2 + plane[1]**2)
    t = np.atleast_2d(t)
    plane = np.atleast_2d(plane[0:3])
    print(points.shape, plane.T.shape, t.shape)
    projection = points + plane.T * t
    return projection


# Transforms the coordinate system of points on a plane orthogonal to the xy-plane from 3D to 2D.
def _get_plane_coordinates(points: np.ndarray, plane: np.ndarray):
    plane_coordinates = np.ndarray((2, points.shape[1]))

    # move the plane so it goes through the origin
    y = points[1]+plane[3]/plane[1]

    # coordinate transformation x -> width
    plane_coordinates[0] = np.sqrt(points[0] ** 2 + y ** 2) / (np.dot([0, 1], plane[0:2]) / np.linalg.norm(plane[0:2]))

    # transform so width is in [0,inf)
    plane_coordinates[0] = plane_coordinates[0] - np.min(plane_coordinates[0])

    plane_coordinates[1] = points[2]
    return plane_coordinates
