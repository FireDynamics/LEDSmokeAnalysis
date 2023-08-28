from typing import Tuple

import numpy as np


# cost function for the LED optimization problem
def target_function(params: np.ndarray, *args: Tuple) -> float:
    """
    Calculates the cost or discrepancy between the given data and the model predictions based on input parameters.

    :param params: Input parameters for the LED model which include:
        - x0, y0: Center positions.
        - dx, dy: Deviations in the x and y direction.
        - a: Amplitude.
        - alpha: Angle of orientation.
        - wx, wy: Widths of LED model in x and y directions.
    :type params: numpy.ndarray

    :param args: Extra arguments containing:
        - data: Observed or recorded LED data.
        - mesh: Mesh grid values of x and y.
    :type args: Tuple

    :return: The cost value (L2 norm + penalty).
    :rtype: float

    .. note::
        The function computes the L2 norm between the data and model, normalized by data size.
        Penalty terms are introduced to ensure that optimized parameters remain physically reasonable.
    """
    data, mesh = args
    x, y = mesh
    nx = np.max(x)
    ny = np.max(y)
    # mask = data > 0.05 * np.max(data)
    data = np.array(data)  # convert to array to allow change of pixel values
    data[data < 0.05 * np.max(data)] = 0
    # l2 = np.sum((data[mask] - led_fit(x, y, *params)[mask]) ** 2)
    # l2 = np.sqrt(l2) / data[mask].size
    l2 = np.sum((data - led_model(x, y, *params)) ** 2)
    l2 = np.sqrt(l2) / data.size
    penalty = 0

    x0, y0, dx, dy, a, alpha, wx, wy = params

    if x0 < 0 or x0 > nx or y0 < 0 or y0 > ny:
        penalty += 1e3 * np.abs(x0 - nx) + 1e3 * np.abs(y0 - ny)
    if dx < 1 or dy < 1:
        penalty += 1. / (np.abs(dx)) ** 4 + 1. / (np.abs(dy)) ** 4
    w0 = 0.001
    if wx < w0 or wy < w0:
        penalty += np.abs(wx - w0) * 1e6 + np.abs(wy - w0) * 1e6

    if np.abs(alpha) > np.pi / 2:
        penalty += (np.abs(alpha) - np.pi / 2) * 1e6

    return l2 + penalty


def led_model(x: np.ndarray, y: np.ndarray, x0: float, y0: float, dx: float, dy: float, a: float, alpha: float, wx: float, wy: float) -> np.ndarray:
    """
    Defines a mathematical model for an LED based on given parameters.

    :param x: Mesh grid values in x direction.
    :type x: np.ndarray
    :param y: Mesh grid values in y direction.
    :type y: np.ndarray
    :param x0: Center position in the x-direction.
    :type x0: float
    :param y0: Center position in the y-direction.
    :type y0: float
    :param dx: Deviation in the x direction.
    :type dx: float
    :param dy: Deviation in the y direction.
    :type dy: float
    :param a: Amplitude of the LED.
    :type a: float
    :param alpha: Angle of orientation of the LED in radians.
    :type alpha: float
    :param wx: Width of the LED model in x direction.
    :type wx: float
    :param wy: Width of the LED model in y direction.
    :type wy: float
    :return: Evaluated model values based on the input parameters.
    :rtype: np.ndarray

    .. note::
        The function computes the amplitude based on the distance from the center
        and the orientation angle, and then scales it using a tanh function.
    """
    nx = x - x0
    ny = y - y0

    r = np.sqrt(nx ** 2 + ny ** 2)

    phi = np.arctan2(ny, nx) + np.pi + alpha

    dr = dx * dy / (np.sqrt((dx * np.cos(phi)) ** 2 + (dy * np.sin(phi)) ** 2))
    dw = wx * wy / (np.sqrt((wx * np.cos(phi)) ** 2 + (wy * np.sin(phi)) ** 2))

    a = a * 0.5 * (1 - np.tanh((r - dr) / dw))

    return a
