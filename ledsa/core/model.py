import numpy as np

# cost function for the led optimization problem
def target_function(params, *args):
    data, mesh = args
    X, Y = mesh
    nx = np.max(X)
    ny = np.max(Y)
    # mask = data > 0.05 * np.max(data)
    data = np.array(data)   # convert to array to allow change of pixel values
    data[data < 0.05 * np.max(data)] = 0
    # l2 = np.sum((data[mask] - led_fit(X, Y, *params)[mask]) ** 2)
    # l2 = np.sqrt(l2) / data[mask].size
    l2 = np.sum((data - led_model(X, Y, *params)) ** 2)
    l2 = np.sqrt(l2) / data.size
    penalty = 0

    x0, y0, dx, dy, A, alpha, wx, wy = params

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


def led_model(x, y, x0, y0, dx, dy, A, alpha, wx, wy):
    nx = x - x0
    ny = y - y0

    r = np.sqrt(nx ** 2 + ny ** 2)

    phi = np.arctan2(ny, nx) + np.pi + alpha

    dr = dx * dy / (np.sqrt((dx * np.cos(phi)) ** 2 + (dy * np.sin(phi)) ** 2))
    dw = wx * wy / (np.sqrt((wx * np.cos(phi)) ** 2 + (wy * np.sin(phi)) ** 2))

    a = A * 0.5 * (1 - np.tanh((r - dr) / dw))

    return a
