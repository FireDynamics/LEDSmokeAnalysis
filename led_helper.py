import numpy as np
import scipy.optimize
import matplotlib.pyplot as plt

"""
------------------------------------
File management
------------------------------------
"""


# should handle all exception for opening files
# when exception is thrown, ask if std conffile should be used or user input
def load_file(filename, delim=' ', dtype='float'):
    try:
        data = np.loadtxt(filename, delimiter=delim, dtype=dtype)
    except OSError as e:
        print('An operation system error occurred while loading {}'.format(filename),
              '. Maybe the file is not there or there is no reading ',
              'permission.\n Error Message: ', e)
        exit(0)
    except Exception as e:
        print('Some error has occured while loading {}'.format(filename),
              '. \n Error Message: ', e)
        exit(0)
    else:
        print('{} successfully loaded.'.format(filename))
    return data


def read_file(filename, channel=0):
    data = plt.imread(filename)
    return data[:, :, channel]


"""
------------------------------------
Input/Output
------------------------------------
"""


def shell_in_ignore_indices():
    return 0


def shell_in_line_edge_indices(config):
    print('The edges of the LED arrays are needed. Please enter the labels of the top most and bottom most LED of each '
          'array. Separate the two labels with a whitespace.')
    labels = str()
    for i in range(int(config['num_of_arrays'])):
        line = input(str(i) + '. array: ')
        labels += '\t    ' + line + '\n'
    config['line_edge_indices'] = '\n' + labels


"""
------------------------------------
Outsourced logic
------------------------------------
"""


def led_fit(x, y, x0, y0, dx, dy, A, alpha, wx, wy, plot=False):
    nx = x - x0
    ny = y - y0

    r = np.sqrt(nx ** 2 + ny ** 2)

    phi = np.arctan2(ny, nx) + np.pi + alpha

    dr = dx * dy / (np.sqrt((dx * np.cos(phi)) ** 2 + (dy * np.sin(phi)) ** 2))
    dw = wx * wy / (np.sqrt((wx * np.cos(phi)) ** 2 + (wy * np.sin(phi)) ** 2))

    a = A * 0.5 * (1 - np.tanh((r - dr) / dw))
    if plot:
        return dr
    return a


def find_leds(search_area):
    def target_function(params, *args):
        data, mesh = args
        X, Y = mesh
        nx = np.max(X)
        ny = np.max(Y)
        mask = data > 0.05 * np.max(data)
        l2 = np.sum((data[mask] - led_fit(X, Y, *params)[mask]) ** 2)
        l2 = np.sqrt(l2) / data[mask].size
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

        # returns 12 every time...
        penalty = 0
        return l2 + penalty

    nx = search_area.shape[0]
    ny = search_area.shape[1]

    center_x = nx // 2
    center_y = ny // 2
    x0 = np.array([center_x, center_y, 2., 2., 200., 1.0, 1.0, 1.0])
    x = np.linspace(0.5, nx - 0.5, nx)
    y = np.linspace(0.5, ny - 0.5, ny)
    mesh = np.meshgrid(x, y)
    res = scipy.optimize.minimize(target_function, x0,
                                  args=(search_area, mesh), method='nelder-mead',
                                  options={'xtol': 1e-8, 'disp': False,
                                           'adaptive': False, 'maxiter': 10000})
    # print(res)
    return res, mesh


def find_search_areas(image, window_radius=10, skip=10):
    im_mean = np.mean(image)
    im_max = np.max(image)
    th = 0.25 * (im_max - im_mean)
    print("mean pixel value:", im_mean)
    print("max pixel value:", im_max)
    im_set = np.zeros_like(image)
    im_set[image > th] = 1

    list_ixy = []
    led_id = 0

    print('finding led search areas')
    for ix in range(window_radius, image.shape[0] - window_radius, skip):
        for iy in range(window_radius, image.shape[1] - window_radius, skip):
            if im_set[ix, iy] > 0.7:
                s_radius = window_radius // 2
                s = np.index_exp[ix - s_radius:ix + s_radius, iy - s_radius:iy + s_radius]
                # print(s, image[s])
                res = np.unravel_index(np.argmax(image[s]), image[s].shape)
                # print(s, res)
                max_x = ix - s_radius + res[0]
                max_y = iy - s_radius + res[1]
                list_ixy.append([led_id, max_x, max_y])
                led_id += 1
                im_set[ix - window_radius:ix + window_radius, iy - window_radius:iy + window_radius] = 0

                print('.', end='', flush=True)

    ixys = np.array(list_ixy)

    print()
    print("found {} leds".format(ixys.shape[0]))

    return ixys


def analyse_position_man(search_areas, config):
    nled = search_areas.shape[0]

    xs = search_areas[:, 2]
    ys = search_areas[:, 1]

    # get indices of LEDs to ignore
    if config['analyse_positions']['ignore_indices'] != 'None':
        ignore_indices = np.array([int(i) for i in config['analyse_positions']['ignore_indices']])
    else:
        ignore_indices = np.array([])

    # get the edges of the LED arrays
    if config['analyse_positions']['line_edge_indices'] == 'None':
        shell_in_line_edge_indices(config['analyse_positions'])
        with open('config.ini', 'w') as configfile:
            config.write(configfile)
    line_edge_indices = config.get2dnparray('analyse_positions', 'line_edge_indices')

    # makes sure that line_edge_indices is a 2d list
    if len(line_edge_indices.shape) == 1:
        line_edge_indices = [line_edge_indices]

    # calculate lengths of the line arrays
    line_distances = np.zeros((nled, len(line_edge_indices)))

    for il in range(len(line_edge_indices)):
        i1 = int(line_edge_indices[il][0])
        i2 = int(line_edge_indices[il][1])

        p1x = xs[i1]
        p1y = ys[i1]
        p2x = xs[i2]
        p2y = ys[i2]

        pd = np.sqrt((p1x - p2x) ** 2 + (p1y - p2y) ** 2)
        d = np.abs(((p2y - p1y) * xs - (p2x - p1x) * ys
                    + p2x * p1y - p2y * p1x) / pd)

        line_distances[:, il] = d

    # construct 2D array for LED indices sorted by line
    line_indices = []
    for il in line_edge_indices:
        line_indices.append([])

    # find for every LED the corresponding array
    for iled in range(nled):

        if iled in ignore_indices:
            continue

        for il_repeat in range(len(line_edge_indices)):
            il = np.argmin(line_distances[iled, :])
            i1 = int(line_edge_indices[il][0])
            i2 = int(line_edge_indices[il][1])

            p1x = xs[i1]
            p1y = ys[i1]
            p2x = xs[i2]
            p2y = ys[i2]

            cx = xs[iled]
            cy = ys[iled]

            d1 = np.sqrt((p1x - cx) ** 2 + (p1y - cy) ** 2)
            d2 = np.sqrt((p2x - cx) ** 2 + (p2y - cy) ** 2)
            d12 = np.sqrt((p1x - p2x) ** 2 + (p1y - p2y) ** 2) + 1e-8

            if d1 < d12 and d2 < d12:
                break

            line_distances[iled, il] *= 2

        line_indices[il].append(iled)
    return line_indices


def process_file(img_filename, search_areas, line_indices, conf):
    # print(search_areas)
    # print(len(search_areas))

    data = read_file('{}{}'.format(conf['img_directory'], img_filename),
                     channel=int(conf['channel']))
    window_radius = int(conf['window_radius'])

    img_data = ''

    for iline in range(int(conf['num_of_arrays'])):
        print('processing LED array ', iline, '...')
        for iled in line_indices[iline]:
            if iled % (int(conf['skip_leds']) + 1) == 0:
                cx = int(search_areas[iled, 1])
                cy = int(search_areas[iled, 2])

                s = np.index_exp[cx - window_radius:cx + window_radius,
                                 cy - window_radius:cy + window_radius]

                fit_res, mesh = find_leds(data[s])

                x, y, dx, dy, A, alpha, wx, wy = fit_res.x

                im_x = x + cx - window_radius
                im_y = y + cy - window_radius

                line_number = iline

                img_data +=('{:4d},{:2d},{:10.4e},{:10.4e},{:10.4e},{:10.4e},{:10.4e},{:10.4e},{:10.4e},{:10.4e},'
                            '{:12d},{:10.4e},{:9d}\n'.format(iled, line_number, im_x, im_y, dx, dy, A, alpha, wx, wy,
                                                             fit_res.success, fit_res.fun, fit_res.nfev))

    return img_data


'''
if __name__ == 'main':

    filename = 'scn_experiments/IMG_7508.JPG'
    data = read_file(filename, channel=0)
    window_radius=10
    xys = find_leds(data, window_radius=window_radius)

    np.savetxt('{}.led_pos.csv'.format(filename), xys,
               header='pixel position x, y', fmt='%d')

    fig = plt.figure(dpi=900)
    ax = plt.gca()

    for i in range(xys.shape[0]):
        ax.add_patch(plt.Circle((xys[i,1], xys[i,0]), radius=window_radius,
                                color='Red', fill=False, alpha=0.5,
                                linewidth=0.1))


    plt.imshow(data, cmap='Greys')
    plt.colorbar()
    plt.savefig('{}.led_pos.plot.pdf'.format(filename))
'''
