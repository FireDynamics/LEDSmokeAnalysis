import numpy as np
import scipy.optimize
import matplotlib.pyplot as plt
import os
from PIL import Image
from PIL.ExifTags import TAGS
from datetime import datetime, timedelta

# os path separator
sep = os.path.sep

# """
# ------------------------------------
# File management
# ------------------------------------
# """


# should handle all exception for opening files
# when exception is thrown, ask if std conf-file should be used or user input
def load_file(filename, delim=' ', dtype='float', atleast_2d=False, silent=False):
    try:
        data = np.loadtxt(filename, delimiter=delim, dtype=dtype)
    except OSError as e:
        if not silent:
            print('An operation system error occurred while loading {}.'.format(filename),
                  'Maybe the file does not exist or there is no reading ',
                  'permission.\n Error Message: ', e)
        raise
    except Exception as e:
        if not silent:
            print('Some error has occurred while loading {}'.format(filename),
                  '. \n Error Message: ', e)
        raise
    else:
        if not silent:
            print('{} successfully loaded.'.format(filename))
    if atleast_2d:
        return np.atleast_2d(data)
    return np.atleast_1d(data)


def read_file(filename, channel=0):
    data = plt.imread(filename)
    return data[:, :, channel]


def get_img_data(config, build_experiment_infos=False, build_analysis_infos=False):
    config_switch = []
    if build_experiment_infos:
        config_switch.append('DEFAULT')
    if build_analysis_infos:
        config_switch.append('analyse_photo')
    img_data = ''
    for build_type in config_switch:
        if config['DEFAULT']['start_time'] == 'None':
            config.get_start_time()
            config.save()
        img_idx = 1
        first_img = config.getint(build_type, 'first_img')
        last_img = config.getint(build_type, 'last_img')
        img_increment = config.getint(build_type, 'skip_imgs') + 1 if build_type == 'analyse_photo' else 1
        img_number_list = _find_img_number_list(first_img, last_img, img_increment)

        for img_number in img_number_list:
    
            # get time from exif data
            exif = _get_exif(config['DEFAULT']['img_directory'] +
                             config['DEFAULT']['img_name_string'].format(int(img_number)))
            if not exif:
                raise ValueError("No EXIF metadata found")
    
            for (idx, tag) in TAGS.items():
                if tag == 'DateTimeOriginal':
                    if idx not in exif:
                        raise ValueError("No EXIF time found")
                    date, time_meta = exif[idx].split(' ')
                    date_time_img = _get_datetime_from_str(date, time_meta)
    
                    # calculate the experiment time and real time
                    experiment_time = date_time_img - config.get_datetime()
                    experiment_time = experiment_time.total_seconds()
                    time_diff = config[build_type]['exif_time_infront_real_time'].split('.')
                    time = date_time_img - timedelta(seconds=int(time_diff[0]), milliseconds=int(time_diff[1]))
                    img_data += (str(img_idx) + ',' + config[build_type]['img_name_string'].format(int(img_number)) + ',' +
                                 time.strftime('%H:%M:%S') + ',' + str(experiment_time) + '\n')
                    img_idx += 1
    return img_data


def get_img_name(img_id):
    infos = load_file('.{}analysis{}image_infos_analysis.csv'.format(sep, sep), ',', 'str', silent=True)
    for i in range(infos.shape[0]):
        if int(infos[i, 0]) == int(img_id):
            return infos[i, 1]
    raise NameError("Could not find an image name to id {}.".format(img_id))


def get_img_id(img_name):
    infos = load_file('.{}analysis{}image_infos_analysis.csv'.format(sep, sep), ',', 'str', silent=True)
    for i in range(infos.shape[0]):
        if infos[i, 1] == img_name:
            return infos[i, 0]
    raise NameError("Could not find an image id for {}.".format(img_name))


def get_last_img_id():
    infos = load_file('.{}analysis{}image_infos_analysis.csv'.format(sep, sep), ',', 'str', silent=True)
    return int(infos[-1, 0])

# """
# ------------------------------------
# Input/Output
# ------------------------------------
# """


def shell_in_ignore_indices():
    return 0


# switch to ledsa_conf
def shell_in_line_edge_indices(config):
    print('The edges of the LED arrays are needed. Please enter the labels of the top most and bottom most LED of each '
          'array. Separate the two labels with a whitespace.')
    labels = str()
    for i in range(int(config['num_of_arrays'])):
        line = input(str(i) + '. array: ')
        labels += '\t    ' + line + '\n'
    config['line_edge_indices'] = '\n' + labels


# switch to Lib/logging at some point
def log_warnings(*argv):
    if not os.path.exists('.{}logfiles'.format(sep)):
        os.makedirs('.{}logfiles'.format(sep))
    logfile = open('.{}logfiles{}warnings.log'.format(sep, sep), 'a')
    for info in argv:
        logfile.write(str(info) + ' ')
    logfile.write('\n')
    logfile.close()


# """
# ------------------------------------
#
# ------------------------------------
# """


def get_img_id_from_time(time):
    infos = load_file('.{}analysis{}image_infos_analysis.csv'.format(sep, sep), ',', 'str', silent=True)
    for i in range(infos.shape[0]):
        if float(infos[i, 3]) == time:
            return int(infos[i, 0])
    raise NameError("Could not find an image id at {}s.".format(time))


def get_time_from_img_id(img_id):
    infos = load_file('.{}analysis{}image_infos_analysis.csv'.format(sep, sep), ',', 'str', silent=True)
    for i in range(infos.shape[0]):
        if float(infos[i, 0]) == img_id:
            return int(infos[i, 3])
    raise NameError("Could not find a time to image {}.".format(img_id))


# """
# ------------------------------------
# Outsourced logic
# ------------------------------------
# """


def led_fit(x, y, x0, y0, dx, dy, A, alpha, wx, wy):
    nx = x - x0
    ny = y - y0

    r = np.sqrt(nx ** 2 + ny ** 2)

    phi = np.arctan2(ny, nx) + np.pi + alpha

    dr = dx * dy / (np.sqrt((dx * np.cos(phi)) ** 2 + (dy * np.sin(phi)) ** 2))
    dw = wx * wy / (np.sqrt((wx * np.cos(phi)) ** 2 + (wy * np.sin(phi)) ** 2))

    a = A * 0.5 * (1 - np.tanh((r - dr) / dw))

    return a


def find_leds(search_area):
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


# calculates for every led, to which array it belongs
def analyse_position_man(search_areas, config):
    num_leds = search_areas.shape[0]

    xs = search_areas[:, 2]
    ys = search_areas[:, 1]

    # get indices of LEDs to ignore
    if config['analyse_positions']['ignore_indices'] != 'None':
        ignore_indices = np.array([int(i) for i in config['analyse_positions']['ignore_indices'].split(' ')])
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
    line_distances = np.zeros((num_leds, len(line_edge_indices)))

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
    for iled in range(num_leds):

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


def process_file(img_filename, search_areas, line_indices, conf, debug=False, debug_led=None):

    data = read_file('{}{}'.format(conf['img_directory'], img_filename),
                     channel=int(conf['channel']))
    window_radius = int(conf['window_radius'])

    img_data = ''

    for iline in range(int(conf['num_of_arrays'])):
        if not debug:
            print('processing LED array ', iline, '...')
        for iled in line_indices[iline]:
            if iled % (int(conf['skip_leds']) + 1) == 0:

                if debug:
                    iled = debug_led

                cx = int(search_areas[iled, 1])
                cy = int(search_areas[iled, 2])

                s = np.index_exp[cx - window_radius:cx + window_radius,
                                 cy - window_radius:cy + window_radius]

                fit_res, mesh = find_leds(data[s])
                mean_color_value = np.mean(data[s])
                sum_color_value = np.sum(data[s])

                if debug:
                    return fit_res

                x, y, dx, dy, A, alpha, wx, wy = fit_res.x

                im_x = x + cx - window_radius
                im_y = y + cy - window_radius

                line_number = iline

                led_data = (f'{iled:4d},{line_number:2d},{im_x:10.4e},{im_y:10.4e},{dx:10.4e},{dy:10.4e},{A:10.4e},'
                            f'{alpha:10.4e},{wx:10.4e},{wy:10.4e},{fit_res.success:12d},{fit_res.fun:10.4e},'
                            f'{fit_res.nfev:9d},{sum_color_value:10.4e},{mean_color_value:10.4e}')
                img_data += led_data + '\n'
                img_file_path = conf['img_directory'] + img_filename

                if not fit_res.success:  # A > 255 or A < 0:
                    log_warnings('Irregularities while fitting:\n    ',
                                 img_file_path, iled, line_number, ' '.join(np.array_str(fit_res.x).split()).
                                 replace('[ ', '[').replace(' ]', ']').replace(' ', ','), fit_res.success, fit_res.fun,
                                 fit_res.nfev, data[s].shape[0], data[s].shape[1], im_x, im_y, window_radius, cx, cy,
                                 conf['channel'])

    return img_data


def find_calculated_imgs(config):
    """searches for the already analysed images and writes into images_to_process.csv the ones
    that are not yet analysed because the analysis was canceled
    """
    import re

    # find all images, which should be analysed
    image_infos = load_file('.{}analysis{}image_infos_analysis.csv'.format(sep, sep), dtype='str', delim=',')
    all_imgs = image_infos[:, 1]
    processed_imgs = []

    # find the images, which are already analysed
    directory_content = os.listdir('.{}analysis{}channel{}'.format(sep, sep, config['channel']))

    # create and compare the two sets
    for file_name in directory_content:
        img = re.search(r"([0-9]+)_led_positions.csv", file_name)
        if img is not None:
            processed_imgs.append(get_img_name(int(img.group(1))))
    remaining_imgs = set(all_imgs)-set(processed_imgs)

    # save the result
    out_file = open('images_to_process.csv', 'w')
    for i in list(remaining_imgs):
        out_file.write('{}\n'.format(i))
    out_file.close()


def create_img_infos_analysis(config):

    # create image info file for all images which are analysed
    img_data = get_img_data(config, build_analysis_infos=True)
    out_file = open('.{}analysis{}image_infos_analysis.csv'.format(sep, sep), 'w')
    out_file.write("#ID,Name,Time,Experiment_Time\n")
    out_file.write(img_data)


def create_imgs_to_process():
    image_infos = load_file('.{}analysis{}image_infos_analysis.csv'.format(sep, sep), dtype='str', delim=',',
                            atleast_2d=True)
    img_filenames = image_infos[:, 1]
    out_file = open('images_to_process.csv', 'w')
    for img in img_filenames:
        out_file.write('{}\n'.format(img))
    out_file.close()


# """
# ------------------------------------
# private functions
# ------------------------------------
# """


def _get_exif(filename):
    image = Image.open(filename)
    image.verify()
    return image._getexif()


def _get_datetime_from_str(date, time):
    if date.find(":") != -1:
        date_time = datetime.strptime(date + ' ' + time, '%Y:%m:%d %H:%M:%S')
    else:
        date_time = datetime.strptime(date + ' ' + time, '%d.%m.%Y %H:%M:%S')
    return date_time


def _find_img_number_list(first, last, increment, number_string_length=4):
    if last >= first:
        return [str(ele).zfill(number_string_length) for ele in range(first, last + 1, increment)]

    largest_number = 0
    for i in range(number_string_length):
        largest_number += 9*10**i
    print(largest_number)
    num_list = [str(ele).zfill(number_string_length) for ele in range(first, largest_number+1, increment)]
    num_list.extend([str(ele).zfill(number_string_length) for ele in
                     range(increment-(largest_number-int(num_list[-1])), last+1, increment)])
    return num_list


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

    return l2 + penalty
