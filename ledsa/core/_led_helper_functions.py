import os
from datetime import timedelta, datetime
import re

import numpy
import numpy as np
import scipy.optimize
from PIL import Image
from PIL.ExifTags import TAGS

# os path separator
from ledsa.core.model import target_function

sep = os.path.sep


def calc_experiment_and_real_time(build_type, config, idx, img_number):
    exif = get_exif(config['DEFAULT']['img_directory'] +
                    config['DEFAULT']['img_name_string'].format(int(img_number)))
    if not exif:
        raise ValueError("No EXIF metadata found")

    if idx not in exif:
        raise ValueError("No EXIF time found")
    date, time_meta = exif[idx].split(' ')
    date_time_img = get_datetime_from_str(date, time_meta)

    experiment_time = date_time_img - config.get_datetime()
    experiment_time = experiment_time.total_seconds()
    time_diff = config[build_type]['exif_time_infront_real_time'].split('.')
    if len(time_diff) == 1:
        time_diff.append('0')
    time = date_time_img - timedelta(seconds=int(time_diff[0]), milliseconds=int(time_diff[1]))
    return experiment_time, time


def get_exif(filename):
    image = Image.open(filename)
    image.verify()
    return image._getexif()


def get_datetime_from_str(date, time):
    if date.find(":") != -1:
        date_time = datetime.strptime(date + ' ' + time, '%Y:%m:%d %H:%M:%S')
    else:
        date_time = datetime.strptime(date + ' ' + time, '%d.%m.%Y %H:%M:%S')
    return date_time


def find_img_number_list(first, last, increment, number_string_length=4):
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


def remove_led_from_mask(im_set, ix, iy, window_radius):
    im_set[ix - window_radius:ix + window_radius, iy - window_radius:iy + window_radius] = 0


def find_led_pos(image, ix, iy, window_radius):
    s_radius = window_radius // 2
    s = np.index_exp[ix - s_radius:ix + s_radius, iy - s_radius:iy + s_radius]
    res = np.unravel_index(np.argmax(image[s]), image[s].shape)
    max_x = ix - s_radius + res[0]
    max_y = iy - s_radius + res[1]
    return max_x, max_y


def match_leds_to_arrays_with_min_dist(dists_led_arrays_search_areas, edge_indices, config, search_areas):
    ignore_indices = get_indices_of_ignored_leds(config)

    xs = search_areas[:, 1]
    ys = search_areas[:, 2]

    num_leds = search_areas.shape[0]

    # construct 2D array for LED indices sorted by line
    led_arrays = []
    for edge_idx in edge_indices:
        led_arrays.append([])

    for iled in range(num_leds):
        if iled in ignore_indices:
            continue

        idx_nearest_array = np.argmin(dists_led_arrays_search_areas[iled, :])
        # TODO: ask Lukas for need of following code

        # for il_repeat in range(len(edge_indices)):
        #     i1 = edge_indices[idx_nearest_array][0]
        #     i2 = edge_indices[idx_nearest_array][1]
        #
        #     x_outer_led1 = xs[i1]
        #     y_outer_led1 = ys[i1]
        #     x_outer_led2 = xs[i2]
        #     y_outer_led2 = ys[i2]
        #
        #     x_led = xs[iled]
        #     y_led = ys[iled]
        #
        #     dist_led_outer_led1 = np.sqrt((x_outer_led1 - x_led) ** 2 + (y_outer_led1 - y_led) ** 2)
        #     dist_led_outer_led2 = np.sqrt((x_outer_led2 - x_led) ** 2 + (y_outer_led2 - y_led) ** 2)
        #     dist_outer_leds = np.sqrt((x_outer_led1 - x_outer_led2) ** 2 + (y_outer_led1 - y_outer_led2) ** 2) + 1e-8
        #
        #     if dist_led_outer_led1 < dist_outer_leds and dist_led_outer_led2 < dist_outer_leds:
        #         break
        #
        #     dists_led_arrays_search_areas[iled, idx_nearest_array] *= 2

        led_arrays[idx_nearest_array].append(iled)
    return led_arrays


def calc_dists_between_led_arrays_and_search_areas(line_edge_indices, search_areas):
    distances_led_arrays_search_areas = np.zeros((search_areas.shape[0], len(line_edge_indices)))

    xs = search_areas[:, 1]
    ys = search_areas[:, 2]

    for line_edge_idx in range(len(line_edge_indices)):
        i1 = line_edge_indices[line_edge_idx][0]
        i2 = line_edge_indices[line_edge_idx][1]

        p1x = xs[i1]
        p1y = ys[i1]
        p2x = xs[i2]
        p2y = ys[i2]

        pd = np.sqrt((p1x - p2x) ** 2 + (p1y - p2y) ** 2)
        d = np.abs(((p2y - p1y) * xs - (p2x - p1x) * ys
                    + p2x * p1y - p2y * p1x) / pd)

        distances_led_arrays_search_areas[:, line_edge_idx] = d
    return distances_led_arrays_search_areas


def get_indices_of_outer_leds(config):
    if config['analyse_positions']['line_edge_indices'] == 'None':
        config.in_line_edge_indices()
        with open('config.ini', 'w') as configfile:
            config.write(configfile)
    line_edge_indices = config.get2dnparray('analyse_positions', 'line_edge_indices')
    # makes sure that line_edge_indices is a 2d list
    if len(line_edge_indices.shape) == 1:
        line_edge_indices = [line_edge_indices]
    return line_edge_indices


def get_indices_of_ignored_leds(config):
    if config['analyse_positions']['ignore_indices'] != 'None':
        ignore_indices = np.array([int(i) for i in config['analyse_positions']['ignore_indices'].split(' ')])
    else:
        ignore_indices = np.array([])
    return ignore_indices


# switch to Lib/logging at some point
def log_warnings(img_filename, fit_res, iled, line_number, data, s, cx, cy, conf):
    res = ' '.join(np.array_str(fit_res.x).split()).replace('[ ', '[').replace(' ]', ']').replace(' ', ',')
    img_file_path = conf['img_directory'] + img_filename
    window_radius = int(conf['window_radius'])

    x, y, _ = fit_res.x

    im_x = x + cx - window_radius
    im_y = y + cy - window_radius

    log = f'Irregularities while fitting:\n    {img_file_path} {iled} {line_number} {res} {fit_res.success} ' \
          f'{fit_res.fun} {fit_res.nfev} {data[s].shape[0]} {data[s].shape[1]} {im_x} {im_y} {window_radius} {cx} ' \
          f'{cy} {conf["channel"]}'
    if not os.path.exists('.{}logfiles'.format(sep)):
        os.makedirs('.{}logfiles'.format(sep))
    logfile = open('.{}logfiles{}warnings.log'.format(sep, sep), 'a')
    logfile.write(log)
    logfile.write('\n')
    logfile.close()


def build_img_data_string(build_type, config):
    img_data = ''
    img_idx = 1
    first_img = config.getint(build_type, 'first_img')
    last_img = config.getint(build_type, 'last_img')
    img_increment = config.getint(build_type, 'skip_imgs') + 1 if build_type == 'analyse_photo' else 1
    img_number_list = find_img_number_list(first_img, last_img, img_increment)
    for img_number in img_number_list:
        for (idx, tag) in TAGS.items():
            if tag == 'DateTimeOriginal':
                experiment_time, time = calc_experiment_and_real_time(build_type, config, idx, img_number)
                img_data += (str(img_idx) + ',' + config[build_type]['img_name_string'].format(int(img_number)) +
                             ',' + time.strftime('%H:%M:%S') + ',' + str(experiment_time) + '\n')
                img_idx += 1
    return img_data


def generate_led_analysis_data_string(cx, cy, fit_res, iled, led_array_idx, mean_color_value, sum_color_value,
                                      window_radius):
    x, y, dx, dy, A, alpha, wx, wy = fit_res.x
    im_x = x + cx - window_radius
    im_y = y + cy - window_radius
    led_data = (f'{iled:4d},{led_array_idx:2d},{im_x:10.4e},{im_y:10.4e},{dx:10.4e},{dy:10.4e},{A:10.4e},'
                f'{alpha:10.4e},{wx:10.4e},{wy:10.4e},{fit_res.success:12d},{fit_res.fun:10.4e},'
                f'{fit_res.nfev:9d},{sum_color_value:10.4e},{mean_color_value:10.4e}\n')
    return led_data


def save_analysis_infos(img_data):
    out_file = open('.{}analysis{}image_infos_analysis.csv'.format(sep, sep), 'w')
    out_file.write("#ID,Name,Time[s],Experiment_Time[s]\n")
    out_file.write(img_data)
    out_file.close()


def save_experiment_infos(img_data):
    out_file = open('image_infos.csv', 'w')
    out_file.write("#Count,Name,Time[s],Experiment_Time[s]\n")
    out_file.write(img_data)
    out_file.close()


def find_analysed_img_ids(config):
    processed_imgs = []
    directory_content = os.listdir('.{}analysis{}channel{}'.format(sep, sep, config['channel']))
    for file_name in directory_content:
        img = re.search(r"([0-9]+)_led_positions.csv", file_name)
        if img is not None:
            processed_imgs.append(int(img.group(1)))
    return processed_imgs


def save_list_of_remaining_imgs_needed_to_be_processed(remaining_imgs):
    out_file = open('images_to_process.csv', 'w')
    for i in list(remaining_imgs):
        out_file.write('{}\n'.format(i))
    out_file.close()


def save_results_in_file(channel, img_data, img_filename, img_id, img_infos, root):
    out_file = open('analysis{}channel{}{}{}_led_positions.csv'.format(sep, channel, sep, img_id), 'w')
    out_file.write(f'# image root = {root[-1]}, photo file name = {img_filename}, ')
    out_file.write(f"channel = {channel}, ")
    out_file.write(f"time[s] = {img_infos[int(img_id) - 1][3]}\n")
    out_file.write("# id,         line,   x,         y,        dx,        dy,")
    out_file.write("         A,     alpha,        wx,        wy, fit_success,")
    out_file.write("   fit_fun, fit_nfev // all spatial quantities in pixel coordinates\n")
    out_file.write(img_data)
    out_file.close()


def find_pos_of_max_col_val_per_area(image, led_mask, skip, window_radius):
    search_areas_list = []
    led_id = 0
    for ix in range(window_radius, image.shape[0] - window_radius, skip):
        for iy in range(window_radius, image.shape[1] - window_radius, skip):
            if led_mask[ix, iy] != 0:
                max_x, max_y = find_led_pos(image, ix, iy, window_radius)
                search_areas_list.append([led_id, max_x, max_y])
                led_id += 1
                remove_led_from_mask(led_mask, ix, iy, window_radius)

                print('.', end='', flush=True)
    search_areas_array = np.array(search_areas_list)
    return search_areas_array


def generate_mask_of_led_areas(image):
    im_mean = np.mean(image)
    im_max = np.max(image)
    th = 0.25 * (im_max - im_mean)
    print("mean pixel value:", im_mean)
    print("max pixel value:", im_max)
    im_set = np.zeros_like(image)
    im_set[image > th] = 1
    return im_set


def generate_led_analysis_data(conf, data, debug, iled, img_filename, led_array_idx, search_areas,
                               window_radius):
    center_sa_x = int(search_areas[iled, 1])
    center_sa_y = int(search_areas[iled, 2])
    search_area = np.index_exp[center_sa_x - window_radius:center_sa_x + window_radius,
                               center_sa_y - window_radius:center_sa_y + window_radius]

    fit_res, mesh = fit_model_to_led(data[search_area])
    if debug:
        return fit_res
    mean_color_value = np.mean(data[search_area])
    sum_color_value = np.sum(data[search_area])
    img_data = generate_led_analysis_data_string(center_sa_x, center_sa_y, fit_res, iled, led_array_idx,
                                                 mean_color_value,
                                                 sum_color_value, window_radius)

    if not fit_res.success:  # A > 255 or A < 0:
        log_warnings(img_filename, fit_res, iled, led_array_idx, data, search_area, center_sa_x, center_sa_y, conf)
    return img_data


def fit_model_to_led(search_area):
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