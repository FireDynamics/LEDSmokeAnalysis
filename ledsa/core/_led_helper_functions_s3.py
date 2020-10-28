import os
import re

import numpy as np
import scipy.optimize
from ledsa.core.model import target_function

sep = os.path.sep


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


def generate_led_analysis_data_string(cx, cy, fit_res, iled, led_array_idx, mean_color_value, sum_color_value,
                                      window_radius):
    x, y, dx, dy, A, alpha, wx, wy = fit_res.x
    im_x = x + cx - window_radius
    im_y = y + cy - window_radius
    led_data = (f'{iled:4d},{led_array_idx:2d},{im_x:10.4e},{im_y:10.4e},{dx:10.4e},{dy:10.4e},{A:10.4e},'
                f'{alpha:10.4e},{wx:10.4e},{wy:10.4e},{fit_res.success:12d},{fit_res.fun:10.4e},'
                f'{fit_res.nfev:9d},{sum_color_value:10.4e},{mean_color_value:10.4e}\n')
    return led_data


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
