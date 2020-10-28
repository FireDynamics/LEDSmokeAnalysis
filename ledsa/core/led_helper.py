from ledsa.core._led_helper_functions import *
from ledsa.core._led_helper_functions_s1 import *
from ledsa.core._led_helper_functions_s2 import *
from ledsa.core._led_helper_functions_s3 import *

import os
import numpy as np
import matplotlib.pyplot as plt

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


# """
# ------------------------------------
# Image ID Functions and Conversions
# ------------------------------------
# """

def get_img_name(img_id):
    infos = load_file('.{}analysis{}image_infos_analysis.csv'.format(sep, sep), ',', 'str',
                      silent=True, atleast_2d=True)
    for i in range(infos.shape[0]):
        if int(infos[i, 0]) == int(img_id):
            return infos[i, 1]
    raise NameError("Could not find an image name to id {}.".format(img_id))


def get_img_id(img_name):
    infos = load_file('.{}analysis{}image_infos_analysis.csv'.format(sep, sep), ',', 'str',
                      silent=True, atleast_2d=True)
    for i in range(infos.shape[0]):
        if infos[i, 1] == img_name:
            return infos[i, 0]
    raise NameError("Could not find an image id for {}.".format(img_name))


def get_last_img_id():
    infos = load_file('.{}analysis{}image_infos_analysis.csv'.format(sep, sep), ',', 'str',
                      silent=True, atleast_2d=True)
    return int(infos[-1, 0])


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
            return int(float(infos[i, 3]))
    raise NameError("Could not find a time to image {}.".format(img_id))


# """
# ------------------------------------
# Outsourced logic - init
# ------------------------------------
# """

def create_needed_directories(config):
    if not os.path.exists('plots'):
        os.mkdir('plots')
        print("Directory plots created ")
    if not os.path.exists('analysis'):
        os.mkdir('analysis')
        print("Directory analysis created ")
    if not os.path.exists('analysis{}channel{}'.format(sep, config['analyse_photo']['channel'])):
        os.mkdir('analysis{}channel{}'.format(sep, config['analyse_photo']['channel']))
        print("Directory analysis{}channel{} created".format(sep, config['analyse_photo']['channel']))


def request_config_parameters(config):
    if config['DEFAULT']['time_img'] == 'None' and \
            config['DEFAULT']['exif_time_infront_real_time'] == 'None':
        config.in_time_img()
        config.save()
    if config['find_search_areas']['reference_img'] == 'None':
        config.in_ref_img()
        config.save()
    if config['DEFAULT']['exif_time_infront_real_time'] == 'None':
        config.in_time_diff_to_img_time()
        config.save()
    if config['DEFAULT']['img_name_string'] == 'None':
        config.in_img_name_string()
        config.save()
    if config['DEFAULT']['first_img'] == 'None':
        config.in_first_img()
        config.save()
    if config['DEFAULT']['last_img'] == 'None':
        config.in_last_img()
        config.save()
    if config['DEFAULT']['num_of_arrays'] == 'None':
        config.in_num_of_arrays()
        config.save()


def generate_image_infos_csv(config, build_experiment_infos=False, build_analysis_infos=False):
    config_switch = []
    if build_experiment_infos:
        config_switch.append('DEFAULT')
    if build_analysis_infos:
        config_switch.append('analyse_photo')
    for build_type in config_switch:
        if config['DEFAULT']['start_time'] == 'None':
            config.get_start_time()
            config.save()
        img_data = build_img_data_string(build_type, config)

        if build_type == 'DEFAULT':
            save_experiment_infos(img_data)
        if build_type == 'analyse_photo':
            save_analysis_infos(img_data)


# """
# ------------------------------------
# Outsourced logic - Step 1
# ------------------------------------
# """

def find_search_areas(image, window_radius=10, skip=10):
    print('finding led search areas')
    led_mask = generate_mask_of_led_areas(image)
    search_areas = find_pos_of_max_col_val_per_area(image, led_mask, skip, window_radius)
    print("\nfound {} leds".format(search_areas.shape[0]))
    return search_areas


def add_search_areas_to_plot(search_areas, ax, config):
    for i in range(search_areas.shape[0]):
        ax.add_patch(plt.Circle((search_areas[i, 2], search_areas[i, 1]),
                                radius=int(config['window_radius']),
                                color='Red', fill=False, alpha=0.25,
                                linewidth=0.1))
        ax.text(search_areas[i, 2] + int(config['window_radius']),
                search_areas[i, 1] + int(config['window_radius']) // 2,
                '{}'.format(search_areas[i, 0]), fontsize=1)


# """
# ------------------------------------
# Outsourced logic - Step 2
# ------------------------------------
# """

def match_leds_to_led_arrays(search_areas, config):
    edge_indices = get_indices_of_outer_leds(config)
    dists_led_arrays_search_areas = calc_dists_between_led_arrays_and_search_areas(edge_indices, search_areas)
    led_arrays = match_leds_to_arrays_with_min_dist(dists_led_arrays_search_areas, edge_indices, config, search_areas)
    return led_arrays


def generate_line_indices_files(line_indices):
    for i in range(len(line_indices)):
        out_file = open('analysis{}line_indices_{:03}.csv'.format(sep, i), 'w')
        for iled in line_indices[i]:
            out_file.write('{}\n'.format(iled))
        out_file.close()


def generate_labeled_led_arrays_plot(line_indices, search_areas):
    """plot the labeled LEDs"""
    for i in range(len(line_indices)):
        plt.scatter(search_areas[line_indices[i], 2],
                    search_areas[line_indices[i], 1],
                    s=0.1, label='led strip {}'.format(i))

    plt.legend()
    plt.savefig('plots{}led_arrays.pdf'.format(sep))


# """
# ------------------------------------
# Outsourced logic - Step 3
# ------------------------------------
# """

def generate_analysis_data(img_filename, search_areas, line_indices, conf, debug=False, debug_led=None):
    data = read_file('{}{}'.format(conf['img_directory'], img_filename),
                     channel=int(conf['channel']))
    window_radius = int(conf['window_radius'])
    img_analysis_data = ''

    if debug:
        fit_res = generate_led_analysis_data(conf, data, debug, debug_led, img_filename, 0, search_areas,
                                             window_radius)
        return fit_res

    for led_array_idx in range(int(conf['num_of_arrays'])):
        print('processing LED array ', led_array_idx, '...')
        for iled in line_indices[led_array_idx]:
            if iled % (int(conf['skip_leds']) + 1) == 0:
                led_analysis_data = generate_led_analysis_data(conf, data, debug, iled, img_filename,
                                                               led_array_idx, search_areas, window_radius)
                img_analysis_data += led_analysis_data
    return img_analysis_data


def create_fit_result_file(img_data, img_id, channel):
    img_infos = load_file('analysis{}image_infos_analysis.csv'.format(sep), dtype='str', delim=',', silent=True,
                          atleast_2d=True)
    root = os.getcwd()
    root = root.split(sep)
    img_filename = get_img_name(img_id)

    save_results_in_file(channel, img_data, img_filename, img_id, img_infos, root)
    
    
def create_imgs_to_process_file():
    image_infos = load_file('.{}analysis{}image_infos_analysis.csv'.format(sep, sep), dtype='str', delim=',',
                            atleast_2d=True)
    img_filenames = image_infos[:, 1]
    out_file = open('images_to_process.csv', 'w')
    for img in img_filenames:
        out_file.write('{}\n'.format(img))
    out_file.close()


# """
# ------------------------------------
# Outsourced logic - Step 3-re
# ------------------------------------
# """

def find_not_analysed_imgs(config):
    image_infos = load_file('.{}analysis{}image_infos_analysis.csv'.format(sep, sep), dtype='str', delim=',',
                            atleast_2d=True)
    all_imgs = image_infos[:, 1]
    processed_img_ids = find_analysed_img_ids(config)
    processed_imgs = np.frompyfunc(get_img_name, 1, 1)(processed_img_ids)
    remaining_imgs = set(all_imgs)-set(processed_imgs)

    save_list_of_remaining_imgs_needed_to_be_processed(remaining_imgs)
