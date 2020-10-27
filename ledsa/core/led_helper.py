import scipy.optimize
import matplotlib.pyplot as plt
from ledsa.core._led_helper_functions import *
from ledsa.core.model import target_function

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
# Outsourced logic - Step 1
# ------------------------------------
# """


def find_search_areas(image, window_radius=10, skip=10):
    im_mean = np.mean(image)
    im_max = np.max(image)
    th = 0.25 * (im_max - im_mean)
    print("mean pixel value:", im_mean)
    print("max pixel value:", im_max)
    im_set = np.zeros_like(image)
    im_set[image > th] = 1

    search_areas_list = []
    led_id = 0

    print('finding led search areas')
    for ix in range(window_radius, image.shape[0] - window_radius, skip):
        for iy in range(window_radius, image.shape[1] - window_radius, skip):
            if im_set[ix, iy] > 0.7:
                max_x, max_y = find_led_pos(image, ix, iy, window_radius)
                search_areas_list.append([led_id, max_x, max_y])
                led_id += 1
                remove_led(im_set, ix, iy, window_radius)

                print('.', end='', flush=True)

    search_areas_array = np.array(search_areas_list)

    print()
    print("found {} leds".format(search_areas_array.shape[0]))

    return search_areas_array


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


# """
# ------------------------------------
# Outsourced logic - Step 3
# ------------------------------------
# """


def process_file(img_filename, search_areas, line_indices, conf, debug=False, debug_led=None):

    data = read_file('{}{}'.format(conf['img_directory'], img_filename),
                     channel=int(conf['channel']))
    window_radius = int(conf['window_radius'])

    img_data = ''

    for led_array_idx in range(int(conf['num_of_arrays'])):
        if not debug:
            print('processing LED array ', led_array_idx, '...')
        for iled in line_indices[led_array_idx]:
            if iled % (int(conf['skip_leds']) + 1) == 0:

                if debug:
                    iled = debug_led

                cx = int(search_areas[iled, 1])
                cy = int(search_areas[iled, 2])

                s = np.index_exp[cx - window_radius:cx + window_radius,
                                 cy - window_radius:cy + window_radius]

                fit_res, mesh = fit_model_to_leds(data[s])
                mean_color_value = np.mean(data[s])
                sum_color_value = np.sum(data[s])

                if debug:
                    return fit_res

                img_data = append_fit_res_to_img_data(cx, cy, fit_res, iled, img_data, led_array_idx, mean_color_value,
                                                      sum_color_value, window_radius)

                if not fit_res.success:  # A > 255 or A < 0:
                    log_warnings(img_filename, fit_res, iled, led_array_idx, data, s, cx, cy, conf)

    return img_data


def fit_model_to_leds(search_area):
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


# """
# ------------------------------------
# Outsourced logic - Step 3-re
# ------------------------------------
# """


def find_not_analysed_imgs(config):
    """searches for the already analysed images and writes into images_to_process.csv the ones
    that are not yet analysed because the analysis was canceled
    """

    # find all images, which should be analysed
    image_infos = load_file('.{}analysis{}image_infos_analysis.csv'.format(sep, sep), dtype='str', delim=',',
                            atleast_2d=True)
    all_imgs = image_infos[:, 1]
    processed_img_ids = find_analysed_img_ids(config)
    processed_imgs = np.vectorize(get_img_name)(processed_img_ids)
    remaining_imgs = set(all_imgs)-set(processed_imgs)

    save_list_of_remaining_imgs_needed_to_be_processed(remaining_imgs)


def create_imgs_to_process():
    image_infos = load_file('.{}analysis{}image_infos_analysis.csv'.format(sep, sep), dtype='str', delim=',',
                            atleast_2d=True)
    img_filenames = image_infos[:, 1]
    out_file = open('images_to_process.csv', 'w')
    for img in img_filenames:
        out_file.write('{}\n'.format(img))
    out_file.close()
