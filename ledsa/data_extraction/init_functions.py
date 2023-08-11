import os
from datetime import timedelta, datetime
from typing import List

from ledsa.core.image_reading import get_exif_entry

from ledsa.core.ConfigData import ConfigData


def create_needed_directories(channels: List[int]) -> None:
    if not os.path.exists('plots'):
        os.mkdir('plots')
        print("Directory plots created ")
    if not os.path.exists('analysis'):
        os.mkdir('analysis')
        print("Directory analysis created ")
    for channel in channels:
        channel_dir = os.path.join('analysis', f'channel{channel}')
        if not os.path.exists(channel_dir):
            os.mkdir(channel_dir)
            print(os.path.relpath(channel_dir))


def request_config_parameters(config: ConfigData) -> None:
    if config['DEFAULT']['img_directory'] == 'None':
        config.in_img_dir()
        config.save()
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
    if config['DEFAULT']['first_img'] == 'None':
        config.in_first_img_experiment()
        config.save()
    if config['DEFAULT']['last_img'] == 'None':
        config.in_last_img_experiment()
        config.save()
    if config['analyse_positions']['num_of_arrays'] == 'None':
        config.in_num_of_arrays()
        config.save()


def generate_image_infos_csv(config: ConfigData, build_experiment_infos=False, build_analysis_infos=False) -> None:
    config_switch = []
    if build_experiment_infos:
        config_switch.append('DEFAULT')
    if build_analysis_infos:
        config_switch.append('analyse_photo')
    for build_type in config_switch:
        if config['DEFAULT']['img_name_string'] == 'None':
            config.in_img_name_string()
            config.save()
        if config['DEFAULT']['start_time'] == 'None':
            config.get_start_time()
            config.save()
        img_data = _build_img_data_string(build_type, config)

        if build_type == 'DEFAULT':
            _save_experiment_infos(img_data)
        if build_type == 'analyse_photo':
            _save_analysis_infos(img_data)


def _calc_experiment_and_real_time(build_type, config, tag, img_number):
    exif_entry = get_exif_entry(config['DEFAULT']['img_directory'] +
                                config['DEFAULT']['img_name_string'].format(int(img_number)), tag)
    date, time_meta = exif_entry.split(' ')
    date_time_img = _get_datetime_from_str(date, time_meta)

    experiment_time = date_time_img - config.get_datetime()
    experiment_time = experiment_time.total_seconds()
    time_diff = config[build_type]['exif_time_infront_real_time'].split('.')
    if len(time_diff) == 1:
        time_diff.append('0')
    time = date_time_img - timedelta(seconds=int(time_diff[0]), milliseconds=int(time_diff[1]))
    return experiment_time, time


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
        largest_number += 9 * 10 ** i
    print(largest_number)
    num_list = [str(ele).zfill(number_string_length) for ele in range(first, largest_number + 1, increment)]
    num_list.extend([str(ele).zfill(number_string_length) for ele in
                     range(increment - (largest_number - int(num_list[-1])), last + 1, increment)])
    return num_list


def _build_img_data_string(build_type, config):
    img_data = ''
    img_idx = 1
    if config['analyse_photo']['first_img'] == 'None':
        config.in_first_img_analysis()
        config.save()
    first_img = config.getint(build_type, 'first_img')

    if config['analyse_photo']['last_img'] == 'None':
        config.in_last_img_analysis()
        config.save()
    last_img = config.getint(build_type, 'last_img')

    img_increment = config.getint(build_type, 'skip_imgs') + 1 if build_type == 'analyse_photo' else 1
    img_number_list = _find_img_number_list(first_img, last_img, img_increment)
    for img_number in img_number_list:
        tag = 'EXIF DateTimeOriginal'
        experiment_time, time = _calc_experiment_and_real_time(build_type, config, tag, img_number)
        img_data += (str(img_idx) + ',' + config[build_type]['img_name_string'].format(int(img_number)) +
                     ',' + time.strftime('%H:%M:%S') + ',' + str(experiment_time) + '\n')
        img_idx += 1
    return img_data


def _save_analysis_infos(img_data):
    file_path = os.path.join('analysis', 'image_infos_analysis.csv')
    out_file = open(file_path, 'w')
    out_file.write("#ID,Name,Time[s],Experiment_Time[s]\n")
    out_file.write(img_data)
    out_file.close()


def _save_experiment_infos(img_data):
    out_file = open('image_infos.csv', 'w')
    out_file.write("#Count,Name,Time[s],Experiment_Time[s]\n")
    out_file.write(img_data)
    out_file.close()
