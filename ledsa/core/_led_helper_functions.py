import os
from datetime import timedelta, datetime
import exifread
from PIL import Image

sep = os.path.sep


def calc_experiment_and_real_time(build_type, config, tag, img_number):
    exif = get_exif(config['DEFAULT']['img_directory'] +
                    config['DEFAULT']['img_name_string'].format(int(img_number)), tag)
    if not exif:
        raise ValueError("No EXIF metadata found")

    if f"EXIF {tag}" not in exif:
        raise ValueError("No EXIF time found")
    date, time_meta = exif[f"EXIF {tag}"].values.split(' ')
    date_time_img = get_datetime_from_str(date, time_meta)

    experiment_time = date_time_img - config.get_datetime()
    experiment_time = experiment_time.total_seconds()
    time_diff = config[build_type]['exif_time_infront_real_time'].split('.')
    if len(time_diff) == 1:
        time_diff.append('0')
    time = date_time_img - timedelta(seconds=int(time_diff[0]), milliseconds=int(time_diff[1]))
    return experiment_time, time


def get_exif(filename, tag):
    image = Image.open(filename)
    image.verify()
    with open(filename, 'rb') as f:
        exif = exifread.process_file(f, details=False, stop_tag=tag)
    return exif


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
        largest_number += 9 * 10 ** i
    print(largest_number)
    num_list = [str(ele).zfill(number_string_length) for ele in range(first, largest_number + 1, increment)]
    num_list.extend([str(ele).zfill(number_string_length) for ele in
                     range(increment - (largest_number - int(num_list[-1])), last + 1, increment)])
    return num_list


def build_img_data_string(build_type, config):
    img_data = ''
    img_idx = 1
    first_img = config.getint(build_type, 'first_img')
    last_img = config.getint(build_type, 'last_img')
    img_increment = config.getint(build_type, 'skip_imgs') + 1 if build_type == 'analyse_photo' else 1
    img_number_list = find_img_number_list(first_img, last_img, img_increment)
    for img_number in img_number_list:
        tag = 'DateTimeOriginal'
        experiment_time, time = calc_experiment_and_real_time(build_type, config, tag, img_number)
        img_data += (str(img_idx) + ',' + config[build_type]['img_name_string'].format(int(img_number)) +
                     ',' + time.strftime('%H:%M:%S') + ',' + str(experiment_time) + '\n')
        img_idx += 1
    return img_data


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
