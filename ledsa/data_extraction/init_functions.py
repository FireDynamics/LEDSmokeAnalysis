import os
from datetime import timedelta
from typing import List

from ledsa.core.ConfigData import ConfigData
from ledsa.core.image_integrity import CorruptedImageError, report_corrupted_images
from ledsa.core.image_reading import get_exif_datetime


def create_needed_directories(channels: List[int]) -> None:
    """
    Create required directories for storing plots and analysis results.

    :param channels: List of channel indices to create subdirectories under 'analysis'.
    :type channels: List[int]
    """
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
    """
    Prompt and update config parameters if they are not set.

    :param config: Configuration data object that contains settings.
    :type config: ConfigData
    """
    if config['DEFAULT']['img_directory'] == 'None':
        config.in_img_dir()
        config.save()
    if config['DEFAULT']['img_name_string'] == 'None':
        config.in_img_name_string()
        config.save()
    if config.get('DEFAULT', 'time_ref_img_id') == 'None' and \
            config.get('DEFAULT', 'time_ref_img_file') == 'None' and \
            config['DEFAULT']['exif_time_infront_real_time'] == 'None':
        config.in_time_ref_img_id()
        config.save()
    if config.get('DEFAULT', 'time_ref_img_id')== 'None' and \
            config.get('DEFAULT', 'time_ref_img_file') == 'None' and \
            config['DEFAULT']['exif_time_infront_real_time'] == 'None':
        config.in_time_ref_img_file()
        config.save()
    if config['DEFAULT']['exif_time_infront_real_time'] == 'None':
        config.in_time_diff_to_img_time()
        config.save()
    if config['find_search_areas']['ref_img_id'] == 'None':
        config.in_ref_img_id()
        config.save()
    if config['find_search_areas']['max_num_leds'] == 'None':
        config.in_max_num_leds()
        config.save()
    if config['DEFAULT']['first_img_experiment_id'] == 'None':
        config.in_first_img_experiment_id()
        config.save()
    if config['DEFAULT']['last_img_experiment_id'] == 'None':
        config.in_last_img_experiment_id()
        config.save()
    if config['analyse_positions']['num_arrays'] == 'None':
        config.in_num_arrays()
        config.save()


def generate_image_infos_csv(config: ConfigData, build_experiment_infos=False, build_analysis_infos=False) -> None:
    """
    Generate CSV files with image information for experiment and/or analysis.
    Contains image name, exif time and experiment time

    :param config: Configuration data object.
    :type config: ConfigData
    :param build_experiment_infos: Whether to build CSV for experiment info.
    :type build_experiment_infos: bool
    :param build_analysis_infos: Whether to build CSV for analysis info.
    :type build_analysis_infos: bool
    :raises RuntimeError: If the first image of the experiment or none of the images of the configured ID
                          range could be read.
    """
    config_switch = []
    if build_experiment_infos:
        config_switch.append('DEFAULT')
    if build_analysis_infos:
        config_switch.append('analyse_photo')
    for build_type in config_switch:
        if config['DEFAULT']['start_time'] == 'None':
            try:
                config.get_start_time()
            except CorruptedImageError as error:
                # Every experiment time is measured from the first image, so it cannot be skipped
                report_corrupted_images([error], 'reading of the experiment start time')
                raise RuntimeError(f'The start time of the experiment cannot be read, because '
                                   f'{error.img_filename} could not be read. Set start_time in config.ini '
                                   f'manually or point first_img_experiment_id at a readable image.') from error
            config.save()
        img_data = _build_img_data_string(build_type, config)

        if build_type == 'DEFAULT':
            _save_experiment_infos(img_data)
        if build_type == 'analyse_photo':
            _save_analysis_infos(img_data)


def _calc_experiment_and_real_time(build_type: str, config: ConfigData, tag: str, img_number: int) -> None:
    """
    Calculate experiment and real-time based on image metadata and config settings.

    :param build_type: Type of build process (either 'DEFAULT' or 'analyse_photo').
    :type build_type: str
    :param config: Configuration data object.
    :type config: ConfigData
    :param tag: Metadata tag to extract time information from image.
    :type tag: str
    :param img_number: Image ID where the time should be read from.
    :type img_number: int
    :return: Tuple containing experiment time and real time.
    :rtype: tuple
    """
    img_path = os.path.join(config['DEFAULT']['img_directory'],
                            config['DEFAULT']['img_name_string'].format(int(img_number)))
    date_time_img = get_exif_datetime(img_path, tag)

    experiment_time = date_time_img - config.get_datetime()
    experiment_time = experiment_time.total_seconds()
    time_diff = config.getfloat(build_type, 'exif_time_infront_real_time')
    time = date_time_img - timedelta(seconds=time_diff)
    return experiment_time, time


def _find_img_number_list(first: int, last: int, increment: int, number_string_length=4) -> List[str]:
    """
    Generate a list of image numbers with specified increment and padding.

    :param first: First image ID.
    :type first: int
    :param last: Last image ID.
    :type last: int
    :param increment: Increment value for image ID range.
    :type increment: int
    :param number_string_length: Number of digits in the image ID (default is 4).
    :type number_string_length: int
    :return: List of image IDs as strings.
    :rtype: List[str]
    """
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


def _build_img_data_string(build_type: str, config: ConfigData) -> str:
    """
    Construct image data string for CSV generation.

    :param build_type: Type of build process (either 'DEFAULT' or 'analyse_photo').
    :type build_type: str
    :param config: Configuration data object.
    :type config: ConfigData
    :return: Formatted image data string.
    :rtype: str
    """
    img_data = ''
    img_idx = 1
    corrupted_images = []
    if config['analyse_photo']['first_img_analysis_id'] == 'None':
        config.in_first_img_analysis_id()
        config.save()
    first_img_id = config.getint(build_type, 'first_img_experiment_id')

    if config['analyse_photo']['last_img_analysis_id'] == 'None':
        config.in_last_img_analysis_id()
        config.save()
    last_img_id = config.getint(build_type, 'last_img_experiment_id' if build_type == 'DEFAULT' else 'last_img_analysis_id')

    img_increment = config.getint(build_type, 'num_skip_imgs') + 1 if build_type == 'analyse_photo' else 1
    img_id_list = _find_img_number_list(first_img_id, last_img_id, img_increment)
    for img_id in img_id_list:
        tag = 'DateTimeOriginal'
        try:
            experiment_time, time = _calc_experiment_and_real_time(build_type, config, tag, img_id)
        except CorruptedImageError as error:
            # Leaving the image out here already excludes it from every following step, while the IDs of the
            # readable images stay a gapless sequence, which the result file headers rely on.
            corrupted_images.append(error)
            continue
        img_data += (str(img_idx) + ',' + config[build_type]['img_name_string'].format(int(img_id)) +
                     ',' + time.strftime('%H:%M:%S.%f')[:-3] + ',' + str(experiment_time) + '\n')
        img_idx += 1
    report_corrupted_images(corrupted_images, 'reading of the image metadata')
    if img_idx == 1:
        raise RuntimeError(f'None of the {len(img_id_list)} images between the IDs {first_img_id} and '
                           f'{last_img_id} could be read. Check img_directory and img_name_string in config.ini.')
    return img_data


def _save_analysis_infos(img_data: str) -> None:
    """
    Save the image data for analysis to a CSV file.

    :param img_data: Formatted string containing image data.
    :type img_data: str
    """
    file_path = os.path.join('analysis', 'image_infos_analysis.csv')
    out_file = open(file_path, 'w', encoding='utf-8')
    out_file.write("#ID,Name,Time[s],Experiment_Time[s]\n")
    out_file.write(img_data)
    out_file.close()


def _save_experiment_infos(img_data: str) -> None:
    """
    Save the image data for experiment to a CSV file.

    :param img_data: Formatted string containing image data.
    :type img_data: str
    """
    out_file = open('image_infos.csv', 'w', encoding='utf-8')
    out_file.write("#Count,Name,Time[s],Experiment_Time[s]\n")
    out_file.write(img_data)
    out_file.close()
