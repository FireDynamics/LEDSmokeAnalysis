import numpy as np

from ledsa.core.file_handling import sep, read_table


def get_img_name(img_id: int) -> np.ndarray:
    infos = read_table('.{}analysis{}image_infos_analysis.csv'.format(sep, sep), ',', 'str',
                       silent=True, atleast_2d=True)
    for i in range(infos.shape[0]):
        if int(infos[i, 0]) == int(img_id):
            return infos[i, 1]
    raise NameError("Could not find an image name to id {}.".format(img_id))


def get_img_id(img_name: str) -> int:
    infos = read_table('.{}analysis{}image_infos_analysis.csv'.format(sep, sep), ',', 'str',
                       silent=True, atleast_2d=True)
    for i in range(infos.shape[0]):
        if infos[i, 1] == img_name:
            return infos[i, 0]
    raise NameError("Could not find an image id for {}.".format(img_name))


def get_last_img_id() -> int:
    infos = read_table('.{}analysis{}image_infos_analysis.csv'.format(sep, sep), ',', 'str',
                       silent=True, atleast_2d=True)
    return int(infos[-1, 0])


def get_img_id_from_time(time: float) -> int:
    infos = read_table('.{}analysis{}image_infos_analysis.csv'.format(sep, sep), ',', 'str', silent=True)
    for i in range(infos.shape[0]):
        if float(infos[i, 3]) == time:
            return int(infos[i, 0])
    raise NameError("Could not find an image id at {}s.".format(time))


def get_time_from_img_id(img_id: int) -> int:
    infos = read_table('.{}analysis{}image_infos_analysis.csv'.format(sep, sep), ',', 'str', silent=True)
    for i in range(infos.shape[0]):
        if float(infos[i, 0]) == img_id:
            return int(float(infos[i, 3]))
    raise NameError("Could not find a time to image {}.".format(img_id))
