import os

from ledsa.core.file_handling import read_table


def format_img_name(img_name_string: str, img_id) -> str:
    """
    Build an image file name from the config template and an image ID.

    Plain ``{}`` placeholders keep the ID as given, so zero-padded IDs such as
    '0001' (e.g. DSC_0001.NEF) are preserved. Numeric format specifications
    such as ``{:04d}`` require an integer, for which the ID is cast.

    :param img_name_string: Name template from the config, e.g. 'DSC_{}.CR3'.
    :type img_name_string: str
    :param img_id: The image ID, as string or integer.
    :return: The formatted image file name.
    :rtype: str
    """
    try:
        return img_name_string.format(img_id)
    except ValueError:
        return img_name_string.format(int(img_id))


def get_img_name(img_id: str) -> str:
    """
    Retrieves the image path corresponding to a given image ID.

    :param img_id: The ID of the image to be retrieved.
    :type img_id: str
    :return: The name of the image corresponding to the provided ID.
    :rtype: str
    :raises NameError: If no image name is found for the provided ID.
    """
    file_path = os.path.join('analysis', 'image_infos_analysis.csv')
    infos = read_table(file_path, ',', 'str', silent=True, atleast_2d=True)
    for i in range(infos.shape[0]):
        if int(infos[i, 0]) == int(img_id):
            return infos[i, 1]
    raise NameError("Could not find an image name to id {}.".format(img_id))


def get_img_id(img_name: str) -> str:
    """
    Retrieves the image ID corresponding to a given image name.

    :param img_name: The name of the image.
    :type img_name: str
    :return: The ID of the image corresponding to the provided name.
    :rtype: str
    :raises NameError: If no image ID is found for the provided image name.
    """
    file_path = os.path.join('analysis', 'image_infos_analysis.csv')
    infos = read_table(file_path, ',', 'str', silent=True, atleast_2d=True)
    for i in range(infos.shape[0]):
        if infos[i, 1] == img_name:
            return infos[i, 0]
    raise NameError("Could not find an image id for {}.".format(img_name))
