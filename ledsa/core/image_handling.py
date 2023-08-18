import os

from ledsa.core.file_handling import read_table


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
