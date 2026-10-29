import os
from typing import List

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


def get_analysis_img_ids(experiment_img_ids: List[int], img_name_string: str) -> List[int]:
    """
    Translate experiment image IDs into the analysis IDs of the same images.

    config.ini names images by the ID their file name is built from, while the analysis counts
    the images it was given from 1 on. The image infos hold both, so they are the translation.

    :param experiment_img_ids: IDs the image file names are built from.
    :type experiment_img_ids: List[int]
    :param img_name_string: Format string of the image file names, e.g. 'img{:04d}.jpg'.
    :type img_name_string: str
    :return: Analysis IDs of the same images.
    :rtype: List[int]
    :raises ValueError: If one of the images was not analysed.
    """
    file_path = os.path.join('analysis', 'image_infos_analysis.csv')
    infos = read_table(file_path, ',', 'str', silent=True, atleast_2d=True)
    analysis_img_id_of_name = {infos[i, 1]: int(infos[i, 0]) for i in range(infos.shape[0])}
    img_names = [img_name_string.format(int(img_id)) for img_id in experiment_img_ids]
    missing = [img_id for img_id, name in zip(experiment_img_ids, img_names)
               if name not in analysis_img_id_of_name]
    if missing:
        raise ValueError(f'The images {missing} of ref_img_experiment_ids in config.ini are not among '
                         f'the analysed images of {file_path}. They lie outside the analysed image '
                         f'range or could not be read. Run the data extraction again after naming them, '
                         f'so that they are analysed whatever num_skip_imgs leaves out.')
    return [analysis_img_id_of_name[name] for name in img_names]


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
