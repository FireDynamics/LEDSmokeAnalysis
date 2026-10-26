"""
Detection and reporting of image files that cannot be read.

An experiment can consist of thousands of photos and its analysis can run for hours, so a single unreadable
photo - truncated by a full memory card, lost in a faulty transfer or never written at all - should not abort
the whole run. The image reading functions raise :class:`CorruptedImageError` instead of the error of the
underlying imaging library, the analysis steps skip those images and every skipped image is listed in
'analysis/corrupted_images.csv'.
"""

import csv
import os
from typing import Dict, Iterable, List, Set

CORRUPTED_IMAGES_FILE_PATH = os.path.join('analysis', 'corrupted_images.csv')
CORRUPTED_IMAGES_COLUMNS = ['image_path', 'analysis_step', 'error_type', 'error_message']
MAX_PRINTED_IMAGES = 10


class CorruptedImageError(Exception):
    """
    Raised when an image file cannot be read.

    This covers every reason why the pixel data or the EXIF metadata of an image are unavailable: the file is
    missing, empty, truncated, not an image at all, stored in a format LEDSA does not support or lacking the
    EXIF entry the analysis needs.

    :ivar img_filename: Name or path of the image that could not be read.
    :vartype img_filename: str
    :ivar error_type: Class name of the underlying error, or 'CorruptedImageError' if there is none.
    :vartype error_type: str
    :ivar error_message: Description of why the image could not be read.
    :vartype error_message: str
    """

    def __init__(self, img_filename: str, error: BaseException = None, message: str = ''):
        """
        :param img_filename: Name or path of the image that could not be read.
        :type img_filename: str
        :param error: Error raised by the imaging library, if the image was rejected while being read.
        :type error: BaseException or None
        :param message: Description of the problem, used instead of the message of error.
        :type message: str
        """
        self.img_filename = img_filename
        self.error_type = type(error).__name__ if error is not None else type(self).__name__
        # Newlines would break the one line per image layout of the report file
        self.error_message = ' '.join((message or str(error)).split())
        super().__init__(_format_message(self.img_filename, self.error_type, self.error_message))

    def __reduce__(self):
        """
        Reduce the error to its stored fields, so that it survives being pickled with its message intact.

        The default implementation of :class:`BaseException` would pass the already formatted message back to
        :meth:`__init__` as the image name, which matters because the multicore analysis returns these errors
        from its worker processes.
        """
        return _rebuild_corrupted_image_error, (self.img_filename, self.error_type, self.error_message)


def _format_message(img_filename: str, error_type: str, error_message: str) -> str:
    """
    Compose the message of a :class:`CorruptedImageError`.

    :param img_filename: Name or path of the image that could not be read.
    :type img_filename: str
    :param error_type: Class name of the underlying error.
    :type error_type: str
    :param error_message: Description of why the image could not be read.
    :type error_message: str
    :return: The composed error message.
    :rtype: str
    """
    return f'{img_filename} could not be read: {error_type}: {error_message}'


def _rebuild_corrupted_image_error(img_filename: str, error_type: str, error_message: str) -> 'CorruptedImageError':
    """
    Recreate a :class:`CorruptedImageError` from its stored fields while it is being unpickled.

    :param img_filename: Name or path of the image that could not be read.
    :type img_filename: str
    :param error_type: Class name of the underlying error.
    :type error_type: str
    :param error_message: Description of why the image could not be read.
    :type error_message: str
    :return: The recreated error.
    :rtype: CorruptedImageError
    """
    error = CorruptedImageError(img_filename, message=error_message)
    error.error_type = error_type
    error.args = (_format_message(img_filename, error_type, error_message),)
    return error


def report_corrupted_images(errors: Iterable[CorruptedImageError], analysis_step: str) -> None:
    """
    List the corrupted images in 'analysis/corrupted_images.csv' and print a summary of the skipped images.

    Images that the report already lists for the same analysis step are not appended a second time, so that
    repeated and restarted runs do not pile up duplicates.

    :param errors: The errors raised for the images that were skipped.
    :type errors: Iterable[CorruptedImageError]
    :param analysis_step: Name of the analysis step during which the images were skipped.
    :type analysis_step: str
    """
    errors = list(errors)
    if not errors:
        return

    already_reported = {(row['image_path'], row['analysis_step']) for row in load_corrupted_images()}
    new_rows = []
    for error in errors:
        row = {'image_path': error.img_filename, 'analysis_step': analysis_step,
               'error_type': error.error_type, 'error_message': error.error_message}
        if (row['image_path'], row['analysis_step']) not in already_reported:
            already_reported.add((row['image_path'], row['analysis_step']))
            new_rows.append(row)

    print(f'{len(errors)} corrupted image(s) skipped during {analysis_step}:')
    for error in errors[:MAX_PRINTED_IMAGES]:
        print(f'  {error.img_filename}: {error.error_type}: {error.error_message}')
    if len(errors) > MAX_PRINTED_IMAGES:
        print(f'  ... and {len(errors) - MAX_PRINTED_IMAGES} more')

    if new_rows:
        _append_to_corrupted_images_file(new_rows)
    print(f'All skipped images are listed in {CORRUPTED_IMAGES_FILE_PATH}.')


def load_corrupted_images() -> List[Dict[str, str]]:
    """
    Read the images listed in 'analysis/corrupted_images.csv'.

    :return: One dictionary per listed image, with the keys given by CORRUPTED_IMAGES_COLUMNS. Empty if no
             image has been reported yet.
    :rtype: List[Dict[str, str]]
    """
    if not os.path.isfile(CORRUPTED_IMAGES_FILE_PATH):
        return []
    with open(CORRUPTED_IMAGES_FILE_PATH, 'r', newline='', encoding='utf-8') as corrupted_images_file:
        return [row for row in csv.DictReader(corrupted_images_file) if row.get('image_path')]


def get_corrupted_image_paths() -> Set[str]:
    """
    Get the paths of all images listed in 'analysis/corrupted_images.csv'.

    :return: Paths of the images that were skipped because they could not be read.
    :rtype: Set[str]
    """
    return {row['image_path'] for row in load_corrupted_images()}


def _append_to_corrupted_images_file(rows: List[Dict[str, str]]) -> None:
    """
    Append image entries to 'analysis/corrupted_images.csv', creating the file with its header if needed.

    :param rows: Entries to append, with the keys given by CORRUPTED_IMAGES_COLUMNS.
    :type rows: List[Dict[str, str]]
    """
    report_dir = os.path.dirname(CORRUPTED_IMAGES_FILE_PATH)
    if report_dir:
        os.makedirs(report_dir, exist_ok=True)
    file_exists = os.path.isfile(CORRUPTED_IMAGES_FILE_PATH)
    with open(CORRUPTED_IMAGES_FILE_PATH, 'a', newline='', encoding='utf-8') as corrupted_images_file:
        writer = csv.DictWriter(corrupted_images_file, fieldnames=CORRUPTED_IMAGES_COLUMNS)
        if not file_exists:
            writer.writeheader()
        writer.writerows(rows)
