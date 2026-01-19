import glob
import os
from datetime import datetime
from os import path
import csv
import pandas as pd

from ledsa.core.image_reading import get_exif_entry

def set_working_dir():
    """
    Prompts user for directory path and changes to that directory if it exists.
    Exits program if directory is invalid.
    """
    working_dir = input("Set Working directory:")
    if path.exists(working_dir):
        os.chdir(working_dir)
        print("Working directory is set to \"{0}\"".format(os.getcwd()))
    else:
        print("\"{0}\" does not exist!".format(os.getcwd()))
        exit()


def get_files():
    """
    Prompts user for image and raw file types to process.
    Extracts capture date from EXIF data of images.
    Finds corresponding raw files if they exist.
    
    Returns:
        pd.DataFrame: DataFrame containing image filenames as index and columns for
                     capture date and associated raw filenames, sorted by capture date.
    """
    # Get file type inputs from user
    image_types = input("What image types do you want to take into account? (Seperate by \",\"):")
    raw_types = input("What raw types do you want to take into account? (Seperate by \",\"):")
    image_types = [x.strip() for x in image_types.split(',')]
    raw_types = [x.strip() for x in raw_types.split(',')]
    image_dict = {}
    image_files = []

    # Find all matching image files
    for file_type in image_types:
        image_files.extend(glob.glob('*.{}'.format(file_type)))

    # Process each image file
    for image in image_files:
        # Extract EXIF datetime data
        with open(image, 'rb') as image_file:
            tag_datetime = 'DateTimeOriginal'
            tag_subsectime = 'SubSecTimeDigitized'
            capture_date = get_exif_entry(image, tag_datetime)
            try:
                # Try parsing with subsecond precision
                subsec_time = get_exif_entry(image, tag_subsectime)
                datetime_object = datetime.strptime(capture_date + "." + subsec_time, '%Y:%m:%d %H:%M:%S.%f')
            except:
                # Fall back to second precision
                datetime_object = datetime.strptime(capture_date, '%Y:%m:%d %H:%M:%S')

        # Find matching raw file if it exists
        for file_type in raw_types:
            raw_file = image.rsplit(".", 1)[0] + "." + file_type
            if path.isfile(raw_file):
                raw_file = raw_file
                break
            else:
                raw_file = "NaT"

        image_dict[image] = [datetime_object, raw_file]

    # Create and sort DataFrame
    image_df = pd.DataFrame.from_dict(image_dict, columns=["capture_date", "raw_file"], orient='index')
    image_df = image_df.sort_values(by=['capture_date'], ascending=True)
    return image_df


def rename_images_by_date(image_df):
    """
    Rename image and raw files based on capture date and user-provided name.
    
    Creates a log file of all renames. After previewing changes, user can choose
    to proceed with renaming or cancel. User can also choose to keep or delete
    the log file afterwards.

    Args:
        image_df (pd.DataFrame): DataFrame containing image metadata and filenames
                                to be processed.
    """
    # Get base name from user and setup logging
    name = input("Please enter name for image:")
    print("Files are renamed as follows:")
    count = 1
    now = datetime.now()
    now_str = now.strftime("%Y_%m_%d_%H_%M_%S")
    filename = f"{now_str}_{name}_rename_log.csv"

    # Preview rename changes and write to log
    with open(filename, 'w') as log_file:
        writer = csv.writer(log_file)
        writer.writerow(("old_image_name", "new_image_name", "old_raw_name", "new_raw_name"))
        for index, row in image_df.iterrows():
            image_year = row['capture_date'].strftime('%-y%m%d')
            raw_old_name = row['raw_file']

            # Generate new filenames
            if pd.isnull(raw_old_name):
                raw_new_name = "NaT"
            else:
                file_extension = raw_old_name.split('.')[-1]
                raw_new_name = "{0}_{1}_{2}.{3}".format(image_year, name, count, file_extension)
            new_image_name = "{0}_{1}_{2}.jpg".format(image_year, name, count)
            print("{0} --> {1}    {2} --> {3}".format(index, new_image_name, raw_old_name, raw_new_name))
            writer.writerow((index, new_image_name, raw_old_name, raw_new_name))
            count += 1

    # Process user choice to proceed or cancel
    while True:
        continue_rename = input("Do you want to continue? yes[y] or no[n]")
        if continue_rename == "y":
            # Perform the actual renaming
            count = 1
            for index, row in image_df.iterrows():
                image_year = row['capture_date'].strftime('%-y%m%d')
                raw_old_name = row['raw_file']

                if not pd.isnull(raw_old_name):
                    file_extension = raw_old_name.split('.')[-1]
                    raw_new_name = "{0}_{1}_{2}.{3}".format(image_year, name, count, file_extension)
                    os.rename(raw_old_name, raw_new_name)
                new_image_name = "{0}_{1}_{2}.jpg".format(image_year, name, count)
                os.rename(index, new_image_name)
                count += 1
            print("Files successfully renamed!".format(count))

            # Handle log file retention
            keep_log = input("Do you want to keep the changelog? yes[y] or no[n]")
            if keep_log == 'y':
                print(f"{filename} saved!")
                exit()
            elif keep_log == 'n':
                os.remove(filename)
                print("Changelog deleted!")
                exit()
        elif continue_rename == "n":
            print("No files were renamed!")
            exit()
        else:
            continue


def main():
    set_working_dir()
    image_df = get_files()
    rename_images_by_date(image_df)


if __name__ == '__main__':
    main()
