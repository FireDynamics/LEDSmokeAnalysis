import glob
import os
from datetime import datetime
from os import path
import csv
import pandas as pd

import exifread


def set_working_dir():
    print("Image preparation assumes you have all photos stored as image data and (optional) RAW data files. Only RAW "
          "files are considered that have a corresponding image file with the same name.")
    # Set working directory to given path; if path is not valid then exit program
    working_dir = input("Set directory where images are stored, make sure you have write permissions to the data "
                        "directory:")
    if path.exists(working_dir):
        os.chdir(working_dir)
        print("Image directory is set to \"{0}\"".format(os.getcwd()))
    else:
        print("\"{0}\" does not exist!".format(os.getcwd()))
        exit()


def get_files():
    # Get Dataframe with Index = image_names; capture_dates, raw_names
    image_types = input("What (not RAW) image types do you want to take into account? (Seperate by \",\"):")
    raw_types = input("What raw types do you want to take into account? (Seperate by \",\"):")
    image_types = [x.strip() for x in image_types.split(',')]
    raw_types = [x.strip() for x in raw_types.split(',')]
    image_dict = {}
    image_files = []

    for file_type in image_types:
        image_files.extend(glob.glob('*.{}'.format(file_type)))
    for image in image_files:
        with open(image, 'rb') as image_file:
            tag_datetime ='DateTimeOriginal'
            tag_subsectime ='SubSecTimeDigitized'
            exif = exifread.process_file(image_file, details=False)
            capture_date = exif[f"EXIF {tag_datetime}"].values
            try:
                subsec_time = exif[f"EXIF {tag_subsectime}"].values
                datetime_object = datetime.strptime(capture_date + "." + subsec_time, '%Y:%m:%d %H:%M:%S.%f')
            except:
                datetime_object = datetime.strptime(capture_date, '%Y:%m:%d %H:%M:%S')

        for file_type in raw_types:
            raw_file = image.rsplit(".", 1)[0] + "." + file_type
            if path.isfile(raw_file):
                raw_file = raw_file
                break
            else:
                raw_file = "NaT"

        image_dict[image] = [datetime_object, raw_file]
    image_df = pd.DataFrame.from_dict(image_dict, columns=["capture_date", "raw_file"], orient='index')
    image_df = image_df.sort_values(by=['capture_date'], ascending=True)
    return image_df


def rename_images_by_date(image_df):
    # Rename image files and raw files based on image_df
    name = input("Please enter name for image:")
    print("Files are renamed as follows:")
    count = 1
    now = datetime.now()
    now_str = now.strftime("%Y_%m_%d_%H_%M_%S")
    filename = f"{now_str}_{name}_rename_log.csv"
    with open(filename, 'w') as log_file:
        writer = csv.writer(log_file)
        writer.writerow(("old_image_name", "new_image_name", "old_raw_name", "new_raw_name"))
        for index, row in image_df.iterrows():
            image_year = row['capture_date'].strftime('%-y%m%d')
            raw_old_name = row['raw_file']

            if pd.isnull(raw_old_name):
                raw_new_name = "NaT"
            else:
                file_extension = raw_old_name.split('.')[-1]
                raw_new_name = "{0}_{1}_{2}.{3}".format(image_year, name, count, file_extension)
            new_image_name = "{0}_{1}_{2}.jpg".format(image_year, name, count)
            print("{0} --> {1}    {2} --> {3}".format(index, new_image_name, raw_old_name, raw_new_name))
            writer.writerow((index, new_image_name, raw_old_name, raw_new_name))
            count += 1

    while True:
        continue_rename = input("Do you want to continue? yes[y] or no[n]")
        if continue_rename == "y":
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
