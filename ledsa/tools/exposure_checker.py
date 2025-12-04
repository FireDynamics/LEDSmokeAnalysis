import os
from ledsa.core.image_reading import read_channel_data_from_img, get_exif_entry
import pandas as pd

# Configure the image path and range
image_dir = os.getcwd()  # Update this to your image directory
list_image = [f for f in os.listdir(image_dir) if '.CR3' in f or '.CR2' in f or '.NEF' in f]

# image_name_string = "DSC_{:04d}.CR3"  # F-string template for image names
# image_range = range(23, 24)  # Range of image numbers to process
channels = [0,1,2]  # Color channel to analyze (0=Red, 1=Green, 2=Blue)
saturation = 2**14-1

df_exposure_test = pd.DataFrame()
# Process each image in the range
for img in list_image:
    # Create the image filename using the template
    image_filename = os.path.join(image_dir,img)
    
    # Check if the file exists
    if not os.path.exists(image_filename):
        print(f"Image {image_filename} not found, skipping.")
        continue
    
    print(f"Processing image: {image_filename}")
    
    # Read the image data for the specified channel
    try:
        for channel in channels:
            channel_array = read_channel_data_from_img(image_filename, channel)
            df_exposure_test.loc[img,f'ch{channel}'] = channel_array.max()
            df_exposure_test.loc[img,f'ch{channel}_sat'] = channel_array.max()/saturation*100
    except:
        pass

print(f'Image with min Channel 0 Saturation {df_exposure_test['ch0_sat'].min()}')
print(f'Image with max Channel 0 Saturation {df_exposure_test['ch0_sat'].max()}')
print(f'Image with min Channel 1 Saturation {df_exposure_test['ch1_sat'].min()}')
print(f'Image with max Channel 1 Saturation {df_exposure_test['ch1_sat'].max()}')
print(f'Image with min Channel 2 Saturation {df_exposure_test['ch2_sat'].min()}')
print(f'Image with max Channel 2 Saturation {df_exposure_test['ch2_sat'].max()}')