from PIL import Image
import os
from tqdm import tqdm
import numpy as np
import xml.etree.ElementTree as ET

input_path = './data/'
output_path = './resize_image/'

res = int(os.environ.get("res" ,640))
max_images = int(os.environ.get("max_images" ,100000))


def resize_img(input_path, output_path, res, max_images=1000):
    # Get a list of all image files to process
    image_files = [item for item in os.listdir(input_path) if item.endswith(".jpg")]

    # Limit the number of images to process to the first 1000
    image_files = image_files[:max_images]

    # Initialize the progress bar
    pbar = tqdm(total=len(image_files), unit="image")

    for item in image_files:
        im = Image.open(os.path.join(input_path, item))
        f, e = os.path.splitext(item)
        imResize = im.resize((res,res), Image.LANCZOS)
        imResize.save(os.path.join(output_path, f + '.jpg'), quality=90)
        pbar.update(1)  # Update the progress bar for each processed image

    pbar.close()  # Close the progress bar

# Call the function to resize
resize_img(input_path, output_path, res, max_images)

print("Image resizing completed!.")