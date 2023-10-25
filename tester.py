import cv2
import numpy as np
import os
from tqdm import tqdm as progress_bar
import tarfile
from googleapiclient.discovery import build
from google.oauth2 import service_account
from googleapiclient.http import MediaFileUpload
import shutil
import tensorflow as tf
from keras.models import load_model
from PIL import Image

model_row = load_model('row_model')
model_column = load_model('column_model')
res= int(os.environ.get('res' ,640 ))



def create_mask(pred_mask1):


  """Return a filter mask with the top 1 predictions
    only.

  """

  pred_mask1 = tf.argmax(pred_mask1, axis=-1)
  pred_mask1 = tf.expand_dims(pred_mask1, axis=-1)

 
  return pred_mask1[0]


def load_img(path):
  test_image = Image.open(path)
  test_data = tf.data.Dataset.list_files(path)
  #size = len(list(test_data))
  size = int(len(list(test_data)))
  BATCH_SIZE = 1
  test=test_data.take(size)
  test = test.map(parse_image)
  test_dataset = test.batch(BATCH_SIZE)

  return test_dataset


def parse_image(img_path):
    """Load an image and its annotation (mask) and returning
    a dictionary.

    Parameters
    ----------
    img_path : str
        Image (not the mask) location.

    Returns
    -------
    dict
        Dictionary mapping an image and its annotation.
    """

    image = tf.io.read_file(img_path)
    image = tf.image.decode_jpeg(image, channels=3)
    image =tf.image.resize(image, [res, res])
    image = tf.cast(image, tf.float32) / 255.0
    #image = tf.image.convert_image_dtype(image, tf.uint8)



    return image



def show_predictions_row(name=None , dataset=None, num=1, output_dir=None):

    if dataset and output_dir:
        for image  in dataset.take(num):

            pred_mask1 = model_row.predict(image, verbose=1)
            column_mask  = create_mask(pred_mask1)

            # Save output

            output_image_path_row = os.path.join(output_dir+"/row/", name)
            cv2.imwrite(output_image_path_row,np.array( tf.keras.preprocessing.image.array_to_img(column_mask.numpy())))



def show_predictions_column(name=None , dataset=None, num=1, output_dir=None):

    if dataset and output_dir:
        for image  in dataset.take(num):

            pred_mask1 = model_column.predict(image, verbose=1)
            column_mask  = create_mask(pred_mask1)

            # Save output

            output_image_path_col = os.path.join(output_dir+"/column/", name)
            cv2.imwrite(output_image_path_col,np.array( tf.keras.preprocessing.image.array_to_img(column_mask.numpy())))




# Define the directory containing the .jpg files
directory_path = "./eval_input/"

# Get a list of all .jpg files in the directory
jpg_files = [file for file in os.listdir(directory_path) if file.endswith(".jpg")]

# Loop through each .jpg file and process it
for jpg_file in jpg_files:
    image_path = os.path.join(directory_path, jpg_file)
    test_dataset = load_img(image_path)
    show_predictions_row(name=jpg_file,dataset=test_dataset , output_dir="./eval_output/")
    show_predictions_column(name=jpg_file,dataset=test_dataset , output_dir="./eval_output/")



