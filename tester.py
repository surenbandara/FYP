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

model1 = load_model('model_Tablenet')
res= int(os.environ.get('res' ,640 ))

def create_mask(pred_mask1, pred_mask2):


  """Return a filter mask with the top 1 predictions
    only.

  """

  pred_mask1 = tf.argmax(pred_mask1, axis=-1)
  pred_mask1 = tf.expand_dims(pred_mask1, axis=-1)

  pred_mask2 = tf.argmax(pred_mask2, axis=-1)
  pred_mask2 = tf.expand_dims(pred_mask2, axis=-1)
  return pred_mask1[0], pred_mask2[0]


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



def show_predictions(name=None , dataset=None, num=1, output_dir=None):
    """Show a sample prediction.

    Parameters
    ----------
    dataset : [type], optional
        Input dataset, by default None
    num : int, optional
        Number of samples to show, by default 1
    output_dir : str, optional
        Directory to save the output images, by default None

    """

    if dataset and output_dir:
        for image  in dataset.take(num):

            pred_mask1, pred_mask2 = model1.predict(image, verbose=1)
            column_mask , row_mask = create_mask(pred_mask1, pred_mask2)

            # Save output

            output_image_path_col = os.path.join(output_dir+"/column/", name)
            cv2.imwrite(output_image_path_col,np.array( tf.keras.preprocessing.image.array_to_img(column_mask.numpy())))

            output_image_path_row = os.path.join(output_dir+"/row/", name)
            cv2.imwrite(output_image_path_row,np.array( tf.keras.preprocessing.image.array_to_img(row_mask.numpy())))




# Define the directory containing the .jpg files
directory_path = "./eval_input/"

# Get a list of all .jpg files in the directory
jpg_files = [file for file in os.listdir(directory_path) if file.endswith(".jpg")]

# Loop through each .jpg file and process it
for jpg_file in jpg_files:
    image_path = os.path.join(directory_path, jpg_file)
    test_dataset = load_img(image_path)
    show_predictions(name=jpg_file,dataset=test_dataset , output_dir="./eval_output/")


