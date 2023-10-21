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
res= int(os.environ.get('res'))

def upload_file(filename):
  # Path to your credentials JSON file (downloaded in Step 1)
  credentials_file = 'credentials.json'

  # The file you want to upload
  file_path = filename

  # The folder ID where you want to upload the file (optional)
  folder_id = '1QNS_1mUtJcRzjLCdkQnvCM9egouhynlP'  # You can find the folder ID in the Google Drive URL

  # Authenticate using your credentials
  credentials = service_account.Credentials.from_service_account_file(
      credentials_file, scopes=['https://www.googleapis.com/auth/drive.file']
  )

  # Build the Google Drive API service
  drive_service = build('drive', 'v3', credentials=credentials)

  # Prepare file metadata
  file_metadata = {
      'name': 'model.tar.gz',  # Name of the file on Google Drive
      'parents': [folder_id] if folder_id else []  # ID of the parent folder (optional)
  }

  # Upload the file
  media = MediaFileUpload(file_path, mimetype='application/gzip')
  uploaded_file = drive_service.files().create(body=file_metadata, media_body=media, fields='id').execute()

  print(f'File ID: {uploaded_file.get("id")}')


def create_mask(pred_mask1):


  """Return a filter mask with the top 1 predictions
    only.

  """

  pred_mask1 = tf.argmax(pred_mask1, axis=-1)
  pred_mask1 = tf.expand_dims(pred_mask1, axis=-1)


  return pred_mask1[0]

def copy_jpg_files(source_dir, destination_dir):
    """Copy all .jpg files from the source directory to the destination directory.

    Parameters:
        source_dir (str): The source directory containing .jpg files.
        destination_dir (str): The destination directory where .jpg files will be copied.
    """
    # Create the destination directory if it doesn't exist
    os.makedirs(destination_dir, exist_ok=True)

    # List all files in the source directory
    files = os.listdir(source_dir)

    for file in files:
        if file.endswith(".jpg"):
            source_path = os.path.join(source_dir, file)
            destination_path = os.path.join(destination_dir, file)

            try:
                shutil.copy2(source_path, destination_path)  # Copy the .jpg file
                print(f"Copied {file} to {destination_dir}")
            except Exception as e:
                print(f"Error copying {file}: {str(e)}")


def create_folder(directory, folder_name):
    """Create a new folder with the given name in the specified directory.

    Parameters:
        directory (str): The directory where the new folder will be created.
        folder_name (str): The name of the new folder to be created.
    
    Returns:
        str: The path of the newly created folder.
    """
    # Join the directory and folder name to get the full path
    new_folder_path = os.path.join(directory, folder_name)

    try:
        # Create the new folder
        os.makedirs(new_folder_path)
        print(f"Created folder: {new_folder_path}")
    except FileExistsError:
        print(f"Folder already exists: {new_folder_path}")
    
    return new_folder_path

def create_tar_gz(archive_name, source_folder):
    """Create a .tar.gz archive for a given folder.

    Parameters:
        archive_name (str): The name of the archive file to be created.
        source_folder (str): The path to the folder you want to archive.

    Returns:
        None
    """
    with tarfile.open(archive_name, "w:gz") as archive:
        archive.add(source_folder, arcname="")
        print(f"Created {archive_name} from {source_folder}")

def delete_folder_and_archive(folder_path):

    # Try to delete a .tar.gz archive with the same name
    archive_name = folder_path + ".tar.gz"
    try:
        os.remove(archive_name)
        print(f"Deleted archive: {archive_name}")
    except FileNotFoundError:
        print(f"No archive found: {archive_name}")
    except Exception as e:
        print(f"Failed to delete archive: {e}")



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
        for image in dataset.take(num):
            pred_mask1 = model1.predict(image, verbose=1)
            column_mask = create_mask(pred_mask1).numpy()

            # Display the input and output images
            #display([image[0], column_mask])
            output_image_path = os.path.join(output_dir, name);print(np.all(np.array( tf.keras.preprocessing.image.array_to_img(column_mask))==0))
            cv2.imwrite(output_image_path,np.array( tf.keras.preprocessing.image.array_to_img(column_mask)))


#create_folder(".","prediction")
#copy_jpg_files(".", "prediction")
#predictions("./resize_image","./testing_area")

# Define the directory containing the .jpg files
directory_path = "./testing_resize_images/"

# Get a list of all .jpg files in the directory
jpg_files = [file for file in os.listdir(directory_path) if file.endswith(".jpg")]

# Loop through each .jpg file and process it
for jpg_file in jpg_files:
    image_path = os.path.join(directory_path, jpg_file)
    test_dataset = load_img(image_path)
    show_predictions(name=jpg_file,dataset=test_dataset , output_dir="./testing_area/")

create_tar_gz("prediction.tar.gz" , "./testing_area")
