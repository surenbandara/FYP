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


chunk_size = int(os.environ.get("chunk_size",100000))
EPOCHS = int(os.environ.get("epochs",2))


model = load_model('model_Tablenet')
##################################################################################################


# Data preperation

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

    # For one Image path:
    # /content/drive/MyDrive/Marmot_data.zip/images/10.1.1.1.2006_3.jpeg
    # Its corresponding row mask path is:
    #  /content/drive/MyDrive/Marmot_data.zip/row_mask/10.1.1.1.2006_3.jpeg
    # Its corresponding column mask path is:
    #  /content/drive/MyDrive/Marmot_data.zip/column_mask/10.1.1.1.2006_3.jpeg

    row_mask_path = tf.strings.regex_replace(img_path, "resize_image", "row_mask")
    row_mask = tf.io.read_file(row_mask_path)

    # The masks contain a class index for each pixels
    row_mask = tf.image.decode_jpeg(row_mask, channels=1)
    row_mask =tf.image.resize(row_mask, [res, res])
    row_mask = tf.cast(row_mask, tf.float32) / 255.0


    column_mask_path = tf.strings.regex_replace(img_path, "resize_image", "column_mask")
    column_mask = tf.io.read_file(column_mask_path)

    # The masks contain a class index for each pixels
    column_mask = tf.image.decode_jpeg(column_mask, channels=1)
    column_mask =tf.image.resize(column_mask, [res, res])
    column_mask = tf.cast(column_mask, tf.float32) / 255.0


    #return image, {'column_mask':column_mask ,'row_mask':row_mask}
    return image, {'column_mask':column_mask }




# Calculate the total size (number of files)
total_size = chunk_size

len_of_train = int(total_size*0.9)
test_size = int(total_size*0.1)

data = tf.data.Dataset.list_files("./resize_image/*jpg")
train = data.take(len_of_train)
test = data.skip(test_size)

train = train.map(parse_image, num_parallel_calls=tf.data.experimental.AUTOTUNE)
test = test.map(parse_image)



BATCH_SIZE = 4
BUFFER_SIZE = 16

STEPS_PER_EPOCH = len_of_train // BATCH_SIZE

dataset = {"train": train, "test": test}

# -- Train Dataset --#

dataset['train'] = dataset['train'].shuffle(buffer_size=BUFFER_SIZE)
dataset['train'] = dataset['train'].batch(BATCH_SIZE)
dataset['train'] = dataset['train'].repeat()
dataset['train'] = dataset['train'].prefetch(buffer_size=tf.data.experimental.AUTOTUNE)

#-- test Dataset --#

dataset['test'] = dataset['test'].repeat()
dataset['test'] = dataset['test'].batch(BATCH_SIZE)


print(dataset['train'])
print(dataset['test'])

train_dataset = dataset['train']
test_dataset = dataset['test']


########################################################################

#Training Model

losses = {
    "column_mask": tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
}

# losses = {
#     "column_mask": tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
#     "row_mask": tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
# }

lossWeights = { "column_mask": 1.0 }

#lossWeights = { "column_mask": 1.0 ,"row_mask": 1.0}

model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=0.0001, epsilon=1e-08),
              loss=losses,
              metrics=['accuracy'],
              loss_weights=lossWeights)


VAL_SUBSPLITS = 1
VALIDATION_STEPS = test_size//BATCH_SIZE//VAL_SUBSPLITS


model_history = model.fit(train_dataset, epochs=EPOCHS,
                          steps_per_epoch=STEPS_PER_EPOCH,
                          validation_steps=VALIDATION_STEPS,
                          validation_data=test_dataset,
                          )

model.save('model_Tablenet')

print("Model saved successfully!")
