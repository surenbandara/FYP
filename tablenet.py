import tensorflow as tf
import matplotlib.pyplot as plt

from tensorflow.keras import Sequential
from tensorflow.keras.models import Model
from tensorflow.keras.layers import BatchNormalization
from tensorflow.keras.layers import Conv2D
from tensorflow.keras.layers import MaxPooling2D
from tensorflow.keras.layers import Activation
from tensorflow.keras.layers import Dropout
from tensorflow.keras.layers import Lambda
from tensorflow.keras.layers import Dense
from tensorflow.keras.layers import Flatten
from tensorflow.keras.layers import Input, Concatenate, UpSampling2D
from keras.applications.resnet import ResNet101
from keras.preprocessing import image
from keras.layers import Input, Flatten, Dense
from keras.models import Model
import numpy as np
import os
import glob

res= int(os.environ.get("res"))
chunk_size = int(os.environ.get("chunk_size"))

#Model creation 

inputShape = (res,res, 3)


inputs = Input(shape=inputShape, name='input')
model_ResNet101_conv = ResNet101(input_tensor=inputs,weights='imagenet', include_top=False , pooling=None)
model_ResNet101_conv.summary()

conv3_block1_0_conv=model_ResNet101_conv.get_layer('conv3_block1_0_conv').output
print(conv3_block1_0_conv.get_shape)
conv4_block1_0_conv=model_ResNet101_conv.get_layer('conv4_block1_0_conv').output
print(conv4_block1_0_conv.get_shape)

x = model_ResNet101_conv.output
x = Conv2D(512,(1,1),activation = "relu", name = "conv2d_1")(x)
x = Dropout(0.8, name='block6_dropout1')(x)
x = Conv2D(512,(1,1),activation = "relu", name = "conv2d_2")(x)
x = Dropout(0.8, name='block6_dropout2')(x)

#decoder for row mask
def decoder_row(input_layer):

  conv7_row = Conv2D(512, (1, 1), activation = 'relu', name='conv7_row')(input_layer)
  conv7_row = Dropout(0.8, name='block7_dropout_row')(conv7_row)

  conv8_row = Conv2D(512, (1, 1), activation = 'relu', name='conv8_row')(conv7_row)
  conv8_row = UpSampling2D(size=(2, 2))(conv8_row)

  concat1=Concatenate()([conv8_row,conv4_block1_0_conv])
  concat1 = UpSampling2D(size=(2, 2))(concat1)

  concat2=Concatenate()([concat1,conv3_block1_0_conv])
  concat2 = UpSampling2D(size=(2,2),name='row_op')(concat2)

  final = UpSampling2D(size=(2,2))(concat2)

  final_layer = tf.keras.layers.Conv2DTranspose(3, 3, strides=2,padding='same', name='row_mask')

  final = final_layer(final)

  return final

#decoder for column mask
def decoder_column(input_layer):
  conv7_col = Conv2D(512, (1, 1), activation = 'relu', name='conv7_col')(input_layer)
  conv7_col = Dropout(0.8, name='block7_dropout_col')(conv7_col)

  conv8_col = Conv2D(512, (1, 1), activation = 'relu', name='conv8_col')(conv7_col)
  conv8_col = UpSampling2D(size=(2, 2))(conv8_col)

  concat1=Concatenate()([conv8_col,conv4_block1_0_conv])
  concat1 = UpSampling2D(size=(2, 2))(concat1)

  concat2=Concatenate()([concat1,conv3_block1_0_conv])
  concat2 = UpSampling2D(size=(2,2),name='column_op')(concat2)

  final = UpSampling2D(size=(2,2))(concat2)

  final_layer = tf.keras.layers.Conv2DTranspose(3, 3, strides=2,padding='same', name='column_mask')

  final = final_layer(final)

  return final

row_mask = decoder_row(x)
column_mask = decoder_column(x)


model = Model(inputs=inputs,outputs=[ column_mask , row_mask],name="table_net")

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


    return image, {'column_mask':column_mask ,'row_mask':row_mask}




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
    "row_mask": tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
}

lossWeights = { "column_mask": 1.0 ,"row_mask": 1.0}

model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=0.0001, epsilon=1e-08),
              loss=losses,
              metrics=['accuracy'],
              loss_weights=lossWeights)


EPOCHS = 3
VAL_SUBSPLITS = 1
VALIDATION_STEPS = test_size//BATCH_SIZE//VAL_SUBSPLITS


model_history = model.fit(train_dataset, epochs=EPOCHS,
                          steps_per_epoch=STEPS_PER_EPOCH,
                          validation_steps=VALIDATION_STEPS,
                          validation_data=test_dataset,
                          )

model.save('model_Tablenet')

print("Model saved successfully!")