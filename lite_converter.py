import tensorflow as tf

for model_name in ['column_model','row_model']:
    # Convert the model
    converter = tf.lite.TFLiteConverter.from_saved_model(model_name) # path to the SavedModel directory
    tflite_model = converter.convert()

    # Save the model.
    with open(model_name+'.tflite', 'wb') as f:
        f.write(tflite_model)