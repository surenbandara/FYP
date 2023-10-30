import firebase_admin
from firebase_admin import ml
from firebase_admin import credentials

firebase_admin.initialize_app(
  credentials.Certificate('credentials.json'),
  )

# First, import and initialize the SDK as shown above.

# Load a tflite file and upload it to Cloud Storage
source = ml.TFLiteGCSModelSource.from_tflite_model_file('column_model.tflite')

# Create the model object
tflite_format = ml.TFLiteFormat(model_source=source)
model = ml.Model(
    display_name="column_model",  # This is the name you use from your app to load the model.
    tags=["column_model"],             # Optional tags for easier management.
    model_format=tflite_format)

# Add the model to your Firebase project and publish it
new_model = ml.create_model(model)
ml.publish_model(new_model.model_id)