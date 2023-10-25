import cv2
import os

def resize_image(image, max_dim=640):
  """Resizes an image to a maximum dimension, keeping the original aspect ratio.

  Args:
    image: A NumPy array representing the image.
    max_dim: The maximum dimension of the resized image.

  Returns:
    A NumPy array representing the resized image.
  """

  height, width, channels = image.shape

  # Calculate the scale factor.
  scale_factor = max_dim / max(height, width)

  # Resize the image.
  resized_image = cv2.resize(image, (int(width * scale_factor), int(height * scale_factor)))

  # Fill the rest with black.
  padded_image = cv2.copyMakeBorder(resized_image, 0, max_dim - resized_image.shape[0], 0, max_dim - resized_image.shape[1], cv2.BORDER_CONSTANT, value=[255, 255, 255])

  return padded_image

# Get the directory containing the images.
image_dir = "D:\ENTC_7\FYP\FYP\eval_input"


ind=0
# Iterate over all the images in the directory.
for image_file in os.listdir(image_dir):

  # Load the image.
  image = cv2.imread(os.path.join(image_dir, image_file))

  # Resize the image.
  resized_image = resize_image(image)

  name="image"+str(ind)

  # Save the resized image as .jpg format.
  cv2.imwrite(os.path.join(image_dir, f"{name}.jpg"), resized_image)

  #os.remove(os.path.join(image_dir, f"{image_file}"))

  ind+=1

