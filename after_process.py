import cv2
import numpy as np
import matplotlib.pyplot as plt

# Load the image
image = cv2.imread("D:\ENTC_7\FYP\FYP\eval_output\column\download.jpeg.jpg")

# Define the color values to replace and the replacement color
replace_color = [255, 255, 255]
replacement_color = [0, 0, 0]

# Iterate over the image and replace pixels
height, width, channels = image.shape
for y in range(height):
    for x in range(width):
        pixel = image[y, x]
        if (not np.array_equal(pixel, replace_color) and not np.array_equal(pixel, replacement_color)) :
            image[y, x] = replace_color

fre = np.zeros(width)

white = [255, 255, 255]
black = [0, 0, 0]

for x in range(width):
    for y in range(height):
        if not (x==0):
            pixel = image[y, x]
            pre_pixel = image[y, x-1]
            if(np.array_equal(pixel, black) and np.array_equal(pre_pixel, white)):
                fre[x]+=1


# Create x-axis values as index numbers
x = np.arange(len(fre))

# Create a plot
plt.plot(x, fre)

# Add labels and a title
plt.xlabel('Index')
plt.ylabel('Value')
plt.title('NumPy Array Data')

# Show the plot
plt.show()