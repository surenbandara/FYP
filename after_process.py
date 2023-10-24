import cv2
import numpy as np
import matplotlib.pyplot as plt
import math

# Load the image
image = cv2.imread("D:\ENTC_7\FYP\FYP\eval_output\column\download.png.jpg")

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

ed1 = np.zeros(width)
ed2 = np.zeros(width)

mid=np.zeros(width)

white = [255, 255, 255]
black = [0, 0, 0]

for y in range(height):
    current_colour=image[y, 0]
    cur_x=0
    for x in range(width):
        if not (x==0):
            pixel = image[y, x]

            if (np.array_equal(black, current_colour) and np.array_equal(pixel, white)):
                mid_pos = math.ceil((cur_x+x)/2)
                mid[mid_pos]+=1

            if (not np.array_equal(pixel, current_colour)) :
                current_colour=pixel
                cur_x=x
            # pre_pixel = image[y, x-1]
            # if(np.array_equal(pixel, black) and np.array_equal(pre_pixel, white)):
            #     ed2[x]+=1
            
            # if(np.array_equal(pixel, white) and np.array_equal(pre_pixel, black)):
            #     ed1[x]+=1
            



# Create x-axis values as index numbers
x = np.linspace(0, width - 1, width)


# Calculate the mean
mean = np.mean(mid)

# Calculate the standard deviation
std_dev = np.std(mid)


print("Mean:", mean)
print("Standard Deviation:", std_dev)

# Create a new array with zeros
filtered_data = np.zeros_like(mid)

# Set values greater than the standard deviation to their original values
filtered_data[mid > std_dev] = mid[mid > std_dev]

mid=filtered_data


# Define the mean and standard deviation
mean = 0  # You can adjust the mean if needed
std_deviation = 1


# Create a kernel for the normal distribution
x_no = np.arange(-5 * std_deviation, 5 * std_deviation + 1)
kernel = 1 / (np.sqrt(2 * np.pi * std_deviation**2)) * np.exp(-((x_no - mean)**2) / (2 * std_deviation**2))

# Normalize the kernel to ensure it sums to 1
kernel = kernel / np.sum(kernel)

print(kernel)

# Perform the convolution
mid = np.convolve(mid, kernel, mode='same')
#mid = np.convolve(mid, kernel, mode='same')

# Create a plot
plt.plot(x, mid)
#plt.plot(x, ed2)

# Add labels and a title
plt.xlabel('Index')
plt.ylabel('Value')
plt.title('NumPy Array Data')

plt.legend()

# Show the plot
plt.show()