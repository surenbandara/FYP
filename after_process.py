import cv2
import numpy as np
import matplotlib.pyplot as plt
import math
import matplotlib.image as mpimg
from PIL import Image
from paddleocr import PaddleOCR,draw_ocr
import os

ocr = PaddleOCR(use_angle_cls=True, lang='en' , use_gpu=False) # need to run only once to download and load model into memory

name = 'image0.jpg'
image_path = "./eval_output/row/"+name

res=640

# Load the image
image = cv2.imread(image_path)

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


mid=np.zeros(width)

white = [255, 255, 255]
black = [0, 0, 0]


for x in range(width):
    current_colour=image[0, x]
    cur_y=0
    check=False
    for y in range(height):
        if not (x==0):
            pixel = image[y, x]

            if (np.array_equal(black, current_colour) and np.array_equal(pixel, white)):
                if not check:
                    check=True
                else:
                    mid_pos = math.ceil((cur_y+y)/2)
                    mid[mid_pos]+=1


            if (not np.array_equal(pixel, current_colour)) :
                current_colour=pixel
                cur_y=y
            
            


def filtering(mid ,ratio=1):
    # Calculate the mean
    mean = np.mean(mid)
    # Calculate the standard deviation
    std_dev = np.std(mid)*ratio


    print("Mean:", mean)
    print("Standard Deviation:", std_dev)

    # Create a new array with zeros
    filtered_data = np.zeros_like(mid)

    # Set values greater than the standard deviation to their original values
    filtered_data[mid > std_dev] = mid[mid > std_dev]

    return filtered_data

def conv(mid , std ,mean ,steps):
    # Define the mean and standard deviation
    mean = mean  # You can adjust the mean if needed
    std_deviation = std


    # Create a kernel for the normal distribution
    x_no = np.arange(-steps * std_deviation, steps * std_deviation + 1)
    kernel = 1 / (np.sqrt(2 * np.pi * std_deviation**2)) * np.exp(-((x_no - mean)**2) / (2 * std_deviation**2))

    # Normalize the kernel to ensure it sums to 1
    kernel = kernel / np.sum(kernel)

    print(kernel)

    # Perform the convolution
    mid = np.convolve(mid, kernel, mode='same')

    return mid


def column_indexes(data):
    ind=[]
    max_val=0
    for i in range(1,len(data)):
        if data[i]==0:
            max_val=0
        if (data[i-1]==0 and data[i]!=0):
            ind.append(i)
        if data[i]!=0:
            if max_val<data[i]:
                max_val=data[i]
                ind[-1]=i


    print("Indices of Maximum Values within Non-Zero Segments:", ind)

    return ind

# Calculate the new length (original length + 2 * (original length - 1))
new_length =mid.size + 2 * (mid.size - 1)

# Create a new array with zeros
stretched_array = np.zeros(new_length)

# Copy the values from the original array to the new array with spacing
stretched_array[::3] = mid

mid=stretched_array

# mid=conv(mid ,8,0,10)
# mid=filtering(mid)
# mid=conv(mid ,5,0,15)
# mid=filtering(mid)
# mid=conv(mid ,6,0,15)
# mid=filtering(mid)

# mid=filtering(mid ,ratio=0.5)


mid=conv(mid ,8,0,10)
mid=filtering(mid,ratio=1)
mid=filtering(mid,ratio=1)
mid=conv(mid ,6,0,15)
mid=filtering(mid ,ratio=1)


horizontal_positions=column_indexes(mid)

horizontal=[]

for i in horizontal_positions:
    horizontal.append(int(i/3))


######################################################################################



image_path = "./eval_output/column/"+name


# Load the image
image = cv2.imread(image_path)

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


mid=np.zeros(width)

white = [255, 255, 255]
black = [0, 0, 0]

for y in range(height):
    current_colour=image[y, 0]
    cur_x=0
    check=False
    for x in range(width):
        if not (x==0):
            pixel = image[y, x]

            if (np.array_equal(black, current_colour) and np.array_equal(pixel, white)):

                if not check:
                    check=True
                else:
                    mid_pos = math.ceil((cur_x+x)/2)
                    mid[mid_pos]+=1

            if (not np.array_equal(pixel, current_colour)) :
                current_colour=pixel
                cur_x=x
            

# mid=conv(mid ,6,0,10)
# mid=filtering(mid)
# mid=conv(mid ,4,0,10)
# mid=filtering(mid)
# mid=conv(mid ,5,0,7)
# mid=filtering(mid)

mid=filtering(mid,ratio=1.5)
mid=conv(mid ,8,0,3)

mid=conv(mid ,7,0,5)
mid=filtering(mid)

mid=conv(mid ,5,0,7)
mid=filtering(mid)


vertical_positions=column_indexes(mid)

##########################################################################
# Load the image
original_image = cv2.imread('./input/'+name)

# Get the size of the image
height, width, channels = original_image.shape

ratio = max(height,width)/640
img_x=[0]
for x in vertical_positions:
    img_x.append(int(x*ratio))
img_x.append(width-1)

img_y=[0]
for y in horizontal:
    img_y.append(int(y*ratio))
img_y.append(height-1)


print(img_x ,  vertical_positions ,width)
print(img_y , horizontal , height)


def ocr_cropping(x1,x2,y1,y2,original_image):
    global ocr
    # Crop the ROI from the original image
    cropped_image = original_image[y1:y2 ,x1:x2]
    try:
        cropped_image_rgb = cv2.cvtColor(cropped_image, cv2.COLOR_BGR2RGB)

        # Convert to a PIL image
        #pil_image = Image.fromarray(cropped_image_rgb)

        # Save the cropped cv2 image as a temporary file
        cv2.imwrite("temp_image.png", cropped_image_rgb)

        result = ocr.ocr("temp_image.png", cls=True)
    
        for idx in range(len(result)):
            res = result[idx]
            for line in res:
                row.append(list(line[-1])[0])

        # Remove the temporary image file
        os.remove('temp_image.png')
    except:
        None

    

table=[]
for y in range(0,len(img_y)-1):
    row=[]
    for x in range(0,len(img_x)-1):
        ocr_cropping(img_x[x],img_x[x+1],img_y[y],img_y[y+1],original_image)
    table.append(row)

print(table)
