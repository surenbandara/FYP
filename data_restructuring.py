from PIL import Image
import os
from tqdm import tqdm
import numpy as np
import xml.etree.ElementTree as ET

input_path = './data/'
output_path = './resize_image/'

res = int(os.environ.get("res"))
max_images = int(os.environ.get("max_images"))


def resize_img(input_path, output_path, res, max_images=1000):
    # Get a list of all image files to process
    image_files = [item for item in os.listdir(input_path) if item.endswith(".jpg")]

    # Limit the number of images to process to the first 1000
    image_files = image_files[:max_images]

    # Initialize the progress bar
    pbar = tqdm(total=len(image_files), unit="image")

    for item in image_files:
        im = Image.open(os.path.join(input_path, item))
        f, e = os.path.splitext(item)
        imResize = im.resize((res,res), Image.LANCZOS)
        imResize.save(os.path.join(output_path, f + '.jpg'), quality=90)
        pbar.update(1)  # Update the progress bar for each processed image

    pbar.close()  # Close the progress bar

# Call the function to resize the first 1000 images
resize_img(input_path, output_path, res, max_images)

print("Image resizing completed for the first 10000 images.")


# Creating an XML Parser

path = ('./resize_image')
path_xml =  ('./data')

data =[]
index=0
for file in os.listdir(path):
  if(index==max_images):
    break
  filename = os.fsdecode(file)
  if True:
       index+=1
       file_name=os.path.join(path_xml, filename.split(".")[0]+".xml")

       parser = ET.XMLParser(encoding="utf-8")
       targetTree = ET.parse(file_name, parser=parser)
       rootTag = targetTree.getroot()

       name = list(rootTag)[1].text
       depth = int(list(list(rootTag)[4])[2].text)
       width = int(list(list(rootTag)[4])[0].text)
       height = int(list(list(rootTag)[4])[1].text)

       table_xmin=[]
       table_ymin=[]
       table_xmax=[]
       table_ymax=[]

       xmin=[]
       ymin=[]
       xmax=[]
       ymax=[]

       for column in rootTag.findall('object'):
         for name in column.findall('name'):

            if(name.text == "table"):
              for bnd in column.findall('bndbox'):
                for x in bnd.findall('xmin'):
                  table_xmin.append(int(int(float(x.text))*res/width))
            elif ((name.text == "table column") or (name.text == "table row1")):
              for bnd in column.findall('bndbox'):
                for x in bnd.findall('xmin'):
                  xmin.append(int(int(float(x.text))*res/width)+2)

       for column in rootTag.findall('object'):
         for name in column.findall('name'):
            if(name.text == "table"):
              for bnd in column.findall('bndbox'):
                for x in bnd.findall('ymin'):
                  table_ymin.append(int(int(float(x.text))*res/height))
            elif ((name.text == "table column") or (name.text == "table row1")):
              for bnd in column.findall('bndbox'):
                for x in bnd.findall('ymin'):
                  ymin.append(int(int(float(x.text))*res/height)+2)


       for column in rootTag.findall('object'):
         for name in column.findall('name'):
            if(name.text == "table"):
              for bnd in column.findall('bndbox'):
                for x in bnd.findall('xmax'):
                  table_xmax.append(int(int(float(x.text))*res/width))
            elif ((name.text == "table column") or (name.text == "table row1")):
              for bnd in column.findall('bndbox'):
                for x in bnd.findall('xmax'):
                  xmax.append(int(int(float(x.text))*res/width)-1)

       for column in rootTag.findall('object'):
         for name in column.findall('name'):
            if(name.text == "table"):
              for bnd in column.findall('bndbox'):
                for x in bnd.findall('ymax'):
                  table_ymax.append(int(int(float(x.text))*res/height))
            elif (name.text == "table column" or name.text == "table row1"):
              for bnd in column.findall('bndbox'):
                for x in bnd.findall('ymax'):
                  ymax.append(int(int(float(x.text))*res/height)-1)



       my_dict = dict({'filename': file_name.split("/")[-1].split(".")[0], 'Width': res, 'Height':res, 'Table_Xmin' :table_xmin , 'Table_Xmax' : table_xmax, 'Table_Ymin' :table_ymin , 'Table_Ymax' : table_ymax , 'Depth' : depth,'xmin':xmin,'ymin':ymin,'xmax':xmax,'ymax':ymax})



       data.append(my_dict )


# Generating Masks for table and columns

for i in data:

    f=i['filename']
    print(f)

    width=i['Width']
    height=i['Height']

    xmin=i['xmin']
    ymin=i['ymin']
    xmax=i['xmax']
    ymax=i['ymax']


    column_mask = np.zeros((height, width), dtype=np.int32)

    # Loop to create column masks
    for j in range(0, len(xmin)):
      column_mask[int(ymin[j]):int(ymax[j]), int(xmin[j]):int(xmax[j])] += 125

    column_mask[column_mask < 200 ] = 0
    column_mask[column_mask >= 200] = 255

    #For edge

    #column_mask = table_mask - column_mask

    im_col = Image.fromarray(column_mask.astype(np.uint8),'L')
    im_col.save('./column_mask/'+ f + ".jpg")
