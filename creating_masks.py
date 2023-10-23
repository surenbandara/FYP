from PIL import Image
import os
from tqdm import tqdm
import numpy as np
import xml.etree.ElementTree as ET


res = int(os.environ.get("res" ,640))
max_images = int(os.environ.get("max_images" ,100000))
padding = int(os.environ.get("padding" ,4))

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

       column_xmin=[]
       column_ymin=[]
       column_xmax=[]
       column_ymax=[]

       row_xmin=[]
       row_ymin=[]
       row_xmax=[]
       row_ymax=[]

       for column in rootTag.findall('object'):
         for name in column.findall('name'):

            if(name.text == "table"):
              for bnd in column.findall('bndbox'):
                for x in bnd.findall('xmin'):
                  table_xmin.append(int(int(float(x.text))*res/width))
            elif (name.text == "table column"):
              for bnd in column.findall('bndbox'):
                for x in bnd.findall('xmin'):
                  column_xmin.append(int(int(float(x.text))*res/width)+padding)
            
            elif (name.text == "table row"):
              for bnd in column.findall('bndbox'):
                for x in bnd.findall('xmin'):
                  row_xmin.append(int(int(float(x.text))*res/width)+padding)

       for column in rootTag.findall('object'):
         for name in column.findall('name'):
            if(name.text == "table"):
              for bnd in column.findall('bndbox'):
                for x in bnd.findall('ymin'):
                  table_ymin.append(int(int(float(x.text))*res/height))
            elif (name.text == "table column"):
              for bnd in column.findall('bndbox'):
                for x in bnd.findall('ymin'):
                  column_ymin.append(int(int(float(x.text))*res/height)+padding)
            
            elif (name.text == "table row"):
              for bnd in column.findall('bndbox'):
                for x in bnd.findall('ymin'):
                  row_ymin.append(int(int(float(x.text))*res/height)+padding)


       for column in rootTag.findall('object'):
         for name in column.findall('name'):
            if(name.text == "table"):
              for bnd in column.findall('bndbox'):
                for x in bnd.findall('xmax'):
                  table_xmax.append(int(int(float(x.text))*res/width))
            elif (name.text == "table column"):
              for bnd in column.findall('bndbox'):
                for x in bnd.findall('xmax'):
                  column_xmax.append(int(int(float(x.text))*res/width)-padding)
            
            elif (name.text == "table row"):
              for bnd in column.findall('bndbox'):
                for x in bnd.findall('xmax'):
                  row_xmax.append(int(int(float(x.text))*res/width)-padding)

       for column in rootTag.findall('object'):
         for name in column.findall('name'):
            if(name.text == "table"):
              for bnd in column.findall('bndbox'):
                for x in bnd.findall('ymax'):
                  table_ymax.append(int(int(float(x.text))*res/height))
            elif (name.text == "table column"):
              for bnd in column.findall('bndbox'):
                for x in bnd.findall('ymax'):
                  column_ymax.append(int(int(float(x.text))*res/height)-padding)
            
            elif (name.text == "table row"):
              for bnd in column.findall('bndbox'):
                for x in bnd.findall('ymax'):
                  row_ymax.append(int(int(float(x.text))*res/height)-padding)



       my_dict = dict({'filename': file_name.split("/")[-1].split(".")[0], 'Width': res, 'Height':res, 'Table_Xmin' :table_xmin , 'Table_Xmax' : table_xmax, 'Table_Ymin' :table_ymin , 'Table_Ymax' : table_ymax , 'Depth' : depth,
                       'column_xmin':column_xmin,'column_ymin':column_ymin,'column_xmax': column_xmax,'column_ymax': column_ymax,
                       'row_xmin':row_xmin,'row_ymin':row_ymin,'row_xmax': row_xmax,'row_ymax': row_ymax})



       data.append(my_dict )


# Generating Masks for table and columns

# Initialize the progress bar
for item in tqdm(data, desc="Processing"):

    f=i['filename']

    width=i['Width']
    height=i['Height']

    column_xmin=i['column_xmin']
    column_ymin=i['column_ymin']
    column_xmax=i['column_xmax']
    column_ymax=i['column_ymax']

    row_xmin=i['row_xmin']
    row_ymin=i['row_ymin']
    row_xmax=i['row_xmax']
    row_ymax=i['row_ymax']

    column_mask = np.zeros((height, width), dtype=np.int32)
    row_mask = np.zeros((height, width), dtype=np.int32)

    # Loop to create column masks
    for j in range(0, len(column_xmin)):
      column_mask[int(column_ymin[j]):int(column_ymax[j]), int(column_xmin[j]):int(column_xmax[j])] =255
    
    for i in range(0, len(row_xmin)):
      row_mask[int(row_ymin[i]):int(row_ymax[i]), int(row_xmin[i]):int(row_xmax[i])] =255


    im_col = Image.fromarray(column_mask.astype(np.uint8),'L')
    im_row = Image.fromarray(row_mask.astype(np.uint8),'L')

    im_col.save('./column_mask/'+ f + ".jpg")
    im_row.save('./row_mask/'+ f + ".jpg")


print("Resizing and masking is done!")