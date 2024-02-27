from io import BytesIO
from img2table.document import PDF

import os
import fitz
import pytesseract
from pdf2image import convert_from_path
from img2table.ocr import TesseractOCR
from img2table.document import Image
from img2table.document import pdf
from tabulate import tabulate
import json
import pandas as pd
import re
import fitz  # PyMuPDF
from tabulate import tabulate
import csv
import whisper
from fpdf import FPDF

from docx2pdf import convert

from spire.presentation import *
from spire.presentation.common import *


def init():
    create_folder("temp")

#Utilz

def create_folder(folder_path):
    try:
        os.makedirs(folder_path)
        print(f"Folder created at {folder_path}")
    except FileExistsError:
        print(f"Folder already exists at {folder_path}")

def delete_folder(folder_path):
    try:
        os.rmdir(folder_path)
        print(f"Folder deleted at {folder_path}")
    except FileNotFoundError:
        print(f"Folder not found at {folder_path}")

def delete_file(file_path):
    
    try:
        os.remove(file_path)
        print(f"File deleted at {file_path}")
    except FileNotFoundError:
        print(f"File not found at {file_path}")

#Doc extractor

def convert_docx_to_pdf(docx_file, pdf_file):
    convert(docx_file, pdf_file)

def doc_extractor(doc_path,output):
    name = doc_path.split("\\")[-1].split(".")[0]+".pdf"
    convert_docx_to_pdf(doc_path, "temp\\"+name)
    pdf_extractor("temp\\"+name ,output)
    delete_file("temp\\"+name)
 

#PPTX extractor

def convert_pptx_to_pdf(pptx_path, pdf_file):

    # Create a Presentation object
    presentation = Presentation()
    presentation.LoadFromFile(pptx_path)

    # Convert the presentation to PDF format
    presentation.SaveToFile(pdf_file, FileFormat.PDF)
    presentation.Dispose()
    print("Convertion done")

def pptx_extractor(pptx_path,output):
    name = pptx_path.split("\\")[-1].split(".")[0]+".pdf"
    convert_pptx_to_pdf(pptx_path, "temp\\"+name)
    pdf_extractor("temp\\"+name ,output)
    delete_file("temp\\"+name)


#Audio extractor

def audio_extractor(audio_path, output):
    model = whisper.load_model("base")
    result = model.transcribe(audio_path)

    with open(output, 'w', encoding="utf-8") as file:
            file.write(result)

#CSV extractor
def read_csv_and_format(csv_file):
    # Read CSV file
    with open(csv_file, 'r') as file:
        reader = csv.reader(file)
        data = list(reader)

    # Format data using tabulate
    table = tabulate(data, headers="firstrow", tablefmt="grid")

    # Print formatted table
    return table

def csv_extractor(csv_path ,output):
    table = read_csv_and_format(csv_path)

    with open(output, 'w', encoding="utf-8") as file:
            file.write(table)

#Pdf extractor
    
def pdf_page_to_json(doc, page_number):
    
    # Get the specified page
    page = doc.load_page(page_number - 1)
    
    # Extract the page text
    text = page.get_text('json')
    return text

def replace_substring(original_string, start_index, new_substring):

    before_substring = original_string[:start_index]
    after_substring = original_string[start_index + len(new_substring):]
    new_string = before_substring + new_substring + after_substring
    return new_string

def json_to_text(json_data,wid_fac):
            height = json_data['height']
            width = json_data['width']

            factor_hei = 75/height
            factor_wid =wid_fac

        
            page =[]
            for hei in range(int(height*factor_hei)):
                line = {"starter":False,"fatcor": factor_wid, "data":" "*int(width*factor_wid)}
                page.append(line)
            
            for block in json_data['blocks']: 
                try:
                    for line in block['lines']:
                        for span in line['spans']:
                            x1, y1 = span['origin']
                        
                            y1 = y1*factor_hei
                            x1 = x1*factor_wid

                            
                            page[int(y1)]["data"] = replace_substring(page[int(y1)]["data"], int(x1), span['text'])
                except Exception as e: 
                    print("error",e)

            
            created_page =""
            for line in page:
                created_page+=line["data"]+"\n"
            

            return created_page

def squ_ava_fac(json_data):
            fac = 0
            nu = 0
            for block in json_data['blocks']: 
                try:
                    for line in block['lines']:
                        for span in line['spans']:
                            bx1,by1,bx2,by2 = span['bbox']
                            fac+=(len(span['text'])/(bx2-bx1))**2
                            nu+=1
                except Exception as e: 
                    print("error",e)

            ava=0

            if(nu!=0):
        
                ava=fac/nu
        
            

            return ava
                    
def replace_from_table(json_data,cordinates,tables , n):
    
    for i in range(n):
        cordinate = cordinates[i]
        table = tables[i]
        delete_block =[]

        ycordinate = []
        for block_ind in range(len(json_data['blocks'])):
                block = json_data['blocks'][block_ind]
                try:
                    for line in block['lines']:      
                        for span in line['spans']:
                            
                                x1, y1, x2, y2 = span['bbox']
                                x1, y1, x2, y2 = x1*2.77, y1*2.77, x2*2.77, y2*2.77
                                if(((cordinate[0]-10<=x1 and x2<=cordinate[2]+10)) and ((cordinate[1]-10<=y1 and y2<=cordinate[3]+10))):
                                    ycordinate.append(y1/2.77)
                                    delete_block.append(block_ind)

                except:
                    print("Error")

        delete_block = list(set(delete_block))
        delete_block.sort(reverse=True)


        for del_ind in delete_block:
            del json_data['blocks'][del_ind]
        

        
        table_lis = table.split("\n")
        ycordinate = list(set(ycordinate))
        ycordinate.sort()

        sum = 0
        for i in range(1,len(ycordinate)):
            sum += ycordinate[i]- ycordinate[i-1]
        space = sum/(len(ycordinate)-1)

        for row_table in range(len(table_lis)):
            
            new_block = {"lines" : [{"spans" : [{"origin" : [cordinate[0]/2.77,cordinate[1]/2.77+row_table*space] ,"text" : table_lis[row_table]}]}]}
            json_data['blocks'].append(new_block)
        

    return json_data

def tabulate_converter(json_data):
    # Convert JSON to a list of dictionaries

    if type(json_data) == list:
        table_data = [list(record.values()) for record in json_data]
        # Get headers from the first record
        headers = list(json_data[0].keys())

    elif type(json_data) == dict:
        table_data = list(json_data.values())

        # Get headers from the first record
        headers = list(json_data.keys())

    elif type(json_data) == str:
        
        # Convert the JSON string to a list of dictionaries
        json_data = json.loads(json_data)

        table_data = [list(record.values()) for record in json_data]
        # Get headers from the first record
        headers = list(json_data[0].keys())


    # Convert to Markdown table
    markdown_table = tabulate(table_data, headers, tablefmt="pipe")

    return markdown_table

def table_titles(tables_dict):
    titles = []
    for t in tables_dict:
        if tables_dict[t] != []:
            t_index=1
            for table in tables_dict[t]:
                titles.append([(t+1, t_index),table.title])
                t_index+=1

    return titles

def table_to_json_based_xlsx(xlsx_path, titles):
    xlsx_tables = []
    xlsx_file = pd.ExcelFile(xlsx_path)  # Replace with your file path
    sheet_names = xlsx_file.sheet_names

    for sheet_name in sheet_names:
        df = pd.read_excel(xlsx_file, sheet_name=sheet_name)

        match = re.search(r"Page (\d+) - Table (\d+)", sheet_name)
        if match:
            page_number, table_number = map(int, match.groups())
        else:
            page_number=''
            table_number=''

        if page_number!='' and table_number!='':
            title=[k[1] for k in titles if k[0]==(page_number, table_number)][0]

        else:
            title=''

        if df.empty:
            temp = [{}]
            for k in range(len(df.columns)):
                try:
                    temp[0][df.columns[k]] = ''
                except:
                    continue

            xlsx_tables.append({'info': (page_number, table_number), 'title': title, 'data': json.dumps(temp)})
        else:
            xlsx_tables.append({'info': (page_number, table_number), 'title': title, 'data': df.to_json(orient="records")})

    return xlsx_tables

def table_extractor(pdf_path,xlsx_path,page_number):

    pdf = PDF(pdf_path,
                detect_rotation=False,
                pdf_text_extraction=True)

    pdf.to_xlsx(xlsx_path,
                    ocr=ocr,
                    implicit_rows=True,
                    borderless_tables=True,
                    min_confidence=90)

    extracted_tables = pdf.extract_tables(ocr=ocr,
                                implicit_rows=True,
                                borderless_tables=True,
                                min_confidence=50)
    

    
    lo=[]
    tables = []
    
    for i in extracted_tables[page_number]:
        bbox=i.bbox
        location =  [bbox.x1,bbox.y1,bbox.x2,bbox.y2]  
        lo.append(location)

        titles = table_titles(extracted_tables)
        tables.append(table_to_json_based_xlsx(xlsx_path, titles)[0])

    return tables,lo

def pdf_extractor(pdf_path ,output):

    doc = fitz.open(pdf_path)

    txt_content = """"""

    get_ava =0 
    ind = 0

    json_data_list = []

    for page_number in range(len(doc)):
        json_data=pdf_page_to_json(doc, page_number)
        json_data = json.loads(json_data)
        json_data_list.append(json_data)
        ava = squ_ava_fac(json_data)
        if(ava!=0) :
            get_ava+=ava
            ind+=1

        
    wid_fac = (get_ava/ind)**0.5

    print(wid_fac)

    for json_data in json_data_list:
        txt = json_to_text(json_data,wid_fac)

        txt_content +=txt


    with open(output, 'w', encoding="utf-8") as file:
            file.write(txt_content)
     

#ocr = TesseractOCR(n_threads=1, lang="eng")
csv_path = "SampleCSVFile_2kb.csv"
doc_path = "AutoRecovery save of Document1.docx"
pdf_path = "fssr_2013e.pdf"
pptx_path = "Front Page.pptx"
audio_path = "F:\FYP\data-extraction-p3\new_\engm1.wav"
output = "example.txt"


init()

#csv_extractor(csv_path,output)
#pdf_extractor(pdf_path,output)
#doc_extractor(doc_path,output)
#pptx_extractor(pptx_path,output)
audio_extractor(audio_path,output)