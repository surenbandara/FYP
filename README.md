Extraction.py
Extraction.py is a Python script for extracting data from a given source to text format. It utilizes tools like ffmpeg and Tesseract to perform data extraction.

Usage
To use Extraction.py, follow the steps below:

Set Up Python Environment:

First, create a Python virtual environment. You can use Miniconda to manage the environment.

bash
Copy code
conda create --name extraction_env python=3.8
Install Requirements:

Activate the created environment and install the required packages listed in the requirements.txt file.

bash
Copy code
conda activate extraction_env
pip install -r requirements.txt
Install ffmpeg and Tesseract:

Install ffmpeg and Tesseract using Miniconda's package manager.

bash
Copy code
conda install -c main ffmpeg
conda install -c conda-forge tesseract
Use Extraction Class:

You can use the Extraction class to extract data from a given path. Instantiate the class and call the relevant methods with the desired input parameters.

python
Copy code
from Extraction import Extraction

# Instantiate the Extraction class
extractor = Extraction()

# Use the extract_data method to extract data from the given path
extractor.extract_data(input_path, output_format)
Replace input_path with the path or URL of the source from which data needs to be extracted, and output_format with the desired format of the output data.

Test with test.py:

You can use test.py to test different kinds of sources. Modify test.py as needed and run it to test the data extraction process with various input sources.

bash
Copy code
python test.py