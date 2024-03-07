# Extraction.py

Extraction.py is a Python script for extracting data from a given source to text format. It utilizes tools like ffmpeg and Tesseract to perform data extraction.

## Usage

To use Extraction.py, follow the steps below:

1. **Set Up Python Environment:**

   First, create a Python virtual environment. You can use Miniconda to manage the environment.

   ```bash
   conda create --name extraction_env python=3.8

2. **Install Requirements:**

    Activate the created environment and install the required packages listed in the requirements.txt file.

    ```bash
    conda activate extraction_env
    pip install -r requirements.txt

3. **Install ffmpeg and Tesseract:**

    Install ffmpeg and Tesseract using Miniconda's package manager.

    ```bash
    conda install -c main ffmpeg
    conda install -c conda-forge tesseract

4. **Use Extraction Class:**

    You can use the Extraction class to extract data from a given path. Instantiate the class and call the relevant methods with the desired input parameters.

    ```python
    from Extraction import Extraction

    # Instantiate the Extraction class
    extractor = Extraction()

    # Use the `extract_data` method to extract data from the given path
    extractor.extract_data(input_path, output_format)
    Replace input_path with the path or URL of the source from which data needs to be extracted, and output_format with the desired format of the output data.

5. **Test with test.py:**

    You can use test.py to test different kinds of sources. Modify test.py as needed and run it to test the data extraction process with various input sources.

    ```bash
    python test.py
