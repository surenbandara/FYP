from flask import Flask, request, jsonify
import os

from new_.FYP.extractor import Extractor

app = Flask(__name__)

# Define the upload folder
UPLOAD_FOLDER = 'uploads'
if not os.path.exists(UPLOAD_FOLDER):
    os.makedirs(UPLOAD_FOLDER)

# Configure upload folder for Flask app
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

extractor= Extractor()

@app.route('/upload', methods=['POST'])
def upload_file():
    # Check if the POST request has the file part
    if 'file' not in request.files:
        return jsonify({'error': 'No file part'})

    file = request.files['file']

    # If the user does not select a file, the browser submits an empty file without a filename
    if file.filename == '':
        return jsonify({'error': 'No selected file'})

    # Save the uploaded file
    filename = os.path.join(app.config['UPLOAD_FOLDER'], file.filename)
    file.save(filename)

    return jsonify({'message': 'File uploaded successfully', 'txt': extractor.extractor(os.path.join(app.config['UPLOAD_FOLDER'], filename))})

if __name__ == '__main__':
    app.run(debug=True)
