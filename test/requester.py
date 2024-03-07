import requests

def upload_file(file_path):
    url = "https://d012-192-248-10-43.ngrok-free.app/upload"
    files = {'file': open(file_path, 'rb')}
    response = requests.post(url, files=files)
    return response

def download_file(url, output_directory):
    response = requests.get(url)
    if response.status_code == 200:
        filename = url.split('/')[-1]
        with open(output_directory + filename, 'wb') as f:
            f.write(response.content)
        print(f"File downloaded to {output_directory + filename}")
    else:
        print("Failed to download file")

if __name__ == "__main__":
    # Replace 'file_path' with the path to the file you want to upload
    file_path = 'SampleCSVFile_2kb.csv'
    response = upload_file(file_path)
    if response.status_code == 200:
        download_url = response.json().get('download_url')
        output_directory = './'
        download_file(download_url, output_directory)
    else:
        print("Failed to upload file",response.status_code)
