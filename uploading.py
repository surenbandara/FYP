from googleapiclient.discovery import build
from google.oauth2 import service_account
from googleapiclient.http import MediaFileUpload

def upload_files_to_google_drive(credentials_file, file_paths, folder_id=None):
    try:
        # Authenticate using your credentials
        credentials = service_account.Credentials.from_service_account_file(
            credentials_file, scopes=['https://www.googleapis.com/auth/drive.file']
        )

        # Build the Google Drive API service
        drive_service = build('drive', 'v3', credentials=credentials)

        for file_path in file_paths:
            # Prepare file metadata
            file_name = file_path.split('/')[-1]  # Get the file name from the path
            file_metadata = {
                'name': file_name,  # Name of the file on Google Drive
                'parents': [folder_id] if folder_id else []  # ID of the parent folder (optional)
            }

            # Upload the file
            media = MediaFileUpload(file_path, mimetype='application/gzip')
            uploaded_file = drive_service.files().create(body=file_metadata, media_body=media, fields='id').execute()

            print(f'File "{file_name}" uploaded to Google Drive with ID: {uploaded_file.get("id")}')

    except Exception as e:
        print(f'An error occurred: {e}')

# Example usage:
credentials_file = 'credentials.json'
file_paths = str(input('Input compressioned files')).split(",")
folder_id = '1QNS_1mUtJcRzjLCdkQnvCM9egouhynlP'  # Optional: Specify the folder ID
upload_files_to_google_drive(credentials_file, file_paths, folder_id)

