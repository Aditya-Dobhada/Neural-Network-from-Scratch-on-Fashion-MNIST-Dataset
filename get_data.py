from zipfile import ZipFile
import os
import urllib
import urllib.request


URL = 'https://nnfs.io/datasets/fashion_mnist_images.zip'  # URL of the zip file to download
FILE = 'fashion_mnist_images.zip'  # Name of the zip file to save locally
FOLDER = 'fashion_mnist_images'  # Folder where the contents of the zip file will be extracted

# Check if the zip file is not already downloaded
if not os.path.isfile(FILE):
    print(f'Downloading {URL} and saving as {FILE}....')
    # If not downloaded, download the file and save it locally
    urllib.request.urlretrieve(URL, FILE)

# Inform that unzipping process is starting
print('Unzipping images...')
# Open the downloaded zip file for reading
with ZipFile(FILE) as zip_images:
    # Extract all contents of the zip file to the specified folder
    zip_images.extractall(FOLDER)

# Inform that the process is complete
print('Done!')