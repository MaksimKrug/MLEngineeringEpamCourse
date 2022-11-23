import os

from google_drive_downloader import GoogleDriveDownloader as gdd


def get_data():
    if not os.path.exists("data"): 
        os.makedirs("data")
    gdd.download_file_from_google_drive(file_id='1ObdHqQuudl2O9gUnfD8ouD34CXZLMTkm',
                                        dest_path="./data/jigsaw-toxic-comment-train.csv",
                                        unzip=False)
