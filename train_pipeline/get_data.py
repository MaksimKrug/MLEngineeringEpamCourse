import gdown
import os

# download a file
url = "https://drive.google.com/uc?id=1sGu6O1z9lj2cSvk4yrJJ4aftiWtSfTwj"
if not os.path.exists("data"):
    os.makedirs("data")
output = "data/jigsaw-toxic-comment-train.csv"
gdown.download(url, output, quiet=False) 
