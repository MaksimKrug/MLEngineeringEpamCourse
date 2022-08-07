import gdown

# download a file
url = "https://drive.google.com/uc?id=1sGu6O1z9lj2cSvk4yrJJ4aftiWtSfTwj"
output = "data/jigsaw-toxic-comment-train.csv"
gdown.download(url, output, quiet=False)