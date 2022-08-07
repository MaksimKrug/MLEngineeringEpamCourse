import gdown

# download a file
url = "https://drive.google.com/file/d/1aoy_Pf2bQys7zut1ecO8ReJoFEGFJKWz/view?usp=sharing"
output = "data/jigsaw-toxic-comment-train.csv"
gdown.download(url, output, quiet=False)