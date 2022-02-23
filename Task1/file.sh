# install miniconda
apt-get update
apt-get install -y wget
wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh
chmod +x Miniconda3-latest-Linux-x86_64.sh
./Miniconda3-latest-Linux-x86_64.sh -b -u -p /miniconda3
# add miniconda to path
export PATH="/miniconda3/bin:$PATH"

# create environment from .yml file
conda env create -f environment.yml
# nltk download
mkdir ../data
source activate my_env
python -c "import nltk;nltk.download('wordnet', download_dir='/miniconda3/envs/my_env/nltk_data/'); nltk.download('omw-1.4', download_dir='/miniconda3/envs/my_env/nltk_data/')"
# data download
gdown --id 1JEXtZMcxOUA4kpWmF6n6sx1Jry4iNVcX
mv jigsaw-toxic-comment-train.csv ../data/
