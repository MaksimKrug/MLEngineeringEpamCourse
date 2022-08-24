## Task
Run docker and execute notebook Example.ipynb. As a result you will have in mounted directory nes file - test.csv.

## Repository structure
Dockerfile - file with commands for build a docker container
file.sh - bash scripts for docker container (install dependencies, load data and ext.)
environment.yml - environment for conda
Example.ipynb - jupyter notebook with main script

## Docker commands
These commands will create docker container
1) Build docker container with name text_classification and new user. All commands with comments can be founded in Dockerfile
``` 
docker build --rm -t text_classification --build-arg USER_ID=$(id -u) --build-arg GROUP_ID=$(id -g) ./ 
```
2) Run docker container and mount current directory to workdir. Port for jupyter Notebook. Interactive mode.
```
docker run --rm -v $(pwd):/workdir -p 8888:8888 -it text_classification
```

## Pipeline inside docker contatiner
You need to execute Example.ipynb file and as a result you will have "preds.csv" file with predicts for test data. This file is completely useless and need just for homework.

1) Run docker container
2) Connect to http://localhost:8888/
3) Log in with password