# MLEngineeringEpamCourse

## To reproduce code use dvc
```
docker build --rm -t task2 ./
docker run --rm -it task2

pip install -r requirements.txt
dvc pull
python3 get_data.py # if needed
dvc repro
```
