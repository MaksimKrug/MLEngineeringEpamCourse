# MLEngineeringEpamCourse

## To reproduce code use dvc
```
docker build --rm -t task2 ./
docker run --rm -it task2

dvc pull
python3 get_data.py # if needed
dvc repro
```
