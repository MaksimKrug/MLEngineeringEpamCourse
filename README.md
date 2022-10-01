# MLEngineeringEpamCourse (Task 4)

AirFlow Server is here: http://localhost:8080/

Here is a small example of Airflow pipelines:
1) etl.py - extract, transform and save the data;
2) train.py - train Random Forest model and save the model and vectors (tf-idf);
3) predict.py - chose random text from preprocessed data and predict label for this text

Commands
```
mldir ./dags ./logs ./plugins ./data
docker-compose up airflow-init && docker-compose up --build
```
