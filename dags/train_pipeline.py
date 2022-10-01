from datetime import datetime, timedelta

from airflow import DAG

# from airflow.operators.bash import BashOperator
from airflow.operators.python import PythonOperator

from utils.train import train

default_args = {
    'owner': "Maksim Krug",
    'depends_on_past': False,
    'email': None,
    'email_on_failure': False,
    'email_on_retry': False,
    'retries': 1,
    'retry_delay': timedelta(seconds=5),
}

#instantiates a directed acyclic graph
dag = DAG(
    'TrainPipeline',
    default_args=default_args,
    description='Here is a Trian Pipeline, nothing special',
    schedule=None,
    start_date=datetime(2022, 9, 20),
)

# pipeline
train = PythonOperator(task_id="train", python_callable=train, dag=dag) 

# Dag
train
