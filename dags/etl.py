from datetime import datetime, timedelta

from airflow import DAG

# from airflow.operators.bash import BashOperator
from airflow.operators.python import PythonOperator

from utils.get_data import get_data
from utils.preprocess_data import preprocess_data
from utils.train import train

default_args = {
    'owner': "Maksim Krug",
    'depends_on_past': True,
    'email': None,
    'email_on_failure': False,
    'email_on_retry': False,
    'retries': 1,
    'retry_delay': timedelta(seconds=5),
}

#instantiates a directed acyclic graph
dag = DAG(
    'ETLPipeline',
    default_args=default_args,
    description='Here is ETL Pipeline, nothing special',
    schedule=None,
    start_date=datetime(2022, 9, 20),
)

# pipeline
get_data = PythonOperator(task_id="load_data", python_callable=get_data, dag=dag)
preprocess_data = PythonOperator(task_id="preprocess_data", python_callable=preprocess_data, dag=dag)

# Dag
get_data >> preprocess_data
