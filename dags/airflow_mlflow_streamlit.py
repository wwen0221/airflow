from airflow import DAG
from airflow.operators.python import PythonOperator
from datetime import datetime
from airflow.contrib.sensors.file_sensor import FileSensor
from zipfile import ZipFile
import sys
import os

sys.path.append('/Users/WW/airflow')
import projects.airflow_mlflow_streamlit.image_classification.main_util as main_util


default_args = {
    'owner': 'WW',
    'depends_on_past': False,
    'email': ['swwen5148@gmail.com'],
    'email_on_failure': False,
    'email_on_retry': False,
    'retries': 1,
    # 'queue': 'bash_queue',
    # 'pool': 'backfill',
    # 'priority_weight': 10,
    # 'end_date': datetime(2016, 1, 1),
    # 'wait_for_downstream': False,
    # 'dag': dag,
    # 'sla': timedelta(hours=2),
    # 'execution_timeout': timedelta(seconds=300),
    # 'on_failure_callback': some_function,
    # 'on_success_callback': some_other_function,
    # 'on_retry_callback': another_function,
    # 'sla_miss_callback': yet_another_function,
    # 'trigger_rule': 'all_success'
}


dag= DAG('tutorial',
    default_args=default_args,
    description='A simple tutorial DAG',
    schedule_interval=None,
    start_date=datetime(2023, 10, 12),
    catchup=False,
    tags=['example']
         )
    
unzip_file_operator = PythonOperator(
    task_id = 'unzip_dataset',
    python_callable = main_util.unzip_file()
)

train_model_operator = PythonOperator(
    task_id= 'model_trainer',
    python_callable = main_util.train_model()
)

file_sensing_task = FileSensor(task_id='sense_the_zipfile',
                                filepath='/Users/WW/airflow/projects/airflow_mlflow_streamlit/dataset/cards/zip_files/*.zip',
                                poke_interval=10,
                                dag=dag)    

file_sensing_task >> unzip_file_operator >> train_model_operator
