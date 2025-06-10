from airflow import DAG
from airflow.operators.python import PythonOperator
from airflow.operators.empty import EmptyOperator
from airflow.operators.bash import BashOperator
from hiring_functions import create_folders, split_data, preprocess_and_train, gradio_interface
from datetime import datetime

# Se inicializa un DAG 

with DAG(
    dag_id = 'hiring_lineal', # dag_id para reconocer dag
    start_date = datetime(2024,10,1), # fecha de inicio
    catchup = False, # sin backfill
    schedule = None # ejecucion manual
    ) as dag:

    # Task 1: marcador de posicion que indique el incio del pipeline
    task_inicio_pipeline = EmptyOperator(task_id='inicio_del_pipeline') 

    # Task 2: Se crean las carpetas usando create_folder()
    task_create_folder = PythonOperator(
        task_id='crear_carpetas',
        python_callable=create_folders
        )
    
    # Task 3: descargar data_1.csv
    fecha = datetime.now().date()
    task_download_dataset = BashOperator(
        task_id='descargar_data_1.csv',
        bash_command=
        "curl -o " 
        "/root/airflow/{{ds}}/raw/data_1.csv "
        "https://gitlab.com/eduardomoyab/laboratorio-13/-/raw/main/files/data_1.csv"
    )

    # Task 4: se aplica split_data()

    task_split_data = PythonOperator(
    task_id='split_data',
    python_callable=split_data
    )

    # Task 5: se aplica el preprocesamiento y entrenamiento con preprocess_and_Train()
    task_prep_train = PythonOperator(
    task_id='preprocesamiento_train',
    python_callable=preprocess_and_train
    )

    # Task 6: se monta la interfaz en gradio 
    task_gradio = PythonOperator(
        task_id = 'interfaz_gradio',
        python_callable = gradio_interface
    )

    # Se define el orden de las tareas
    task_inicio_pipeline >> task_create_folder >> task_download_dataset >> task_split_data >> task_prep_train >> task_gradio