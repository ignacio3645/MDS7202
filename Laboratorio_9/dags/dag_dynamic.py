from airflow import DAG
from airflow.operators.python import PythonOperator, BranchPythonOperator
from airflow.operators.empty import EmptyOperator
from airflow.operators.bash import BashOperator
from airflow.utils.trigger_rule import TriggerRule
from datetime import datetime
from hiring_dynamic_functions import create_folders, load_and_merge, split_data, train_model, evaluate_models
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier

def branching_logic(**kwargs):
    # Obtener la fecha de ejecución como string YYYY-MM-DD
    execution_date = kwargs['ds']
    fecha_actual = datetime.strptime(execution_date, "%Y-%m-%d").date()

    # Comparar con 1 de noviembre de 2024
    if fecha_actual < datetime(2024, 11, 1).date():
        return 'descargar_data_1'
    else:
        return 'descargar_data_1_y_2'

# DAG definición
with DAG(
    dag_id='hiring_dynamic_pipeline',
    start_date=datetime(2024, 10, 1),
    schedule_interval='0 15 5 * *',  # Cada día 5 a las 15:00 UTC
    catchup=True
) as dag:

    # 1. Inicio
    start = EmptyOperator(task_id='inicio_pipeline')

    # 2. Crear carpetas
    crear_carpetas = PythonOperator(
        task_id='crear_estructura_directorios',
        python_callable=create_folders
    )

    # 3. Branching: decisión qué archivos descargar
    branching = BranchPythonOperator(
        task_id='branching_data',
        python_callable=branching_logic,
        provide_context=True
    )

    # 4. Descargar solo data_1
    descargar_data_1 = BashOperator(
        task_id='descargar_data_1',
        bash_command=(
            "mkdir -p /root/airflow/{{ ds }}/raw && "
            "curl -s -o /root/airflow/{{ ds }}/raw/data_1.csv "
            "https://gitlab.com/eduardomoyab/laboratorio-13/-/raw/main/files/data_1.csv"
        )
    )

    # 5. Descargar data_1 y data_2
    descargar_data_1_y_2 = BashOperator(
        task_id='descargar_data_1_y_2',
        bash_command=(
            "mkdir -p /root/airflow/{{ ds }}/raw && "
            "curl -s -o /root/airflow/{{ ds }}/raw/data_1.csv "
            "https://gitlab.com/eduardomoyab/laboratorio-13/-/raw/main/files/data_1.csv && "
            "curl -s -o /root/airflow/{{ ds }}/raw/data_2.csv "
            "https://gitlab.com/eduardomoyab/laboratorio-13/-/raw/main/files/data_2.csv"
        )
    )

    # 6. Unir los archivos
    unir_datasets = PythonOperator(
        task_id='unir_datasets',
        python_callable=load_and_merge,
        trigger_rule='one_success'
    )

    # 7. Split de los datos
    hacer_split = PythonOperator(
        task_id='hacer_split',
        python_callable=split_data
    )

    # 8. Entrenamiento: Random Forest
    entrenar_rf = PythonOperator(
        task_id='entrenar_random_forest',
        python_callable=train_model,
        op_args=[RandomForestClassifier()]
    )

    # 9. Entrenamiento: Logistic Regression
    entrenar_logreg = PythonOperator(
        task_id='entrenar_logistic_regression',
        python_callable=train_model,
        op_args=[LogisticRegression()]
    )

    # 10. Entrenamiento: Decision Tree
    entrenar_dt = PythonOperator(
        task_id='entrenar_decision_tree',
        python_callable=train_model,
        op_args=[DecisionTreeClassifier()]
    )

    # 11. Evaluar modelos
    evaluar = PythonOperator(
        task_id='evaluar_modelos',
        python_callable=evaluate_models,
        trigger_rule=TriggerRule.ALL_SUCCESS
    )

    # Flujo del DAG
    (
        start
        >> crear_carpetas
        >> branching
        >> [descargar_data_1, descargar_data_1_y_2]
        >> unir_datasets
        >> hacer_split
        >> [entrenar_rf, entrenar_logreg, entrenar_dt]
        >> evaluar
    )
