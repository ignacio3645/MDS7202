# Se crea un DockerFile para montar un contenedor 
FROM python:3.10-slim

# Establece el directorio de trabajo en el contenedor
WORKDIR /root/airflow

# Establece la variable de entorno AIRFLOW_HOME
ENV AIRFLOW_HOME=/root/airflow

# Instala curl 
RUN apt-get update && apt-get install -y curl

# Instala Apache Airflow
RUN pip install "apache-airflow==2.9.1"
RUN pip install scikit-learn
RUN pip install pandas
RUN pip install numpy
RUN pip install gradio


# Inicializa la base de datos de Airflow
RUN airflow db init

# Expone el puerto 8080 para el servidor web de Airflow
EXPOSE 8080

# Crea el usuario admin de Airflow
RUN airflow users create --role Admin --username admin --email admin \
 --firstname admin --lastname admin --password admin

# Copia las carpetas necesarias al contenedor
COPY ./dags $AIRFLOW_HOME/dags
COPY ./logs $AIRFLOW_HOME/logs
COPY ./plugins $AIRFLOW_HOME/plugins

# Comando para iniciar el servidor web y el scheduler
CMD ["sh", "-c", "airflow webserver -p 8080 & airflow scheduler"]