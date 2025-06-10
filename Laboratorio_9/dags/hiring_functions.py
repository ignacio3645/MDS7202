import os 
from datetime import datetime
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, f1_score
import joblib
import gradio as gr


def create_folders():
    # Se crea carpeta que utiliza la fecha de ejucion como nombre 
    fecha = datetime.now().date()
    os.mkdir(str(fecha))


    # Adicionalmente, se crean las subcarpetas raw, splits, models
    os.mkdir(str(fecha) + '/raw')
    os.mkdir(str(fecha) + '/splits')
    os.mkdir(str(fecha) + '/models')

def split_data():
    fecha = datetime.now().date()
    df = pd.read_csv(str(fecha) + '/raw/data_1.csv')
    # Se separan los datos en train, validation y test
    X_train, X_test, y_train, y_test = train_test_split(df.drop(columns='HiringDecision'), df['HiringDecision'], test_size=0.2, random_state=42, shuffle=True, stratify=df['HiringDecision']) # se hace de manera aleatoria, manteniendo la distribucion original
    
    # Se definen los nuevos datasets
    df_train = X_train.copy()
    df_train['HiringDecision'] = y_train
    df_test = X_test.copy()
    df_test['HiringDecision'] = y_test

    # Se guardan en la carpeta splits
    df_train.to_csv(str(fecha) + '/splits/df_train.csv')
    df_test.to_csv(str(fecha) + '/splits/df_test.csv')


def preprocess_and_train():
    fecha = datetime.now().date()

    # Se leen los df de train y test
    df_train = pd.read_csv(str(fecha) + '/splits/df_train.csv')
    df_test = pd.read_csv(str(fecha) + '/splits/df_test.csv')

    pipeline = Pipeline([('RandomForestClassifier', RandomForestClassifier())])
    
    # Se aplica el pipeline 
    pipeline.fit(df_train.drop(columns='HiringDecision'), df_train['HiringDecision'])

    # Se obtienen predicciones 
    y_pred = pipeline.predict(df_test.drop(columns='HiringDecision'))

    # Se obtiene accuracy 
    acc = accuracy_score(df_test['HiringDecision'], y_pred)

    print(f'Accuracy: {acc}')

    # Se imprime el f1score de la clase positiva 
    df_test_pos = df_test[df_test['HiringDecision'] == 1]
    y_pred_pos = pipeline.predict(df_test_pos.drop(columns='HiringDecision'))
    f1s = f1_score(df_test_pos['HiringDecision'], y_pred_pos)

    print(f'F1 Score: {f1s}')

    # Se guarda el modelo en models
    joblib.dump(pipeline, str(fecha) + '/models/pipeline_RandomForestClassifier.joblib')


# Se incorporan funciones predict y gradio_interface
def predict(file,model_path):

    pipeline = joblib.load(model_path)
    input_data = pd.read_json(file)
    predictions = pipeline.predict(input_data)
    print(f'La prediccion es: {predictions}')
    labels = ["No contratado" if pred == 0 else "Contratado" for pred in predictions]

    return {'Predicción': labels[0]}


def gradio_interface():
    fecha = datetime.now().date()
    
    model_path= str(fecha) + '/models/pipeline_RandomForestClassifier.joblib'

    interface = gr.Interface(
        fn=lambda file: predict(file, model_path),
        inputs=gr.File(label="Sube un archivo JSON"),
        outputs="json",
        title="Hiring Decision Prediction",
        description="Sube un archivo JSON con las características de entrada para predecir si Vale será contratada o no."
    )
    interface.launch(share=True)
