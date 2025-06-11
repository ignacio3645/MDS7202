import os
from datetime import datetime
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.metrics import accuracy_score
import joblib

def create_folders():
    fecha = datetime.now().date()
    os.makedirs(f"{fecha}/raw", exist_ok=True)
    os.makedirs(f"{fecha}/preprocessed", exist_ok=True)
    os.makedirs(f"{fecha}/splits", exist_ok=True)
    os.makedirs(f"{fecha}/models", exist_ok=True)

def load_and_merge():
    fecha = datetime.now().date()
    dfs = []

    for file_name in ["data_1.csv", "data_2.csv"]:
        file_path = f"{fecha}/raw/{file_name}"
        if os.path.exists(file_path):
            df = pd.read_csv(file_path)
            dfs.append(df)

    if dfs:
        df_final = pd.concat(dfs, ignore_index=True)
        df_final.to_csv(f"{fecha}/preprocessed/data_merged.csv", index=False)
    else:
        raise FileNotFoundError("No se encontró ninguno de los archivos 'data_1.csv' ni 'data_2.csv' en la carpeta 'raw'.")

def split_data():
    fecha = datetime.now().date()
    data_path = f"{fecha}/preprocessed/data_merged.csv"
    df = pd.read_csv(data_path, index_col=0)

    X_train, X_test, y_train, y_test = train_test_split(
        df.drop(columns='HiringDecision'),
        df['HiringDecision'],
        test_size=0.2,
        random_state=42,
        stratify=df['HiringDecision']
    )

    df_train = X_train.copy()
    df_train['HiringDecision'] = y_train
    df_test = X_test.copy()
    df_test['HiringDecision'] = y_test

    df_train.to_csv(f"{fecha}/splits/df_train.csv", index=False)
    df_test.to_csv(f"{fecha}/splits/df_test.csv", index=False)

def train_model(model):
    from inspect import isclass

    fecha = datetime.now().date()
    df_train = pd.read_csv(f"{fecha}/splits/df_train.csv")

    X_train = df_train.drop(columns='HiringDecision')
    y_train = df_train['HiringDecision']

    # Se detectan columnas numéricas y categóricas
    num_cols = X_train.select_dtypes(include='number').columns.tolist()
    cat_cols = X_train.select_dtypes(exclude='number').columns.tolist()

    preprocessor = ColumnTransformer(transformers=[
        ('num', StandardScaler(), num_cols),
        ('cat', OneHotEncoder(handle_unknown='ignore'), cat_cols)
    ])

    # Identificador del modelo
    model_name = model.__class__.__name__ if not isclass(model) else model.__name__

    pipeline = Pipeline([
        ('preprocessor', preprocessor),
        ('classifier', model)
    ])

    pipeline.fit(X_train, y_train)

    model_path = f"{fecha}/models/pipeline_{model_name}.joblib"
    joblib.dump(pipeline, model_path)

def evaluate_models():
    from glob import glob

    fecha = datetime.now().date()
    df_test = pd.read_csv(f"{fecha}/splits/df_test.csv")
    X_test = df_test.drop(columns='HiringDecision')
    y_test = df_test['HiringDecision']

    best_model = None
    best_acc = 0
    best_model_name = ""

    model_paths = glob(f"{fecha}/models/*.joblib")

    for path in model_paths:
        model = joblib.load(path)
        y_pred = model.predict(X_test)
        acc = accuracy_score(y_test, y_pred)

        model_name = os.path.basename(path)
        print(f"Modelo: {model_name} | Accuracy: {acc:.4f}")

        if acc > best_acc:
            best_acc = acc
            best_model = model
            best_model_name = model_name

    if best_model:
        best_model_path = f"{fecha}/models/best_model.joblib"
        joblib.dump(best_model, best_model_path)
        print(f"\n🔍 Mejor modelo: {best_model_name} con Accuracy: {best_acc:.4f}")
    else:
        print("No se encontraron modelos para evaluar.")
