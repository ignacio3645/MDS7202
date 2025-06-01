import pandas as pd
from sklearn import set_config
from sklearn.model_selection import train_test_split
import mlflow
import optuna
from optuna.samplers import TPESampler
optuna.logging.set_verbosity(optuna.logging.WARNING)
from sklearn.metrics import f1_score
import xgboost as xgb
import pickle

# Se abre el csv
df = pd.read_csv('water_potability.csv')


# Train, val 
set_config(transform_output="pandas")

# Se separan los datos en train, validation y test
X_train, X_val, y_train, y_val = train_test_split(df.drop(columns='Potability'), df['Potability'], test_size=0.2, random_state=42, shuffle=True) # se hace de manera aleatoria, manteniendo la distribucion original

# Funcion que devuelve el mejor modelo
def get_best_model(experiment_id):
    runs = mlflow.search_runs(experiment_id)
    best_model_id = runs.sort_values("metrics.valid_f1")["run_id"].iloc[0]
    best_model = mlflow.sklearn.load_model("runs:/" + best_model_id + "/model")

    return best_model


def objective(trial):
    # Hiperparámetros a optimizar
    learning_rate = trial.suggest_float("learning_rate", 0.001, 0.1)
    n_estimators = trial.suggest_int("n_estimators", 50, 1000)
    max_depth = trial.suggest_int("max_depth", 3, 10)
    max_leaves = trial.suggest_int("max_leaves", 0, 100)
    min_child_weight = trial.suggest_int("min_child_weight", 1, 5)
    reg_alpha = trial.suggest_float("reg_alpha", 0.0, 1.0)
    reg_lambda = trial.suggest_float("reg_lambda", 0.0, 1.0)

    # Se crea el expermento
    mlflow.create_experiment("XGBoost con optuna")
    
    # Se determina el comienzo antes del entrenamiento
    with mlflow.start_run():
        # Entrenamos con lo optimizado
        model = xgb.XGBRegressor(
            learning_rate=learning_rate,
            n_estimators=n_estimators,
            max_depth=max_depth,
            max_leaves=max_leaves,
            min_child_weight=min_child_weight,
            reg_alpha=reg_alpha,
            reg_lambda=reg_lambda,
            random_state=42,
            verbosity=0
        )

        model.fit(X_train, y_train)

        # Se obtiene f1_score
        y_pred = model.predict(X_val)
        f1s = f1_score(y_val, y_pred)

        # Se guarda f1_score
        mlflow.log_metric("valid_f1", f1s)

        # Se guardan los gráficos
        # mlflow.log_figure(, artifact_file="plots")

    return f1s


# Funcion a implementar
def optimize_model():
    sampler = TPESampler(seed=42)
    study = optuna.create_study(direction="maximize", sampler=sampler)
    study.optimize(objective, timeout=300)  # por 5 minutos

    # Se obtiene el mejor modelo
    best_model = get_best_model(mlflow.get_experiment_by_name(f"XGBoost con optuna").experiment_id)

    with open("best_model_xgboost_optuna.pkl", "wb") as f:
        pickle.dump(best_model, f)

    # Se guarda el mejor modelo en carpeta models
    mlflow.xgboost.log_model(best_model, artifact_path="models")


if __name__ == "__main__":
    optimize_model()