from fastapi import FastAPI
from pydantic import BaseModel
import pickle
import numpy as np
import uvicorn

# Cargamos el modelo entrenado (esto hay que cambiarlo por el modelo que nos de la p1)
with open("Laboratorio_8\modelo_xgb_basico.pkl", "rb") as f:
    model = pickle.load(f)

# Inicializar FastAPI
app = FastAPI(
    title="API de Predicción de Potabilidad del Agua",
    description="Esta API recibe mediciones físico-químicas del agua y predice si es potable (1) o no (0) usando un modelo XGBoost optimizado con Optuna.",
    version="1.0"
)

# Definición de esquema de entrada usando Pydantic
class WaterSample(BaseModel):
    ph: float
    Hardness: float
    Solids: float
    Chloramines: float
    Sulfate: float
    Conductivity: float
    Organic_carbon: float
    Trihalomethanes: float
    Turbidity: float

# Ruta GET de inicio
@app.get("/")
def home():
    return {
        "mensaje": "Este modelo predice la potabilidad del agua.",
        "entrada": ["ph", "Hardness", "Solids", "Chloramines", "Sulfate", "Conductivity", "Organic_carbon", "Trihalomethanes", "Turbidity"],
        "salida": "potabilidad (0: no potable, 1: potable)"
    }

# Ruta POST para predicción
@app.post("/potabilidad/")
def predict_potabilidad(sample: WaterSample):
    # Convertir datos a array para el modelo
    input_data = np.array([[
        sample.ph,
        sample.Hardness,
        sample.Solids,
        sample.Chloramines,
        sample.Sulfate,
        sample.Conductivity,
        sample.Organic_carbon,
        sample.Trihalomethanes,
        sample.Turbidity
    ]])

    # Realizar predicción
    pred = model.predict(input_data)[0]

    return {"potabilidad": int(pred)}

# Ejecutar el servidor si se llama directamente
if __name__ == "__main__":
    uvicorn.run("main:app", host="127.0.0.1", port=8000, reload=True)