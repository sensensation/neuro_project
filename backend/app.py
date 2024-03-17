from fastapi import FastAPI

from pydantic import BaseModel
import numpy as np
import joblib
from keras.models import load_model
from sklearn.discriminant_analysis import StandardScaler

from backend.schemas import PredictionData

app = FastAPI()

# Загрузка обученной модели и scaler
model = load_model("data/weather_model.keras")
scaler: StandardScaler = joblib.load("data/scaler.gz")

class PredictionData(BaseModel):
    data: list[list[float]]  # Ожидаем данные в виде списка списков (признаки для предсказаний)

@app.post("/predict")
async def make_prediction(data: PredictionData):
    input_data = np.array(data.data)  # Преобразование в numpy массив
    scaled_input = scaler.transform(input_data)  # Масштабирование данных
    
    predictions = model.predict(scaled_input)  # Предсказание
    predictions_binary = [1 if x > 0.5 else 0 for x in predictions.flatten()]  # Преобразование в бинарные значения
    print(scaled_input, predictions, input_data, scaled_input)
    return {"prediction": predictions_binary}  # Возврат предсказаний
