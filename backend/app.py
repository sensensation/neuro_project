import pandas as pd
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi import Request

from backend.schemas import TrainResult
from data_process import preprocess_data
from neuralwebs.BERT.weather_model import build_transformer_model
from neuralwebs.D_CNN.weather_model import build_model as build_1d_cnn_model
from neuralwebs.FCNN.weather_model import build_model as build_fcnn_model
from train import train_model_async

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Allows all origins
    allow_methods=["*"],  # Allows all methods
    allow_headers=["*"],  # Allows all headers
)


def load_data(filepath):
    data = pd.read_csv(filepath)

    data["DATE"] = pd.to_datetime(data["DATE"])
    data["YEAR"] = data["DATE"].dt.year
    data["MONTH"] = data["DATE"].dt.month
    data["DAY"] = data["DATE"].dt.day
    return data


@app.post("/train_fcnn/", response_model=TrainResult)
async def train_fcnn():
    data = load_data("./data/seattle_weather_1948-2017.csv")
    X_train, X_test, y_train, y_test = preprocess_data(data)
    fcnn_model = build_fcnn_model(X_train[0].shape)
    history = await train_model_async(fcnn_model, X_test, y_test, epochs=15)
    return TrainResult(
        loss=history.history["val_loss"][-1], mae=history.history["val_mae"][-1]
    )


@app.post("/train_1d_cnn/", response_model=TrainResult)
async def train_1d_cnn(request: Request):
    data = load_data("./data/seattle_weather_1948-2017.csv")
    X_train, X_test, y_train, y_test = preprocess_data(data)
    cnn_model = build_1d_cnn_model(X_train[0].shape)
    history = await train_model_async(cnn_model, X_train, y_train, epochs=15)
    return TrainResult(
        loss=history.history["val_loss"][-1], mae=history.history["val_mae"][-1]
    )


@app.post("/train_bert/", response_model=TrainResult)
async def train_transformer():
    data = load_data("./data/seattle_weather_1948-2017.csv")
    X_train, X_test, y_train, y_test = preprocess_data(data)
    transformer_model = build_transformer_model(
        sequence_length=X_train.shape[1],  # Количество временных шагов
        num_features=X_train.shape[2],  # Количество признаков на каждом временном шаге
        d_model=128,  # Глубина модели
        num_heads=8,  # Количество голов внимания
        num_layers=4,  # Количество слоев трансформера
        dff=512,  # Размерность скрытого слоя внутренних полносвязных слоев
        rate=0.1,  # Процент дропаута
    )
    history = await train_model_async(transformer_model, X_test, y_test, epochs=15)
    return TrainResult(
        loss=history.history["val_loss"][-1], mae=history.history["val_mae"][-1]
    )


@app.get("/dataset_info/")
async def get_dataset_info():
    data = pd.read_csv("./data/seattle_weather_1948-2017.csv")
    info = {
        "number_of_records": len(data),
        "number_of_columns": len(data.columns),
        "columns": data.columns.tolist(),
    }
    return info
