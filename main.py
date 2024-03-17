from sklearn.metrics import accuracy_score
from data_process import preprocess_data
from weather_model import build_model
from train import train_model

import pandas as pd


def load_data(filepath):
    """Загружает данные из CSV файла."""
    return pd.read_csv(filepath)


def make_prediction(model, X):
    """Выполняет предсказание с помощью обученной модели."""
    predictions = model.predict(X)
    return predictions


if __name__ == "__main__":
    # Загрузка и предобработка данных
    data = load_data("./data/seattle_weather_1948-2017.csv")
    print(data.columns)
    X_train, X_test, y_train, y_test = preprocess_data(data, "RAIN")

    # Построение и обучение модели
    model = build_model(X_train.shape[1])
    train_model(model, X_train, y_train, X_test, y_test)

    # Предсказание
    predictions = make_prediction(model, X_test)
    print(predictions[:5])  # Вывод первых пяти предсказаний

    # Преобразование предсказаний в бинарные значения (если необходимо)
    predictions_binary = [1 if x > 0.5 else 0 for x in predictions]

    # Расчет точности
    accuracy = accuracy_score(y_test, predictions_binary)
    print(f"Точность модели: {accuracy}")
    model.save("data/weather_model.keras")

