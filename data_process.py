# import joblib
# from sklearn.model_selection import train_test_split
# from sklearn.preprocessing import StandardScaler


# def preprocess_data(data, target_column):
#     """Предобработка данных: разделение на признаки и цель, нормализация."""
#     X = data.drop([target_column, "DATE"], axis=1)

#     # Замена NaN значений в 'RAIN' на False (0) или на моду
#     data[target_column] = data[target_column].fillna(0)
#     y = data[target_column].astype("int")

#     # Разделение на обучающую и тестовую выборки
#     X_train, X_test, y_train, y_test = train_test_split(
#         X, y, test_size=0.2, random_state=42
#     )

#     # Нормализация данных
#     scaler = StandardScaler()
#     X_train_scaled = scaler.fit_transform(X_train)
#     X_test_scaled = scaler.transform(X_test)

#     # Сохранение обученного scaler
#     joblib.dump(scaler, "data/scaler.gz")

#     return X_train_scaled, X_test_scaled, y_train, y_test

import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler


def preprocess_data(data, sequence_length=7):
    features = ["PRCP", "TMIN", "YEAR", "MONTH", "DAY"]
    X = data[features]
    y = data["TMAX"]
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    X_seq, y_seq = create_sequences(X_scaled, y, sequence_length)
    X_train, X_test, y_train, y_test = train_test_split(
        X_seq, y_seq, test_size=0.2, random_state=42
    )
    return X_train, y_train


def create_sequences(input_data, output_data, sequence_length):
    sequences = []
    output = []
    for i in range(len(input_data) - sequence_length):
        sequences.append(input_data[i : i + sequence_length])
        output.append(output_data[i + sequence_length])
    return np.array(sequences), np.array(output)
