from keras.layers import Conv1D, Dense, Flatten, MaxPooling1D
from keras.models import Sequential
from keras.optimizers import Adam


def build_model(input_shape):
    """Строит одномерную свёрточную нейронную сеть."""
    model = Sequential(
        [
            Conv1D(
                filters=64, kernel_size=2, activation="relu", input_shape=input_shape
            ),
            MaxPooling1D(pool_size=2),
            Conv1D(filters=32, kernel_size=2, activation="relu"),
            Flatten(),
            Dense(64, activation="relu"),
            Dense(1),
        ]
    )
    model.compile(optimizer=Adam(), loss="mse", metrics=["mae"])
    return model
