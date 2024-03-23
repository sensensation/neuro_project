from keras.layers import Dense, Flatten
from keras.models import Sequential
from keras.optimizers import Adam


def build_model(input_shape):
    """Строит полносвязную нейронную сеть."""
    model = Sequential(
        [
            Flatten(input_shape=input_shape),
            Dense(64, activation="relu"),
            Dense(64, activation="relu"),
            Dense(1),
        ]
    )
    model.compile(optimizer=Adam(), loss="mse", metrics=["mae"])
    return model
