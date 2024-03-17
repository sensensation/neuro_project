from keras import Sequential, layers

def build_model(input_shape):
    """Строит полносвязную нейронную сеть."""
    model = Sequential(
        [
            layers.Dense(64, input_shape=(input_shape,), activation="relu"),
            layers.Dense(32, activation="relu"),
            layers.Dense(1, activation="sigmoid"),
        ]
    )
    return model
