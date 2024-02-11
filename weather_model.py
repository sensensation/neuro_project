
from keras import layers, Sequential
import tensorflow as tf

def build_model(input_shape):
    # return print(tf.__version__)
    """Строит полносвязную нейронную сеть."""
    model = Sequential([
        layers.Dense(64, input_shape=(input_shape,), activation='relu'),
        layers.Dense(32, activation='relu'),
        layers.Dense(1, activation='sigmoid')
    ])
    return model
