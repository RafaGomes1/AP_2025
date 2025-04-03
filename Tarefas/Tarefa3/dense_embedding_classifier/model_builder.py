from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Input, GlobalAveragePooling1D, Dense
from architecture.layers import get_embedding_layer, get_common_layers
from architecture.activations import get_activations
from config import MAX_WORDS, MAX_LEN, EMBEDDING_DIM


def build_model():
    model = Sequential()
    model.add(Input((MAX_LEN,)))
    model.add(get_embedding_layer(MAX_WORDS, EMBEDDING_DIM, seed=44))
    model.add(GlobalAveragePooling1D())

    layers = get_common_layers()
    activations = get_activations()

    for layer, activation in zip(layers, activations):
        model.add(layer)
        if isinstance(activation, str):
            layer.activation = activation
        else:
            model.add(activation)

    model.add(Dense(1, activation='sigmoid'))
    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
    return model