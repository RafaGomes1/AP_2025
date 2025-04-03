from tensorflow.keras.layers import LeakyReLU

def get_activations():
    return [LeakyReLU() for _ in range(2)] + ['relu']
