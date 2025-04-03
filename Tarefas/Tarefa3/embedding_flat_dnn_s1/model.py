import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Flatten, Dense, Dropout, Embedding, Input
from tensorflow.keras import initializers

embedding_dim = 128

def create_model(max_words, max_len):
    model = Sequential([
        Input(shape=(max_len,)),
        Embedding(max_words, embedding_dim, embeddings_initializer=initializers.GlorotUniform(seed=44)),
        Flatten(),
        Dense(128, activation='relu'),
        Dropout(0.4),
        Dense(64, activation='relu'),
        Dense(1, activation='sigmoid')
    ])

    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
    return model
