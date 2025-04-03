from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Flatten, Dense, Dropout, Embedding, Input
from tensorflow.keras import initializers
from config import MAX_WORDS, MAX_LEN, EMBEDDING_DIM

def build_model():
    model = Sequential([
        Input((MAX_LEN,)),
        Embedding(MAX_WORDS, EMBEDDING_DIM, embeddings_initializer=initializers.GlorotUniform(seed=44)),
        Flatten(),
        Dense(128, activation='relu'),
        Dropout(0.4),
        Dense(64, activation='relu'),
        Dense(1, activation='sigmoid')
    ])
    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
    return model
