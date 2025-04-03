import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, Embedding, Input, SpatialDropout1D, Conv1D, GlobalMaxPooling1D, Bidirectional, LSTM, BatchNormalization

embedding_dim = 100  # GloVe 100d

def load_glove_embeddings(tokenizer, max_words):
    embeddings_index = {}
    with open('../data/glove.6B.100d.txt', 'r', encoding='utf-8') as f:
        for line in f:
            values = line.split()
            word = values[0]
            vector = np.asarray(values[1:], dtype='float32')
            embeddings_index[word] = vector

    embedding_matrix = np.zeros((max_words, embedding_dim))
    for word, i in tokenizer.word_index.items():
        if i < max_words:
            embedding_vector = embeddings_index.get(word)
            if embedding_vector is not None:
                embedding_matrix[i] = embedding_vector
    return embedding_matrix

def create_complex_model(tokenizer, max_words, max_len):
    embedding_matrix = load_glove_embeddings(tokenizer, max_words)

    model = Sequential([
        Input(shape=(max_len,)),
        Embedding(max_words, embedding_dim, weights=[embedding_matrix], trainable=False),
        SpatialDropout1D(0.3),
        Conv1D(128, 5, activation='relu'),
        Bidirectional(LSTM(64, return_sequences=True)),
        GlobalMaxPooling1D(),
        Dense(128, activation='relu'),
        BatchNormalization(),
        Dropout(0.4),
        Dense(64, activation='relu'),
        Dropout(0.3),
        Dense(1, activation='sigmoid')
    ])

    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
    return model
