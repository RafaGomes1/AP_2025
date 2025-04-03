from tensorflow.keras.layers import Dense, Dropout, Embedding, Input, GlobalAveragePooling1D, BatchNormalization

def get_embedding_layer(max_words, embedding_dim, seed):
    from tensorflow.keras import initializers
    return Embedding(max_words, embedding_dim, embeddings_initializer=initializers.GlorotUniform(seed=seed))

def get_common_layers():
    return [
        Dense(256),
        BatchNormalization(),
        Dropout(0.5),
        Dense(128),
        BatchNormalization(),
        Dropout(0.4),
        Dense(64)
    ]