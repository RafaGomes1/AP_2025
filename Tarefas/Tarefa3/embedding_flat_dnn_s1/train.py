import numpy as np
import tensorflow as tf
import random
from data_loader import load_training_data, load_validation_data
from tokenizer import create_tokenizer, tokenize_and_pad, max_words, max_len
from model import create_model

def set_seed(seed=42):
    np.random.seed(seed)
    tf.random.set_seed(seed)
    random.seed(seed)

def train_model():
    set_seed()
    
    X_train_texts, y_train = load_training_data()
    X_val_texts, y_val = load_validation_data()

    tokenizer = create_tokenizer(X_train_texts)
    X_train_pad = tokenize_and_pad(tokenizer, X_train_texts)
    X_val_pad = tokenize_and_pad(tokenizer, X_val_texts)

    model = create_model(max_words, max_len)

    model.fit(X_train_pad, y_train, epochs=15, batch_size=64,
              validation_data=(X_val_pad, y_val), verbose=1)

    model.save("trained_model.keras")
    tokenizer_json = tokenizer.to_json()
    with open("tokenizer.json", "w", encoding="utf-8") as f:
        f.write(tokenizer_json)

if __name__ == "__main__":
    train_model()
