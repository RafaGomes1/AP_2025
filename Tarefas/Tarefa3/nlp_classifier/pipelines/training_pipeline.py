from preprocessing.tokenizer_utils import create_tokenizer, tokenize_and_pad
from data_loader import load_training_data, load_validation_data
from models.model_factory import get_model
from utils.seed import set_seed
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
import json

def train_pipeline():
    set_seed()

    X_train_texts, y_train = load_training_data()
    X_val_texts, y_val = load_validation_data()

    tokenizer = create_tokenizer(X_train_texts)
    X_train_pad = tokenize_and_pad(tokenizer, X_train_texts)
    X_val_pad = tokenize_and_pad(tokenizer, X_val_texts)

    model = get_model(tokenizer)

    callbacks = [
        EarlyStopping(monitor='val_loss', patience=3, restore_best_weights=True),
        ModelCheckpoint('models/trained_complex_model.keras', monitor='val_loss', save_best_only=True)
    ]

    model.fit(X_train_pad, y_train, epochs=20, batch_size=64,
              validation_data=(X_val_pad, y_val), callbacks=callbacks)

    with open("tokenizer.json", "w", encoding="utf-8") as f:
        f.write(tokenizer.to_json())
