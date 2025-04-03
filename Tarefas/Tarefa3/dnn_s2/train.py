from config import EPOCHS, BATCH_SIZE
from model_builder import build_model

def train_model(X_train, y_train, X_val, y_val):
    model = build_model()
    model.fit(X_train, y_train, epochs=EPOCHS, batch_size=BATCH_SIZE,
              validation_data=(X_val, y_val), verbose=1)
    return model
