from model_builder import build_model
from callbacks.callbacks import get_callbacks
from config import EPOCHS, BATCH_SIZE
from logging_utils import get_logger

logger = get_logger("TRAIN")

def train_model(X_train, y_train, X_val, y_val):
    model = build_model()
    logger.info("A iniciar treino com %d amostras...", len(X_train))
    model.fit(X_train, y_train, epochs=EPOCHS, batch_size=BATCH_SIZE,
              validation_data=(X_val, y_val), verbose=1,
              callbacks=get_callbacks())
    return model