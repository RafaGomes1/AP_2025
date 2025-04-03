from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint

def get_callbacks():
    return [
        EarlyStopping(monitor='val_loss', patience=3, restore_best_weights=True),
        ModelCheckpoint("best_model.keras", save_best_only=True, monitor='val_loss')
    ]