import pandas as pd
import tensorflow as tf
from tensorflow.keras.preprocessing.text import tokenizer_from_json
from tokenizer import tokenize_and_pad, max_len
from data_loader import load_test_data
import json

def predict_and_save_complex():
    model = tf.keras.models.load_model("trained_complex_model.keras")

    with open("tokenizer.json", "r", encoding="utf-8") as f:
        tokenizer_json = f.read()
        tokenizer = tokenizer_from_json(tokenizer_json)

    X_test_texts, ids = load_test_data()
    X_test_pad = tokenize_and_pad(tokenizer, X_test_texts)

    preds = model.predict(X_test_pad)
    pred_labels = ["AI" if p > 0.5 else "Human" for p in preds.flatten()]

    output_df = pd.DataFrame({"ID": ids, "Label": pred_labels})
    output_df.to_csv("results/cnn_lstm.csv", sep='\t', index=False)
    print("Ficheiro de submissão gerado: cnn_lstm.csv")

if __name__ == "__main__":
    predict_and_save_complex()
