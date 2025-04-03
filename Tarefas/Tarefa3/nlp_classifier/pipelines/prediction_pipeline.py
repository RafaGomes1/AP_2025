import tensorflow as tf
import pandas as pd
from preprocessing.tokenizer_utils import tokenize_and_pad
from data_loader import load_test_data
from tensorflow.keras.preprocessing.text import tokenizer_from_json
import json

def predict_pipeline():
    model = tf.keras.models.load_model("models/trained_complex_model.keras")

    with open("tokenizer.json", "r", encoding="utf-8") as f:
        tokenizer = tokenizer_from_json(f.read())

    X_test_texts, ids = load_test_data()
    X_test_pad = tokenize_and_pad(tokenizer, X_test_texts)

    preds = model.predict(X_test_pad)
    pred_labels = ["AI" if p > 0.5 else "Human" for p in preds.flatten()]

    pd.DataFrame({"ID": ids, "Label": pred_labels}) \
      .to_csv("results/nlp_classifier.csv", sep='\t', index=False)

    print("Ficheiro de submissão gerado.")
