from data_loader import load_dataset
from tokenizer_helper import fit_tokenizer, tokenize_pad
from train import train_model
from predict import make_predictions
from config import *

def main():
    X_train = load_dataset(TRAIN_INPUT_PATH)
    y_train_df = load_dataset(TRAIN_OUTPUT_PATH)
    X_val = load_dataset(VAL_INPUT_PATH)
    y_val_df = load_dataset(VAL_OUTPUT_PATH)
    X_test = load_dataset(TEST_INPUT_PATH)

    fit_tokenizer(X_train["Text"])
    X_train_pad = tokenize_pad(X_train["Text"])
    X_val_pad = tokenize_pad(X_val["Text"])
    X_test_pad = tokenize_pad(X_test["Text"])

    y_train = y_train_df["Label"].map({"AI": 1, "Human": 0}).values
    y_val = y_val_df["Label"].map({"AI": 1, "Human": 0}).values

    model = train_model(X_train_pad, y_train, X_val_pad, y_val)

    make_predictions(model, X_test_pad, X_test["ID"])

if __name__ == "__main__":
    main()
