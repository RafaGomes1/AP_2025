import pandas as pd

def load_dataset(path, sep='\t'):
    return pd.read_csv(path, sep=sep, encoding='utf-8')

def load_training_data():
    X_train = load_dataset("../data/test_input.csv")
    y_train = load_dataset("../data/test_output.csv")
    y_train = y_train["Label"].map({"AI": 1, "Human": 0}).values
    return X_train["Text"], y_train

def load_validation_data():
    X_val = load_dataset("../data/human_ai_input.csv")
    y_val = load_dataset("../data/human_ai_output.csv")
    y_val = y_val["Label"].map({"AI": 1, "Human": 0}).values
    return X_val["Text"], y_val

def load_test_data():
    X_test = load_dataset("../data/dataset3_inputs.csv")
    ids = X_test["ID"]
    return X_test["Text"], ids
