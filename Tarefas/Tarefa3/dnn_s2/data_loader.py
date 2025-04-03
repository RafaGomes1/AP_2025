import pandas as pd

def load_dataset(path, sep='\t'):
    return pd.read_csv(path, sep=sep, encoding='utf-8')