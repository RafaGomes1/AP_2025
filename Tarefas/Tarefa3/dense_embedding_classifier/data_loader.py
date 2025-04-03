import pandas as pd

def load_dataset(path, sep='\t'):
    return pd.read_csv(path, sep=sep, encoding='utf-8')

# ===================== tokenizer_helper.py =====================
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from config import MAX_WORDS, MAX_LEN

_tokenizer = Tokenizer(num_words=MAX_WORDS, oov_token="<OOV>")

def fit_tokenizer(texts):
    _tokenizer.fit_on_texts(texts)

def tokenize_pad(texts):
    seq = _tokenizer.texts_to_sequences(texts)
    return pad_sequences(seq, maxlen=MAX_LEN)

# ===================== preprocess_utils.py =====================
import re

def clean_text(text):
    text = re.sub(r"http\S+", "", text)
    text = re.sub(r"[^a-zA-Z0-9\s]", "", text)
    return text.lower().strip()