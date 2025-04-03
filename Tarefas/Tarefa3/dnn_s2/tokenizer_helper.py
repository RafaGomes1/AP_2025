from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from config import MAX_WORDS, MAX_LEN

tokenizer = Tokenizer(num_words=MAX_WORDS, oov_token="<OOV>")

def fit_tokenizer(texts):
    tokenizer.fit_on_texts(texts)

def tokenize_pad(texts):
    sequences = tokenizer.texts_to_sequences(texts)
    return pad_sequences(sequences, maxlen=MAX_LEN)
