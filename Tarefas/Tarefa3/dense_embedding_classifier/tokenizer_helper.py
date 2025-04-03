from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from config import MAX_WORDS, MAX_LEN

_tokenizer = None  # Tokenizer privado, gerido via API

def initialize_tokenizer(oov_token="<OOV>"):
    """
    Inicializa o tokenizer global com os parâmetros definidos.
    """
    global _tokenizer
    _tokenizer = Tokenizer(num_words=MAX_WORDS, oov_token=oov_token)

def fit_tokenizer(texts):
    """
    Ajusta o tokenizer ao vocabulário dos textos.
    """
    if _tokenizer is None:
        raise ValueError("Tokenizer not initialized. Call initialize_tokenizer() first.")
    _tokenizer.fit_on_texts(texts)

def texts_to_padded_sequences(texts):
    """
    Converte textos em sequências de índices e aplica padding.
    """
    if _tokenizer is None:
        raise ValueError("Tokenizer not initialized. Call initialize_tokenizer() first.")
    sequences = _tokenizer.texts_to_sequences(texts)
    return pad_sequences(sequences, maxlen=MAX_LEN)

def get_tokenizer():
    """
    Devolve o tokenizer atual.
    """
    return _tokenizer
