from models.model_architectures import create_complex_model
from config.settings import MAX_WORDS, MAX_LEN

def get_model(tokenizer):
    return create_complex_model(tokenizer, MAX_WORDS, MAX_LEN)
