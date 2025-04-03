import re

def clean_text(text):
    """
    Limpa texto cru: remove URLs, pontuação, múltiplos espaços e normaliza para lowercase.
    """
    text = re.sub(r"http\S+", "", text)                       # remove URLs
    text = re.sub(r"[^a-zA-Z0-9\s]", "", text)                # remove pontuação
    text = re.sub(r"\s+", " ", text)                          # espaços duplicados
    return text.lower().strip()

def clean_texts(texts):
    """
    Aplica clean_text a uma série de textos (pandas.Series).
    """
    return texts.apply(clean_text)
