"""Shared normalization; preserve case for the cased encoder."""
import re
import unicodedata


def normalize(text: str) -> str:
    if not isinstance(text, str):
        raise ValueError("text deve ser uma string")
    text = unicodedata.normalize("NFC", text)
    if any(unicodedata.category(c) == "Cc" and not c.isspace() for c in text):
        raise ValueError("text contém caracteres de controle")
    text = re.sub(r"\s+", " ", text).strip()
    if not text:
        raise ValueError("text vazio")
    if len(text) > 20000:
        raise ValueError("text excede 20000 caracteres")
    return text


def folded(text: str) -> str:
    return "".join(c for c in unicodedata.normalize("NFD", text.lower())
                   if unicodedata.category(c) != "Mn")
