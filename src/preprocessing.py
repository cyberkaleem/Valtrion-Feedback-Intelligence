import re


def clean_text(text: str) -> str:
    """
    Clean raw text while preserving sentiment-critical words like not, no, never.
    """
    if text is None:
        return ""

    text = str(text)

    # Remove URLs
    text = re.sub(r"http\S+|www\S+", " ", text)

    # Remove mentions
    text = re.sub(r"@\w+", " ", text)

    # Keep hashtag word but remove the # symbol
    text = text.replace("#", "")

    # Lowercase
    text = text.lower()

    # Keep alphabets, spaces, and apostrophes
    text = re.sub(r"[^a-z\s']", " ", text)

    # Remove extra spaces
    text = re.sub(r"\s+", " ", text).strip()

    return text


def tokenize_text(text: str) -> list:
    """
    Tokenize cleaned text into words.
    """
    cleaned = clean_text(text)
    return cleaned.split()