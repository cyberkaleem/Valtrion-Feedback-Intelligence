from typing import Dict, Any, List

from transformers import pipeline
from transformers.utils import logging as hf_logging


MODEL_NAME = "cardiffnlp/twitter-roberta-base-sentiment-latest"

# Reduce unnecessary Hugging Face loading logs
hf_logging.set_verbosity_error()

_transformer_pipeline = None


LABEL_MAP = {
    "LABEL_0": "negative",
    "LABEL_1": "neutral",
    "LABEL_2": "positive",
    "negative": "negative",
    "neutral": "neutral",
    "positive": "positive",
}


def load_transformer_pipeline():
    """
    Load the transformer sentiment pipeline only once.
    This avoids reloading the model for every prediction.
    """
    global _transformer_pipeline

    if _transformer_pipeline is None:
        _transformer_pipeline = pipeline(
            task="sentiment-analysis",
            model=MODEL_NAME,
            top_k=None,
        )

    return _transformer_pipeline


def normalize_label(label: str) -> str:
    """
    Convert model labels into standard project labels:
    negative, neutral, positive.
    """
    label = str(label).strip()
    return LABEL_MAP.get(label, label.lower())


def predict_transformer_sentiment(text: str) -> Dict[str, Any]:
    """
    Predict sentiment using the transformer model.

    Returns:
    {
        "text": "...",
        "sentiment": "negative",
        "confidence": 0.9336,
        "probabilities": {
            "negative": 0.9336,
            "neutral": 0.0597,
            "positive": 0.0066
        },
        "model": "cardiffnlp/twitter-roberta-base-sentiment-latest"
    }
    """
    if not text or not str(text).strip():
        raise ValueError("Input text cannot be empty.")

    classifier = load_transformer_pipeline()

    raw_output = classifier(str(text), truncation=True)

    # Expected format with top_k=None:
    # [[{"label": "negative", "score": 0.93}, ...]]
    if isinstance(raw_output, list) and raw_output and isinstance(raw_output[0], list):
        predictions: List[Dict[str, Any]] = raw_output[0]
    else:
        predictions = raw_output

    probabilities = {}

    for item in predictions:
        label = normalize_label(item["label"])
        score = round(float(item["score"]), 4)
        probabilities[label] = score

    # Ensure all classes exist in output
    for label in ["negative", "neutral", "positive"]:
        probabilities.setdefault(label, 0.0)

    sentiment = max(probabilities, key=probabilities.get)
    confidence = probabilities[sentiment]

    return {
        "text": text,
        "sentiment": sentiment,
        "confidence": confidence,
        "probabilities": {
            "negative": probabilities["negative"],
            "neutral": probabilities["neutral"],
            "positive": probabilities["positive"],
        },
        "model": MODEL_NAME,
    }


if __name__ == "__main__":
    user_text = input("Enter feedback text: ")
    result = predict_transformer_sentiment(user_text)

    print()
    print("Transformer Sentiment Result")
    print("-" * 35)
    print(f"Text: {result['text']}")
    print(f"Sentiment: {result['sentiment']}")
    print(f"Confidence: {result['confidence']}")
    print("Probabilities:")
    for label, score in result["probabilities"].items():
        print(f"  {label}: {score}")
    print(f"Model: {result['model']}")