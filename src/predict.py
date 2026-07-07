import json
from pathlib import Path
from typing import Dict, Any

import joblib

ROOT_DIR = Path(__file__).resolve().parents[1]
MODEL_PATH = ROOT_DIR / "models" / "sentiment_pipeline.joblib"


def load_model():
    """
    Load trained sentiment model pipeline.
    """
    if not MODEL_PATH.exists():
        raise FileNotFoundError(
            "Trained model not found. Run: python -m src.train_model"
        )

    return joblib.load(MODEL_PATH)


def predict_sentiment(text: str) -> Dict[str, Any]:
    """
    Predict sentiment and confidence score for a single text input.
    """
    if not text or not str(text).strip():
        raise ValueError("Input text cannot be empty.")

    model = load_model()

    prediction = model.predict([text])[0]

    result = {
        "text": text,
        "sentiment": prediction,
        "confidence": None,
        "probabilities": {},
    }

    if hasattr(model, "predict_proba"):
        probabilities = model.predict_proba([text])[0]
        classes = model.classes_

        probability_dict = {
            label: round(float(prob), 4)
            for label, prob in zip(classes, probabilities)
        }

        result["probabilities"] = probability_dict
        result["confidence"] = round(float(max(probabilities)), 4)

    return result


if __name__ == "__main__":
    user_text = input("Enter feedback text: ")
    output = predict_sentiment(user_text)
    print(json.dumps(output, indent=4))