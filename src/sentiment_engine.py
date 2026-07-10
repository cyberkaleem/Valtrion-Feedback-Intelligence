import json
from typing import Dict, Any
from unittest import result

from src.predict import predict_sentiment
from src.transformer_sentiment import predict_transformer_sentiment
from src.reliability_engine import analyze_reliability
from src.intensity_engine import analyze_sentiment_intensity
from src.aspect_analyzer import analyze_aspects


def compare_sentiment_models(text: str) -> Dict[str, Any]:
    """
    Run both the classical ML model and transformer model,
    then return a clean comparison result.

    Classical model:
    - TF-IDF + Logistic Regression
    - Used as baseline/comparison

    Transformer model:
    - CardiffNLP RoBERTa sentiment model
    - Used as advanced/recommended prediction engine
    """
    if not text or not str(text).strip():
        raise ValueError("Input text cannot be empty.")

    classical_result = predict_sentiment(text)
    transformer_result = predict_transformer_sentiment(text)

    classical_sentiment = classical_result.get("sentiment")
    transformer_sentiment = transformer_result.get("sentiment")

    model_agreement = classical_sentiment == transformer_sentiment
    reliability_analysis = analyze_reliability(
    recommended_sentiment=transformer_sentiment,
    recommended_confidence=transformer_result.get("confidence"),
    recommended_probabilities=transformer_result.get("probabilities", {}),
    model_agreement=model_agreement,
)
    intensity_analysis = analyze_sentiment_intensity(
    text=text,
    sentiment=transformer_sentiment,
    confidence=transformer_result.get("confidence"),
    probabilities=transformer_result.get("probabilities", {}),
)
    aspect_analysis = analyze_aspects(text)


    result = {
        "text": text,
        "recommended_sentiment": transformer_sentiment,
        "recommended_model": "transformer",
        "model_agreement": model_agreement,
        "reliability": reliability_analysis,
        "intensity": intensity_analysis,
        "aspect_analysis": aspect_analysis,
        "classical_model": {
            "model_type": "TF-IDF + Logistic Regression",
            "sentiment": classical_sentiment,
            "confidence": classical_result.get("confidence"),
            "probabilities": classical_result.get("probabilities", {}),
        },
        "transformer_model": {
            "model_type": "CardiffNLP RoBERTa Transformer",
            "sentiment": transformer_sentiment,
            "confidence": transformer_result.get("confidence"),
            "probabilities": transformer_result.get("probabilities", {}),
            "model_name": transformer_result.get("model"),
        },
    }

    return result


def print_comparison(result: Dict[str, Any]) -> None:
    """
    Pretty-print model comparison result in terminal.
    """
    print()
    print("Sentiment Model Comparison")
    print("=" * 45)
    print(f"Text: {result['text']}")
    print(f"Recommended Sentiment: {result['recommended_sentiment']}")
    print(f"Recommended Model: {result['recommended_model']}")
    print(f"Model Agreement: {result['model_agreement']}")

    print()
    print("Reliability Assessment")
    print("-" * 45)
    print(f"Confidence Level: {result['reliability']['confidence_level']}")
    print(f"Decision Margin: {result['reliability']['decision_margin']}")
    print(f"Review Status: {result['reliability']['review_status']}")
    print(f"Reason: {result['reliability']['reliability_reason']}")

    print()
    print("Sentiment Intensity")
    print("-" * 45)
    print(f"Intensity: {result['intensity']['sentiment_intensity']}")
    print(f"Probability Spread: {result['intensity']['probability_spread']}")
    print(f"Reason: {result['intensity']['intensity_reason']}")

    print()
    print("Aspect Intelligence")
    print("-" * 45)
    aspect_analysis = result.get("aspect_analysis", {})
    print(f"Aspect Count: {aspect_analysis.get('aspect_count')}")
    print(f"Mixed Feedback: {aspect_analysis.get('mixed_feedback')}")

    for aspect in aspect_analysis.get("aspects", []):
        print()
        print(f"Aspect: {aspect.get('aspect')}")
        print(f"Evidence: {aspect.get('evidence')}")
        print(f"Sentiment: {aspect.get('sentiment')}")
        print(f"Confidence: {aspect.get('confidence')}")

    print()
    print("Classical Model")
    print("-" * 45)
    print(f"Type: {result['classical_model']['model_type']}")
    print(f"Sentiment: {result['classical_model']['sentiment']}")
    print(f"Confidence: {result['classical_model']['confidence']}")
    print("Probabilities:")
    for label, score in result["classical_model"]["probabilities"].items():
        print(f"  {label}: {score}")

    print()
    print("Transformer Model")
    print("-" * 45)
    print(f"Type: {result['transformer_model']['model_type']}")
    print(f"Sentiment: {result['transformer_model']['sentiment']}")
    print(f"Confidence: {result['transformer_model']['confidence']}")
    print("Probabilities:")
    for label, score in result["transformer_model"]["probabilities"].items():
        print(f"  {label}: {score}")

    print()
    print("Raw JSON")
    print("-" * 45)
    print(json.dumps(result, indent=4))


if __name__ == "__main__":
    user_text = input("Enter feedback text: ")
    comparison_result = compare_sentiment_models(user_text)
    print_comparison(comparison_result)