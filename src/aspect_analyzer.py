import json
import re
from typing import Dict, Any, List

from src.transformer_sentiment import predict_transformer_sentiment


ASPECT_KEYWORDS = {
    "Product Quality": [
        "product",
        "quality",
        "item",
        "material",
        "build",
        "defective",
        "damaged",
        "broken",
        "works",
        "working",
        "performance",
    ],
    "Delivery": [
        "delivery",
        "shipping",
        "shipped",
        "arrived",
        "late",
        "delayed",
        "courier",
        "package",
        "parcel",
        "dispatch",
    ],
    "Customer Support": [
        "support",
        "service",
        "staff",
        "agent",
        "representative",
        "helped",
        "help",
        "response",
        "reply",
        "rude",
        "customer care",
    ],
    "Price and Value": [
        "price",
        "cost",
        "expensive",
        "cheap",
        "affordable",
        "value",
        "worth",
        "money",
        "overpriced",
    ],
    "Payment and Refund": [
        "payment",
        "refund",
        "charged",
        "transaction",
        "billing",
        "paid",
        "deducted",
        "cashback",
    ],
    "App or Website Experience": [
        "app",
        "website",
        "site",
        "login",
        "crash",
        "crashed",
        "slow",
        "interface",
        "page",
        "error",
    ],
    "Packaging": [
        "packaging",
        "box",
        "packed",
        "seal",
        "wrapped",
        "cover",
    ],
}


def split_into_clauses(text: str) -> List[str]:
    """
    Split feedback into smaller clauses using punctuation and contrast words.
    """
    if not text:
        return []

    text = str(text).strip()

    # Add separators around contrast/conjunction words.
    text = re.sub(
        r"\b(but|however|although|though|and|while)\b",
        "|",
        text,
        flags=re.IGNORECASE,
    )

    # Split using punctuation or inserted separators.
    parts = re.split(r"[.!?;,|]+", text)

    clauses = [part.strip() for part in parts if part.strip()]
    return clauses


def detect_aspects_in_text(text: str) -> List[str]:
    """
    Detect aspect categories present in a piece of text.
    """
    text_lower = text.lower()
    detected_aspects = []

    for aspect, keywords in ASPECT_KEYWORDS.items():
        if any(keyword in text_lower for keyword in keywords):
            detected_aspects.append(aspect)

    return detected_aspects


def collect_aspect_clauses(text: str) -> Dict[str, List[str]]:
    """
    Map each detected aspect to relevant clauses.
    """
    clauses = split_into_clauses(text)

    aspect_clauses = {}

    for clause in clauses:
        detected_aspects = detect_aspects_in_text(clause)

        for aspect in detected_aspects:
            aspect_clauses.setdefault(aspect, []).append(clause)

    return aspect_clauses


def get_mixed_feedback_status(aspect_results: List[Dict[str, Any]]) -> bool:
    """
    Identify whether feedback has multiple sentiment directions across aspects.
    """
    sentiments = {
        item["sentiment"]
        for item in aspect_results
        if item.get("sentiment") in {"positive", "negative", "neutral"}
    }

    return len(sentiments) > 1


def analyze_aspects(text: str) -> Dict[str, Any]:
    """
    Analyze aspect-level sentiment using keyword aspect detection
    and transformer sentiment prediction for each aspect clause.
    """
    if not text or not str(text).strip():
        raise ValueError("Input text cannot be empty.")

    aspect_clauses = collect_aspect_clauses(text)

    aspect_results = []

    for aspect, clauses in aspect_clauses.items():
        aspect_text = " ".join(clauses)

        sentiment_result = predict_transformer_sentiment(aspect_text)

        aspect_results.append(
            {
                "aspect": aspect,
                "evidence": aspect_text,
                "sentiment": sentiment_result["sentiment"],
                "confidence": sentiment_result["confidence"],
                "probabilities": sentiment_result["probabilities"],
            }
        )

    if not aspect_results:
        sentiment_result = predict_transformer_sentiment(text)

        aspect_results.append(
            {
                "aspect": "General Experience",
                "evidence": text,
                "sentiment": sentiment_result["sentiment"],
                "confidence": sentiment_result["confidence"],
                "probabilities": sentiment_result["probabilities"],
            }
        )

    mixed_feedback = get_mixed_feedback_status(aspect_results)

    return {
        "text": text,
        "aspect_count": len(aspect_results),
        "mixed_feedback": mixed_feedback,
        "aspects": aspect_results,
    }


def print_aspect_analysis(result: Dict[str, Any]) -> None:
    """
    Pretty-print aspect analysis in terminal.
    """
    print()
    print("Aspect-Based Sentiment Analysis")
    print("=" * 45)
    print(f"Text: {result['text']}")
    print(f"Aspect Count: {result['aspect_count']}")
    print(f"Mixed Feedback: {result['mixed_feedback']}")

    for item in result["aspects"]:
        print()
        print(f"Aspect: {item['aspect']}")
        print(f"Evidence: {item['evidence']}")
        print(f"Sentiment: {item['sentiment']}")
        print(f"Confidence: {item['confidence']}")
        print("Probabilities:")
        for label, score in item["probabilities"].items():
            print(f"  {label}: {score}")

    print()
    print("Raw JSON")
    print("-" * 45)
    print(json.dumps(result, indent=4))


if __name__ == "__main__":
    user_text = input("Enter feedback text: ")
    output = analyze_aspects(user_text)
    print_aspect_analysis(output)