from typing import Dict, Any


STRONG_POSITIVE_CUES = [
    "excellent",
    "amazing",
    "awesome",
    "fantastic",
    "perfect",
    "love",
    "loved",
    "great",
    "very happy",
    "extremely happy",
    "highly recommend",
    "best",
]

STRONG_NEGATIVE_CUES = [
    "terrible",
    "horrible",
    "worst",
    "awful",
    "hate",
    "hated",
    "very bad",
    "extremely bad",
    "angry",
    "frustrated",
    "disappointed",
    "never again",
    "not responding",
    "nobody helped",
]

MILD_POSITIVE_CUES = [
    "on time",
    "as expected",
    "works as described",
    "received",
    "contained all listed items",
    "no issues",
    "fine",
    "okay",
    "decent",
    "acceptable",
]

MILD_NEUTRAL_CUES = [
    "received yesterday",
    "package contained",
    "order number",
    "listed items",
    "normal",
    "standard",
    "nothing special",
    "no major issues",
]


def contains_any(text: str, phrases: list) -> bool:
    """
    Check whether any phrase exists in the given text.
    """
    text = text.lower()
    return any(phrase in text for phrase in phrases)


def calculate_probability_spread(probabilities: Dict[str, float]) -> float:
    """
    Calculate how spread out the prediction is.
    Higher spread usually means stronger sentiment confidence.
    """
    if not probabilities:
        return 0.0

    scores = list(probabilities.values())
    return round(float(max(scores) - min(scores)), 4)


def analyze_sentiment_intensity(
    text: str,
    sentiment: str,
    confidence: float,
    probabilities: Dict[str, float],
) -> Dict[str, Any]:
    """
    Analyze sentiment intensity.

    Output examples:
    - strong positive
    - mild positive
    - strong negative
    - mild negative
    - factual neutral
    - mixed neutral
    """
    text_lower = str(text).lower()
    sentiment = str(sentiment).lower()

    probability_spread = calculate_probability_spread(probabilities)

    has_strong_positive = contains_any(text_lower, STRONG_POSITIVE_CUES)
    has_strong_negative = contains_any(text_lower, STRONG_NEGATIVE_CUES)
    has_mild_positive = contains_any(text_lower, MILD_POSITIVE_CUES)
    has_mild_neutral = contains_any(text_lower, MILD_NEUTRAL_CUES)

    confidence = float(confidence or 0.0)

    if sentiment == "positive":
        if has_strong_positive:
            intensity = "strong"
            reason = "Strong positive expression detected."
        elif has_mild_positive and not has_strong_positive:
            intensity = "mild"
            reason = "Positive sentiment is based on satisfaction or expectation being met."
        elif confidence >= 0.85 and probability_spread >= 0.70:
            intensity = "strong"
            reason = "Positive prediction has high confidence and clear probability separation."
        elif confidence >= 0.60:
            intensity = "moderate"
            reason = "Positive prediction is reasonably confident."
        else:
            intensity = "mild"
            reason = "Positive prediction has limited confidence."

    elif sentiment == "negative":
        if has_strong_negative:
            intensity = "strong"
            reason = "Strong negative expression detected."
        elif confidence >= 0.85 and probability_spread >= 0.70:
            intensity = "strong"
            reason = "Negative prediction has high confidence and clear probability separation."
        elif confidence >= 0.60:
            intensity = "moderate"
            reason = "Negative prediction is reasonably confident."
        else:
            intensity = "mild"
            reason = "Negative prediction has limited confidence."

    else:
        positive_prob = probabilities.get("positive", 0.0)
        negative_prob = probabilities.get("negative", 0.0)

        if has_mild_neutral:
            intensity = "factual"
            reason = "Text appears mostly factual with no strong emotional expression."
        elif positive_prob > 0.25 or negative_prob > 0.25:
            intensity = "mixed"
            reason = "Neutral prediction contains some positive or negative signal."
        else:
            intensity = "factual"
            reason = "Neutral prediction has low emotional intensity."

    return {
        "sentiment_intensity": intensity,
        "probability_spread": probability_spread,
        "intensity_reason": reason,
    }


if __name__ == "__main__":
    samples = [
        {
            "text": "The service was excellent and I am very happy",
            "sentiment": "positive",
            "confidence": 0.9869,
            "probabilities": {
                "negative": 0.0042,
                "neutral": 0.0089,
                "positive": 0.9869,
            },
        },
        {
            "text": "The delivery arrived on time and the product was as expected.",
            "sentiment": "positive",
            "confidence": 0.9282,
            "probabilities": {
                "negative": 0.0044,
                "neutral": 0.0675,
                "positive": 0.9282,
            },
        },
        {
            "text": "The product was received yesterday and the package contained all listed items.",
            "sentiment": "neutral",
            "confidence": 0.6276,
            "probabilities": {
                "negative": 0.0048,
                "neutral": 0.6276,
                "positive": 0.3676,
            },
        },
        {
            "text": "the airline service was very bad",
            "sentiment": "negative",
            "confidence": 0.9336,
            "probabilities": {
                "negative": 0.9336,
                "neutral": 0.0598,
                "positive": 0.0066,
            },
        },
    ]

    for sample in samples:
        result = analyze_sentiment_intensity(
            text=sample["text"],
            sentiment=sample["sentiment"],
            confidence=sample["confidence"],
            probabilities=sample["probabilities"],
        )

        print()
        print(sample["text"])
        print(result)