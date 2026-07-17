import re
import json
from typing import Dict, Any, List, Tuple
from unittest import result

def phrase_exists(text: str, phrase: str) -> bool:
    """
    Match full words/phrases only.
    Prevents errors like detecting 'happy' inside 'unhappy'.
    """
    text = str(text).lower()
    phrase = str(phrase).lower().strip()

    pattern = r"\b" + re.escape(phrase) + r"\b"
    return re.search(pattern, text) is not None

def is_negated_positive_cue(text: str, phrase: str) -> bool:
    """
    Prevent positive emotion cues from being detected when they are negated.

    Examples:
    - not satisfied
    - not happy
    - not good
    - never satisfied
    - wasn't happy
    """
    text = text.lower()
    phrase = phrase.lower()

    negation_pattern = (
        r"\b("
        r"not|no|never|hardly|barely|"
        r"isn't|wasn't|aren't|weren't|"
        r"don't|doesn't|didn't|can't|cannot"
        r")\b"
        r"(?:\s+\w+){0,3}"
        r"\s+"
        + re.escape(phrase)
        + r"\b"
    )

    return re.search(negation_pattern, text) is not None



EMOTION_CUES: Dict[str, List[Tuple[str, float]]] = {
    "anger": [
        ("angry", 2.5),
        ("furious", 2.8),
        ("mad", 2.0),
        ("hate", 2.5),
        ("horrible", 2.0),
        ("worst", 2.5),
        ("rude", 2.0),
        ("unacceptable", 2.2),
        ("never again", 2.6),
    ],
    "frustration": [
        ("frustrated", 2.8),
        ("nobody helped", 2.6),
        ("not responding", 2.5),
        ("no response", 2.4),
        ("kept waiting", 2.0),
        ("again and again", 1.8),
        ("still not fixed", 2.4),
        ("issue not resolved", 2.4),
        ("support did not help", 2.6),
    ],
    "disappointment": [
        ("disappointed", 2.8),
        ("not satisfied", 2.4),
        ("expected better", 2.2),
        ("poor quality", 2.3),
        ("bad experience", 2.2),
        ("let down", 2.3),
        ("not worth", 2.0),
        ("waste of money", 2.7),
        ("unhappy", 2.6),
        ("broken", 2.2),
        ("broken condition", 2.5),
        ("damaged product", 2.4),
    ],
    "confusion": [
        ("confused", 2.6),
        ("unclear", 2.0),
        ("do not understand", 2.4),
        ("don't understand", 2.4),
        ("not sure", 1.8),
        ("where is", 1.6),
        ("what happened", 2.0),
        ("why", 1.2),
    ],
    "concern": [
        ("worried", 2.5),
        ("concerned", 2.4),
        ("money deducted", 2.6),
        ("payment failed", 2.2),
        ("refund not received", 2.6),
        ("account locked", 2.4),
        ("security", 2.0),
        ("risk", 1.8),
        ("money was deducted", 2.6),
        ("amount deducted", 2.4),
        ("money debited", 2.4),
        ("amount debited", 2.4),
        ("account hacked", 2.8),
        ("account was hacked", 2.8),
        ("hacked", 2.6),
        ("unauthorized transaction", 2.8),
        ("fraud", 2.8),
        ("scam", 2.6),
    ],
    "satisfaction": [
        ("happy", 2.5),
        ("very happy", 2.8),
        ("satisfied", 2.5),
        ("excellent", 2.4),
        ("amazing", 2.4),
        ("great", 2.0),
        ("love", 2.5),
        ("perfect", 2.3),
        ("highly recommend", 2.7),
        ("on time", 1.5),
        ("as expected", 1.4),
    ],
}


def find_emotion_cues(text: str) -> Dict[str, List[str]]:
    """
    Detect emotion cues in the given text.
    """
    text_lower = str(text).lower()
    detected = {}

    for emotion, cues in EMOTION_CUES.items():
        for phrase, _weight in cues:
            if phrase_exists(text_lower, phrase):
              if emotion == "satisfaction" and is_negated_positive_cue(text_lower, phrase):
                continue

              detected.setdefault(emotion, []).append(phrase)

    return detected


def score_emotions(
    text: str,
    sentiment: str = None,
) -> Dict[str, float]:
    """
    Score each emotion using keyword cues and sentiment fallback.
    """
    text_lower = str(text).lower()
    scores = {
        "anger": 0.0,
        "frustration": 0.0,
        "disappointment": 0.0,
        "confusion": 0.0,
        "concern": 0.0,
        "satisfaction": 0.0,
        "neutral": 0.0,
    }

    for emotion, cues in EMOTION_CUES.items():
        for phrase, weight in cues:
            if phrase_exists(text_lower, phrase):
                if emotion == "satisfaction" and is_negated_positive_cue(text_lower, phrase):
                    continue

                scores[emotion] += weight

    sentiment = str(sentiment).lower() if sentiment else None

    # Sentiment-based fallback.
    if sentiment == "positive":
        scores["satisfaction"] += 1.0
    elif sentiment == "negative":
        scores["disappointment"] += 0.8
    elif sentiment == "neutral":
        scores["neutral"] += 1.0

    # Extra business-context rules.
    if phrase_exists(text_lower, "late") or phrase_exists(text_lower, "delayed"):
        scores["frustration"] += 1.2

    if phrase_exists(text_lower, "rude") or phrase_exists(text_lower, "worst"):
        scores["anger"] += 1.0

    if phrase_exists(text_lower, "failed") or phrase_exists(text_lower, "error"):
        scores["concern"] += 1.0

    return scores


def get_emotion_strength(score: float) -> str:
    """
    Convert raw score into readable emotion strength.
    """
    if score >= 4.0:
        return "strong"

    if score >= 2.0:
        return "moderate"

    if score > 0:
        return "mild"

    return "none"


def calculate_emotion_confidence(score: float) -> float:
    """
    Convert rule-based emotion score into a confidence-like value.
    This is not a neural model probability. It is an interpretive confidence score.
    """
    if score <= 0:
        return 0.5

    confidence = 0.45 + min(score / 6.0, 0.5)
    return round(float(confidence), 4)

def get_secondary_emotions(
    scores: Dict[str, float],
    primary_emotion: str,
    max_secondary: int = 2,
) -> List[Dict[str, Any]]:
    """
    Return secondary emotions that have meaningful scores
    but are not the primary emotion.
    """
    secondary_candidates = []

    for emotion, score in scores.items():
        if emotion == primary_emotion:
            continue

        if emotion == "neutral":
            continue

        if score >=1.0:
            secondary_candidates.append(
                {
                    "emotion": emotion,
                    "score": round(float(score), 4),
                    "strength": get_emotion_strength(score),
                    "confidence": calculate_emotion_confidence(score),
                }
            )

    secondary_candidates = sorted(
        secondary_candidates,
        key=lambda item: item["score"],
        reverse=True,
    )

    return secondary_candidates[:max_secondary]


def generate_emotion_reason(
    emotion: str,
    detected_cues: Dict[str, List[str]],
    sentiment: str = None,
) -> str:
    """
    Create a readable reason for the detected emotion.
    """
    cues = detected_cues.get(emotion, [])

    if cues:
        cue_text = ", ".join(cues[:5])
        return f"Detected emotion cues related to {emotion}: {cue_text}."

    if emotion == "satisfaction":
        return "Positive sentiment suggests customer satisfaction."

    if emotion == "disappointment":
        return "Negative sentiment suggests customer disappointment."

    if emotion == "neutral":
        return "The feedback appears mostly factual or emotionally neutral."

    return f"The feedback pattern suggests {emotion}."


def detect_emotion(
    text: str,
    sentiment: str = None,
) -> Dict[str, Any]:
    """
    Detect the primary emotion behind a feedback text.

    Returns:
    {
        "primary_emotion": "frustration",
        "emotion_strength": "moderate",
        "emotion_confidence": 0.78,
        "detected_cues": {...},
        "emotion_scores": {...},
        "emotion_reason": "..."
    }
    """
    if not text or not str(text).strip():
        raise ValueError("Input text cannot be empty.")

    scores = score_emotions(text=text, sentiment=sentiment)
    detected_cues = find_emotion_cues(text)

    primary_emotion = max(scores, key=scores.get)
    primary_score = scores[primary_emotion]

    if primary_score == 0:
        primary_emotion = "neutral"
        primary_score = 1.0

    emotion_strength = get_emotion_strength(primary_score)
    emotion_confidence = calculate_emotion_confidence(primary_score)

    secondary_emotions = get_secondary_emotions(
        scores=scores,
        primary_emotion=primary_emotion,
    )

    emotion_reason = generate_emotion_reason(
        emotion=primary_emotion,
        detected_cues=detected_cues,
        sentiment=sentiment,
    )
    
    return {
        "primary_emotion": primary_emotion,
        "emotion_strength": emotion_strength,
        "emotion_confidence": emotion_confidence,
        "secondary_emotions": secondary_emotions,
        "detected_cues": detected_cues,
        "emotion_scores": {
            emotion: round(float(score), 4)
            for emotion, score in scores.items()
        },
        "emotion_reason": emotion_reason,
        "secondary_emotions": secondary_emotions,
    }


def print_emotion_result(result: Dict[str, Any]) -> None:
    """
    Pretty-print emotion detection result.
    """
    print()
    print("Emotion Detection")
    print("=" * 45)
    print(f"Primary Emotion: {result['primary_emotion']}")
    print(f"Emotion Strength: {result['emotion_strength']}")
    print(f"Emotion Confidence: {result['emotion_confidence']}")
    print(f"Reason: {result['emotion_reason']}")
    print()
    print("Secondary Emotions:")
    secondary_emotions = result.get("secondary_emotions", [])

    if secondary_emotions:
        for item in secondary_emotions:
            print(
                f"  {item['emotion']} "
                f"({item['strength']}, confidence={item['confidence']})"
            )
    else:
        print("  None")

    print()
    print("Detected Cues:")
    for emotion, cues in result["detected_cues"].items():
        print(f"  {emotion}: {cues}")

    print()
    print("Raw JSON")
    print("-" * 45)
    print(json.dumps(result, indent=4))


if __name__ == "__main__":
    user_text = input("Enter feedback text: ")
    user_sentiment = input("Enter sentiment if known, else press Enter: ").strip()

    output = detect_emotion(
        text=user_text,
        sentiment=user_sentiment if user_sentiment else None,
    )

    print_emotion_result(output)