from typing import Dict, Any


def calculate_decision_margin(probabilities: Dict[str, float]) -> float:
    """
    Calculate the gap between the highest and second-highest class probabilities.
    Higher margin means the model is more decisive.
    """
    if not probabilities or len(probabilities) < 2:
        return 0.0

    sorted_scores = sorted(probabilities.values(), reverse=True)
    margin = sorted_scores[0] - sorted_scores[1]

    return round(float(margin), 4)


def get_confidence_level(confidence: float, decision_margin: float) -> str:
    """
    Convert model confidence and margin into a readable confidence level.
    """
    if confidence >= 0.85 and decision_margin >= 0.30:
        return "high"

    if confidence >= 0.60 and decision_margin >= 0.15:
        return "medium"

    return "low"


def get_review_status(
    confidence_level: str,
    model_agreement: bool,
    recommended_sentiment: str,
) -> str:
    """
    Decide whether the prediction can be trusted automatically
    or should be reviewed manually.
    """
    if confidence_level == "high" and model_agreement:
        return "auto_classified"

    if confidence_level == "high" and not model_agreement:
        return "auto_classified_with_model_disagreement"

    if confidence_level == "medium":
        return "review_optional"

    return "needs_manual_review"


def get_reliability_reason(
    confidence_level: str,
    decision_margin: float,
    model_agreement: bool,
) -> str:
    """
    Generate a simple explanation for the reliability decision.
    """
    if confidence_level == "high" and model_agreement:
        return (
            "Both models agree and the recommended model has high confidence."
        )

    if confidence_level == "high" and not model_agreement:
        return (
            "The transformer model has high confidence, but the classical "
            "baseline disagrees. Transformer result is recommended."
        )

    if confidence_level == "medium":
        return (
            "The recommended model is reasonably confident, but the result "
            "may benefit from optional review."
        )

    return (
        "The prediction confidence or decision margin is low. Manual review "
        "is recommended."
    )


def analyze_reliability(
    recommended_sentiment: str,
    recommended_confidence: float,
    recommended_probabilities: Dict[str, float],
    model_agreement: bool,
) -> Dict[str, Any]:
    """
    Analyze reliability of the recommended sentiment prediction.
    """
    decision_margin = calculate_decision_margin(recommended_probabilities)

    confidence_level = get_confidence_level(
        confidence=float(recommended_confidence),
        decision_margin=decision_margin,
    )

    review_status = get_review_status(
        confidence_level=confidence_level,
        model_agreement=model_agreement,
        recommended_sentiment=recommended_sentiment,
    )

    reliability_reason = get_reliability_reason(
        confidence_level=confidence_level,
        decision_margin=decision_margin,
        model_agreement=model_agreement,
    )

    return {
        "confidence_level": confidence_level,
        "decision_margin": decision_margin,
        "review_status": review_status,
        "reliability_reason": reliability_reason,
    }


if __name__ == "__main__":
    sample_probabilities = {
        "negative": 0.1265,
        "neutral": 0.6283,
        "positive": 0.2451,
    }

    result = analyze_reliability(
        recommended_sentiment="neutral",
        recommended_confidence=0.6283,
        recommended_probabilities=sample_probabilities,
        model_agreement=False,
    )

    print(result)