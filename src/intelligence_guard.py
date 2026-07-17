import json
from typing import Dict, Any, List


NEGATIVE_EMOTIONS = {
    "anger",
    "frustration",
    "disappointment",
    "concern",
}

POSITIVE_EMOTIONS = {
    "satisfaction",
}

RISKY_PRIORITY_LEVELS = {
    "high",
    "critical",
}


def count_detected_cues(detected_cues: Dict[str, List[str]]) -> int:
    """
    Count all detected cue phrases across categories.
    """
    if not detected_cues:
        return 0

    return sum(len(cues) for cues in detected_cues.values())


def add_issue(
    issues: List[Dict[str, Any]],
    module: str,
    severity: str,
    message: str,
) -> None:
    """
    Add a structured reliability issue.
    """
    issues.append(
        {
            "module": module,
            "severity": severity,
            "message": message,
        }
    )


def validate_emotion_output(
    sentiment: str,
    emotion_analysis: Dict[str, Any],
) -> List[Dict[str, Any]]:
    """
    Validate emotion output against sentiment and cue support.
    """
    issues = []

    sentiment = str(sentiment).lower()
    emotion_analysis = emotion_analysis or {}

    primary_emotion = str(
        emotion_analysis.get("primary_emotion", "")
    ).lower()

    emotion_confidence = float(
        emotion_analysis.get("emotion_confidence") or 0.0
    )

    detected_cues = emotion_analysis.get("detected_cues", {})
    cue_count = count_detected_cues(detected_cues)

    if sentiment == "negative" and primary_emotion in POSITIVE_EMOTIONS:
        add_issue(
            issues,
            module="emotion",
            severity="high",
            message=(
                "Emotion output conflicts with negative sentiment. "
                "Positive emotion detected for negative feedback."
            ),
        )

    if sentiment == "positive" and primary_emotion in NEGATIVE_EMOTIONS:
        add_issue(
            issues,
            module="emotion",
            severity="high",
            message=(
                "Emotion output conflicts with positive sentiment. "
                "Negative emotion detected for positive feedback."
            ),
        )

    if emotion_confidence < 0.65:
        add_issue(
            issues,
            module="emotion",
            severity="medium",
            message="Emotion confidence is low or weak.",
        )

    if cue_count == 0 and primary_emotion != "neutral":
        add_issue(
            issues,
            module="emotion",
            severity="medium",
            message=(
                "Emotion appears to be inferred from fallback logic "
                "without direct emotion cues."
            ),
        )

    return issues


def validate_aspect_output(
    aspect_analysis: Dict[str, Any],
) -> List[Dict[str, Any]]:
    """
    Validate aspect output for low confidence and suspicious overlap.
    """
    issues = []

    aspect_analysis = aspect_analysis or {}
    aspects = aspect_analysis.get("aspects", [])

    if not aspects:
        add_issue(
            issues,
            module="aspect",
            severity="medium",
            message="No aspect output was produced.",
        )
        return issues

    for aspect in aspects:
        aspect_name = aspect.get("aspect", "Unknown")
        confidence = float(aspect.get("confidence") or 0.0)

        if confidence < 0.60:
            add_issue(
                issues,
                module="aspect",
                severity="medium",
                message=(
                    f"Aspect '{aspect_name}' has low confidence "
                    f"({confidence:.2f})."
                ),
            )

    evidence_list = [
        str(aspect.get("evidence", "")).strip().lower()
        for aspect in aspects
        if aspect.get("evidence")
    ]

    unique_evidence = set(evidence_list)

    if len(aspects) > 1 and len(unique_evidence) == 1:
        add_issue(
            issues,
            module="aspect",
            severity="medium",
            message=(
                "Multiple aspects share the exact same evidence. "
                "Aspect detection may be overlapping."
            ),
        )

    return issues


def validate_priority_output(
    sentiment: str,
    priority_analysis: Dict[str, Any],
) -> List[Dict[str, Any]]:
    """
    Validate priority output against cues, sentiment, and resolution state.
    """
    issues = []

    sentiment = str(sentiment).lower()
    priority_analysis = priority_analysis or {}

    priority_level = str(
        priority_analysis.get("priority_level", "")
    ).lower()

    priority_confidence = str(
        priority_analysis.get("priority_confidence", "")
    ).lower()

    detected_cues = priority_analysis.get("detected_priority_cues", {})
    critical_cues = detected_cues.get("critical", [])
    high_cues = detected_cues.get("high", [])
    resolution_detected = bool(
        priority_analysis.get("resolution_detected", False)
    )

    if priority_level == "critical" and not critical_cues:
        add_issue(
            issues,
            module="priority",
            severity="high",
            message=(
                "Critical priority was assigned without critical-risk cues."
            ),
        )

    if priority_level in RISKY_PRIORITY_LEVELS and sentiment == "positive":
        add_issue(
            issues,
            module="priority",
            severity="high",
            message=(
                "High or critical priority conflicts with positive sentiment."
            ),
        )

    if priority_level in RISKY_PRIORITY_LEVELS and resolution_detected:
        add_issue(
            issues,
            module="priority",
            severity="medium",
            message=(
                "High priority was assigned even though resolution cues "
                "were detected."
            ),
        )

    if priority_level == "high" and not high_cues and not critical_cues:
        add_issue(
            issues,
            module="priority",
            severity="medium",
            message=(
                "High priority was assigned without high or critical cues."
            ),
        )

    if priority_confidence == "low":
        add_issue(
            issues,
            module="priority",
            severity="medium",
            message="Priority confidence is low.",
        )

    return issues


def get_guard_status(issues: List[Dict[str, Any]]) -> str:
    """
    Convert issues into an overall guard status.
    """
    severities = {issue["severity"] for issue in issues}

    if "high" in severities:
        return "review_required"

    if "medium" in severities:
        return "review_optional"

    return "stable"


def generate_guard_reason(
    guard_status: str,
    issues: List[Dict[str, Any]],
) -> str:
    """
    Generate readable guard explanation.
    """
    if not issues:
        return "No major reliability issues detected across intelligence modules."

    high_issues = [
        issue["message"] for issue in issues
        if issue["severity"] == "high"
    ]

    medium_issues = [
        issue["message"] for issue in issues
        if issue["severity"] == "medium"
    ]

    if guard_status == "review_required":
        return "Review required. " + " ".join(high_issues[:3])

    if guard_status == "review_optional":
        return "Optional review suggested. " + " ".join(medium_issues[:3])

    return "Intelligence output appears stable."


def validate_intelligence_outputs(
    text: str,
    sentiment: str,
    reliability: Dict[str, Any],
    aspect_analysis: Dict[str, Any],
    emotion_analysis: Dict[str, Any],
    priority_analysis: Dict[str, Any],
) -> Dict[str, Any]:
    """
    Validate the reliability of Valtrion's business intelligence layers.

    This does not replace the model outputs.
    It audits them and flags risky/uncertain outputs.
    """
    issues = []

    issues.extend(
        validate_emotion_output(
            sentiment=sentiment,
            emotion_analysis=emotion_analysis,
        )
    )

    issues.extend(
        validate_aspect_output(
            aspect_analysis=aspect_analysis,
        )
    )

    issues.extend(
        validate_priority_output(
            sentiment=sentiment,
            priority_analysis=priority_analysis,
        )
    )

    # Also respect the core sentiment reliability layer.
    reliability = reliability or {}
    review_status = str(reliability.get("review_status", "")).lower()

    if review_status == "needs_manual_review":
        add_issue(
            issues,
            module="sentiment",
            severity="medium",
            message="Core sentiment engine recommends manual review.",
        )

    guard_status = get_guard_status(issues)
    guard_reason = generate_guard_reason(
        guard_status=guard_status,
        issues=issues,
    )

    return {
        "guard_status": guard_status,
        "guard_reason": guard_reason,
        "issue_count": len(issues),
        "issues": issues,
    }


def print_guard_result(result: Dict[str, Any]) -> None:
    """
    Pretty-print guard result.
    """
    print()
    print("Intelligence Reliability Guard")
    print("=" * 45)
    print(f"Guard Status: {result['guard_status']}")
    print(f"Issue Count: {result['issue_count']}")
    print(f"Reason: {result['guard_reason']}")

    print()
    print("Issues:")
    if result["issues"]:
        for issue in result["issues"]:
            print(
                f"- [{issue['severity']}] "
                f"{issue['module']}: {issue['message']}"
            )
    else:
        print("- None")

    print()
    print("Raw JSON")
    print("-" * 45)
    print(json.dumps(result, indent=4))


if __name__ == "__main__":
    sample = validate_intelligence_outputs(
        text="I was unhappy with the product.",
        sentiment="negative",
        reliability={
            "review_status": "auto_classified",
        },
        aspect_analysis={
            "aspects": [
                {
                    "aspect": "Product Quality",
                    "confidence": 0.89,
                    "evidence": "I was unhappy with the product",
                }
            ]
        },
        emotion_analysis={
            "primary_emotion": "satisfaction",
            "emotion_confidence": 0.86,
            "detected_cues": {
                "satisfaction": ["happy"],
            },
        },
        priority_analysis={
            "priority_level": "medium",
            "priority_confidence": "medium",
            "resolution_detected": False,
            "detected_priority_cues": {
                "critical": [],
                "high": [],
                "medium": ["unhappy"],
            },
        },
    )

    print_guard_result(sample)