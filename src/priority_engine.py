import json
import re
from typing import Dict, Any, List


CRITICAL_CUES = [
    "fraud",
    "scam",
    "unauthorized transaction",
    "account hacked",
    "data breach",
    "security issue",
    "legal action",
    "unsafe",
    "dangerous",
    "injury",
    "account was hacked",
    "hacked",
]

HIGH_PRIORITY_CUES = [
    "payment failed",
    "money was deducted",
    "money deducted",
    "amount deducted",
    "amount debited",
    "money debited",
    "charged twice",
    "refund not received",
    "account locked",
    "order not delivered",
    "not delivered",
    "broken condition",
    "damaged product",
    "defective product",
    "nobody helped",
    "not responding",
    "no response",
    "urgent",
]

MEDIUM_PRIORITY_CUES = [
    "late",
    "delayed",
    "rude",
    "poor quality",
    "bad experience",
    "not satisfied",
    "unhappy",
    "issue not resolved",
    "slow",
    "error",
    "failed",
    "broken",
    "damaged",
    "poor",
    "very poor",
    "poor product",
]

LOW_PRIORITY_CUES = [
    "okay",
    "fine",
    "average",
    "normal",
    "nothing special",
    "as expected",
    "received",
]

RESOLUTION_CUES = [
    "resolved",
    "fixed",
    "solved",
    "working now",
    "issue resolved",
    "issue fixed",
    "problem solved",
    "refund received",
    "refund processed",
    "processed quickly",
    "no longer",
    "not anymore",
]


IMPORTANT_ASPECTS = {
    "Payment and Refund",
    "Customer Support",
    "App or Website Experience",
}


def phrase_exists(text: str, phrase: str) -> bool:
    """
    Match complete words/phrases only.
    Prevents false matches like 'app' inside 'unhappy'.
    """
    text = str(text).lower()
    phrase = str(phrase).lower().strip()

    pattern = r"\b" + re.escape(phrase) + r"\b"
    return re.search(pattern, text) is not None


def detect_cues(text: str, cues: List[str]) -> List[str]:
    """
    Return matched cues using safe phrase matching.
    """
    return [cue for cue in cues if phrase_exists(text, cue)]


def get_detected_priority_cues(text: str) -> Dict[str, List[str]]:
    """
    Detect priority-related cues.
    """
    return {
        "critical": detect_cues(text, CRITICAL_CUES),
        "high": detect_cues(text, HIGH_PRIORITY_CUES),
        "medium": detect_cues(text, MEDIUM_PRIORITY_CUES),
        "low": detect_cues(text, LOW_PRIORITY_CUES),
        "resolution": detect_cues(text, RESOLUTION_CUES),
    }


def get_negative_aspects(aspect_analysis: Dict[str, Any]) -> List[Dict[str, Any]]:
    """
    Extract negative aspects from aspect analysis.
    """
    aspect_analysis = aspect_analysis or {}
    aspects = aspect_analysis.get("aspects", [])

    return [
        aspect for aspect in aspects
        if str(aspect.get("sentiment", "")).lower() == "negative"
    ]


def get_positive_resolution_effect(
    detected_cues: Dict[str, List[str]],
    sentiment: str,
) -> float:
    """
    Reduce priority when the issue appears resolved or recovered.
    """
    resolution_cues = detected_cues.get("resolution", [])

    if not resolution_cues:
        return 0.0

    sentiment = str(sentiment).lower() if sentiment else ""

    if sentiment == "positive":
        return 3.0

    if sentiment == "neutral":
        return 2.0

    return 1.2


def calculate_priority_score(
    text: str,
    sentiment: str = None,
    reliability: Dict[str, Any] = None,
    emotion_analysis: Dict[str, Any] = None,
    aspect_analysis: Dict[str, Any] = None,
) -> Dict[str, Any]:
    """
    Hybrid priority scoring using:
    - urgency cues
    - sentiment
    - reliability
    - emotion
    - aspect analysis
    - resolution/recovery cues
    """
    detected_cues = get_detected_priority_cues(text)

    score = 0.0
    reasons = []
    evidence_sources = set()

    # Explicit cue scoring
    if detected_cues["critical"]:
        score += 7.0
        evidence_sources.add("critical_cues")
        reasons.append(
            "Critical risk cues detected: "
            + ", ".join(detected_cues["critical"])
            + "."
        )

    if detected_cues["high"]:
        high_cue_score = 3.0 + (0.8 * (len(detected_cues["high"]) - 1))
        score += high_cue_score
        evidence_sources.add("high_priority_cues")
        reasons.append(
            "High-priority cues detected: "
            + ", ".join(detected_cues["high"])
            + "."
        )

    if detected_cues["medium"]:
        medium_cue_score = 1.6 + (0.4 * (len(detected_cues["medium"]) - 1))
        score += medium_cue_score
        evidence_sources.add("medium_priority_cues")
        reasons.append(
            "Medium-priority cues detected: "
            + ", ".join(detected_cues["medium"])
            + "."
        )

    # Sentiment signal
    sentiment = str(sentiment).lower() if sentiment else ""

    if sentiment == "negative":
        score += 1.0
        evidence_sources.add("sentiment")
        reasons.append("Overall sentiment is negative.")
    elif sentiment == "neutral":
        score += 0.3
        evidence_sources.add("sentiment")
        reasons.append("Overall sentiment is neutral.")
    elif sentiment == "positive":
        score -= 0.5
        reasons.append("Overall sentiment is positive, reducing urgency.")

    # Reliability signal
    reliability = reliability or {}
    confidence_level = str(reliability.get("confidence_level", "")).lower()
    review_status = str(reliability.get("review_status", "")).lower()

    if review_status == "needs_manual_review":
        score += 0.5
        evidence_sources.add("reliability")
        reasons.append("Prediction requires manual review.")

    if confidence_level == "low":
        score += 0.3
        evidence_sources.add("reliability")
        reasons.append("Prediction confidence is low.")

    # Emotion signal
    emotion_analysis = emotion_analysis or {}
    primary_emotion = str(emotion_analysis.get("primary_emotion", "")).lower()
    emotion_strength = str(emotion_analysis.get("emotion_strength", "")).lower()

    if primary_emotion in {"anger", "frustration", "concern"}:
        score += 1.2
        evidence_sources.add("emotion")
        reasons.append(f"Primary emotion indicates {primary_emotion}.")

    if emotion_strength == "strong":
        score += 0.9
        evidence_sources.add("emotion_strength")
        reasons.append("Emotion strength is strong.")
    elif emotion_strength == "moderate":
        score += 0.4
        evidence_sources.add("emotion_strength")
        reasons.append("Emotion strength is moderate.")

    # Aspect signal
    aspect_analysis = aspect_analysis or {}
    negative_aspects = get_negative_aspects(aspect_analysis)

    if len(negative_aspects) >= 2:
        score += 1.0
        evidence_sources.add("aspect")
        reasons.append("Multiple negative aspects detected.")
    elif len(negative_aspects) == 1:
        score += 0.5
        evidence_sources.add("aspect")
        reasons.append("One negative aspect detected.")

    for aspect in negative_aspects:
        aspect_name = aspect.get("aspect")

        if aspect_name in IMPORTANT_ASPECTS:
            score += 1.0
            evidence_sources.add("important_aspect")
            reasons.append(
                f"Negative issue detected in important aspect: {aspect_name}."
            )

    # Resolution/recovery reduction
    resolution_reduction = get_positive_resolution_effect(
        detected_cues=detected_cues,
        sentiment=sentiment,
    )

    if resolution_reduction > 0:
        score -= resolution_reduction
        evidence_sources.add("resolution")
        reasons.append(
            "Resolution/recovery cues detected: "
            + ", ".join(detected_cues["resolution"])
            + ". Priority reduced."
        )

    score = max(score, 0.0)

    return {
        "priority_score": round(float(score), 2),
        "detected_priority_cues": detected_cues,
        "priority_reasons": reasons,
        "evidence_sources": sorted(evidence_sources),
        "resolution_detected": bool(detected_cues["resolution"]),
    }


def get_priority_level(score: float) -> str:
    """
    Convert score into priority level.

    Critical is reserved for safety, security, fraud, legal, or severe account-risk cases.
    High is reserved for strong operational/customer-impact cases.
    Medium is for negative but non-urgent complaints.
    """
    if score >= 7.0:
        return "critical"

    if score >= 5.0:
        return "high"

    if score >= 2.0:
        return "medium"

    return "low"



def get_priority_confidence(
    priority_score: float,
    evidence_sources: List[str],
    detected_cues: Dict[str, List[str]],
) -> str:
    """
    Determine confidence in the priority decision.
    """
    evidence_count = len(evidence_sources)

    if detected_cues.get("critical"):
        return "high"
    
    if len(detected_cues.get("high", [])) >= 2:
        return "high"

    if priority_score >= 5.0 and evidence_count >= 3:
        return "high"

    if priority_score >= 2.5 and evidence_count >= 2:
        return "medium"

    if priority_score < 2.5 and evidence_count <= 1:
        return "medium"

    return "low"


def get_recommended_action(
    priority_level: str,
    aspect_analysis: Dict[str, Any] = None,
    detected_cues: Dict[str, List[str]] = None,
) -> str:
    """
    Generate business action based on priority and detected issue type.
    """
    aspect_analysis = aspect_analysis or {}
    detected_cues = detected_cues or {}

    negative_aspects = get_negative_aspects(aspect_analysis)
    negative_aspect_names = {aspect.get("aspect") for aspect in negative_aspects}

    if priority_level == "critical":
        return "Immediate escalation required. Assign to senior support or risk team."

    if "Payment and Refund" in negative_aspect_names:
        return "Escalate to billing/payment support and verify the transaction or refund status."

    if "Customer Support" in negative_aspect_names:
        return "Escalate to customer support lead and review agent response quality."

    if "App or Website Experience" in negative_aspect_names:
        return "Assign to technical support or product engineering for investigation."

    if "Delivery" in negative_aspect_names:
        return "Assign to logistics/support team and investigate delivery status."

    if "Product Quality" in negative_aspect_names:
        return "Assign to product quality team and check replacement or return eligibility."

    if priority_level == "high":
        return "Escalate to the responsible support team and resolve as soon as possible."

    if priority_level == "medium":
        return "Review and assign to the relevant team during normal support workflow."

    return "No urgent action required. Track for trend analysis."


def generate_priority_reason(
    priority_level: str,
    reasons: List[str],
) -> str:
    """
    Create readable priority explanation.
    """
    if reasons:
        return " ".join(reasons)

    if priority_level == "low":
        return "No major urgency signals were detected."

    return "Priority was determined from sentiment, emotion, aspect, and urgency signals."


def analyze_priority(
    text: str,
    sentiment: str = None,
    reliability: Dict[str, Any] = None,
    emotion_analysis: Dict[str, Any] = None,
    aspect_analysis: Dict[str, Any] = None,
) -> Dict[str, Any]:
    """
    Analyze feedback priority / urgency.
    """
    if not text or not str(text).strip():
        raise ValueError("Input text cannot be empty.")

    score_result = calculate_priority_score(
        text=text,
        sentiment=sentiment,
        reliability=reliability,
        emotion_analysis=emotion_analysis,
        aspect_analysis=aspect_analysis,
    )

    priority_score = score_result["priority_score"]

    critical_cues = score_result["detected_priority_cues"].get("critical", [])

    # Critical priority should be reserved for true risk/security/legal/safety cues.
    # If no critical cue is present, cap the case at high priority.
    if not critical_cues and priority_score >= 7.0:
        priority_score = 6.9

    priority_score = min(max(priority_score, 0.0), 10.0)
    priority_score = round(priority_score, 1)
    priority_level = get_priority_level(priority_score)

    priority_confidence = get_priority_confidence(
        priority_score=priority_score,
        evidence_sources=score_result["evidence_sources"],
        detected_cues=score_result["detected_priority_cues"],
    )

    priority_reason = generate_priority_reason(
        priority_level=priority_level,
        reasons=score_result["priority_reasons"],
    )

    recommended_action = get_recommended_action(
        priority_level=priority_level,
        aspect_analysis=aspect_analysis,
        detected_cues=score_result["detected_priority_cues"],
    )

    return {
        "priority_level": priority_level,
        "priority_score": priority_score,
        "priority_confidence": priority_confidence,
        "priority_reason": priority_reason,
        "recommended_action": recommended_action,
        "resolution_detected": score_result["resolution_detected"],
        "evidence_sources": score_result["evidence_sources"],
        "detected_priority_cues": score_result["detected_priority_cues"],
    }


def print_priority_result(result: Dict[str, Any]) -> None:
    """
    Pretty-print priority result.
    """
    print()
    print("Priority / Urgency Detection")
    print("=" * 45)
    print(f"Priority Level: {result['priority_level']}")
    print(f"Priority Score: {result['priority_score']}")
    print(f"Priority Confidence: {result['priority_confidence']}")
    print(f"Resolution Detected: {result['resolution_detected']}")
    print(f"Reason: {result['priority_reason']}")
    print(f"Recommended Action: {result['recommended_action']}")

    print()
    print("Evidence Sources:")
    for source in result["evidence_sources"]:
        print(f"  - {source}")

    print()
    print("Detected Priority Cues:")
    for level, cues in result["detected_priority_cues"].items():
        print(f"  {level}: {cues}")

    print()
    print("Raw JSON")
    print("-" * 45)
    print(json.dumps(result, indent=4))


if __name__ == "__main__":
    user_text = input("Enter feedback text: ")
    user_sentiment = input("Enter sentiment if known, else press Enter: ").strip()

    output = analyze_priority(
        text=user_text,
        sentiment=user_sentiment if user_sentiment else None,
    )

    print_priority_result(output)