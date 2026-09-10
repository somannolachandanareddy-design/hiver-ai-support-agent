HIGH_RISK_TERMS = {
    "fraud",
    "scam",
    "stolen",
    "unauthorized",
    "legal",
    "lawsuit",
    "security breach"
}


def decide_escalation(
    message,
    intent_confidence,
    retrieval_score,
    confidence_threshold=0.55,
    retrieval_threshold=0.35
):
    """
    Decide whether the request should be
    automatically handled or escalated.
    """

    text = message.lower()

    matched_terms = [
        term
        for term in HIGH_RISK_TERMS
        if term in text
    ]

    # High-risk request
    if matched_terms:

        return {
            "decision": "ESCALATE",
            "reason": (
                "Potentially high-risk issue: "
                + ", ".join(matched_terms)
            )
        }

    # Low intent confidence
    if intent_confidence < confidence_threshold:

        return {
            "decision": "ESCALATE",
            "reason": (
                "Low intent-classification confidence."
            )
        }

    # Weak historical evidence
    if retrieval_score < retrieval_threshold:

        return {
            "decision": "ESCALATE",
            "reason": (
                "Insufficiently similar historical evidence."
            )
        }

    # Safe enough to auto-handle
    return {
        "decision": "AUTO-HANDLE",
        "reason": (
            "Intent confidence and historical "
            "evidence meet the required thresholds."
        )
    }