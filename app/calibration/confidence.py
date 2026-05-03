import logging
from typing import List, Dict
from app.calibration.models import EvidenceItem

logger = logging.getLogger(__name__)


def compute_item_weight(item: EvidenceItem) -> float:
    """
    Weight = model confidence × retrieval quality
    """
    return item.confidence * item.score


def compute_directional_scores(items: List[EvidenceItem]) -> Dict[str, float]:
    """
    Compute normalized weighted stance scores.
    """
    support, contradict, neutral = 0.0, 0.0, 0.0

    weights = []
    for item in items:
        w = compute_item_weight(item)
        weights.append(w)

        if item.stance == "supports":
            support += w
        elif item.stance == "contradicts":
            contradict += w
        else:
            neutral += w

    total = support + contradict + neutral

    if total == 0:
        return {"supports": 0.0, "contradicts": 0.0, "neutral": 1.0}

    return {
        "supports": support / total,
        "contradicts": contradict / total,
        "neutral": neutral / total,
    }


def compute_agreement(scores: Dict[str, float]) -> float:
    """
    Agreement = difference between support and contradiction.
    """
    return abs(scores["supports"] - scores["contradicts"])


def compute_diversity(items: List[EvidenceItem]) -> float:
    """
    Unique paper ratio.
    """
    if not items:
        return 0.0

    unique_ids = set(i.paper_id for i in items)
    return len(unique_ids) / len(items)


def detect_conflicts(items: List[EvidenceItem]) -> List[Dict]:
    """
    Detect high-confidence conflicting evidence.
    """
    conflicts = []
    seen = set()

    supports = [i for i in items if i.stance == "supports" and i.confidence > 0.7]
    contradicts = [i for i in items if i.stance == "contradicts" and i.confidence > 0.7]

    for s in supports:
        for c in contradicts:
            if s.paper_id == c.paper_id:
                continue

            # enforce minimum credibility
            if s.citation_count < 20 or c.citation_count < 20:
                continue

            key = tuple(sorted([s.paper_id, c.paper_id]))
            if key in seen:
                continue

            seen.add(key)

            severity = (
                "high" if (s.confidence > 0.85 and c.confidence > 0.85) else "moderate"
            )

            conflicts.append(
                {"paper_a": s.paper_id, "paper_b": c.paper_id, "severity": severity}
            )

    return conflicts


def compute_confidence(
    scores: Dict[str, float], items: List[EvidenceItem], conflicts: List[Dict]
) -> float:
    """
    Final calibrated confidence score.
    """

    if not items:
        return 0.0

    agreement = compute_agreement(scores)
    diversity = compute_diversity(items)

    avg_conf = sum(i.confidence for i in items) / len(items)

    # weighted formula (aligned with system design)
    confidence = (
        0.4 * agreement + 0.3 * avg_conf + 0.2 * scores["supports"] + 0.1 * diversity
    )

    # conflict penalty
    if conflicts:
        confidence *= 0.85

    return round(min(0.95, max(0.0, confidence)), 4)


def calibrate_confidence(items: List[Dict]) -> Dict:
    """
    Main calibration function.
    """

    if not items:
        return {
            "final_decision": "neutral",
            "confidence_score": 0.0,
            "confidence_label": "very low",
            "evidence_distribution": {"supports": 0, "contradicts": 0, "neutral": 1},
            "agreement_score": 0.0,
            "diversity_score": 0.0,
            "conflicts": [],
        }

    evidence = [
        EvidenceItem(
            stance=i["stance"],
            confidence=i["confidence"],
            paper_id=i["paper_id"],
            citation_count=i.get("citation_count", 0),
            year=i.get("year"),
            score=i.get("score", 0.0),
        )
        for i in items
    ]

    scores = compute_directional_scores(evidence)
    agreement = compute_agreement(scores)
    diversity = compute_diversity(evidence)

    conflicts = detect_conflicts(evidence)

    confidence_score = compute_confidence(scores, evidence, conflicts)

    # decision logic (avoid neutral dominance unless strong)
    if scores["supports"] > scores["contradicts"]:
        final_decision = "supports"
    elif scores["contradicts"] > scores["supports"]:
        final_decision = "contradicts"
    else:
        final_decision = "neutral"

    # confidence label
    if confidence_score >= 0.75:
        label = "high"
    elif confidence_score >= 0.5:
        label = "moderate"
    elif confidence_score >= 0.3:
        label = "low"
    else:
        label = "very low"

    return {
        "final_decision": final_decision,
        "confidence_score": confidence_score,
        "confidence_label": label,
        "evidence_distribution": scores,
        "agreement_score": round(agreement, 4),
        "diversity_score": round(diversity, 4),
        "conflicts": conflicts,
    }
