# app/classification/stance_classifier.py

import json
import logging
from typing import Dict, Optional

from app.retrieval.models import Paper
from app.classification.prompt import build_stance_prompt
from app.llm.router import call_llm

logger = logging.getLogger(__name__)


def _safe_fallback() -> Dict:
    return {
        "stance": "neutral",
        "confidence": 0.3,
        "reason": "Abstract missing or insufficient for classification",
    }


def _extract_json(text: str) -> str:
    start = text.find("{")
    end = text.rfind("}")
    if start == -1 or end == -1:
        raise ValueError("No JSON found")
    return text[start : end + 1]


def _parse_response(response: str) -> Optional[Dict]:
    """
    Parse and validate LLM JSON response (robust to messy outputs).
    """
    try:
        # Try direct parse
        try:
            data = json.loads(response)
        except Exception:
            raw = _extract_json(response)
            data = json.loads(raw)

        # Normalize stance
        stance = str(data.get("stance", "")).lower().strip()

        if stance not in ["supports", "contradicts", "neutral"]:
            if "support" in stance:
                stance = "supports"
            elif "contradict" in stance or "oppose" in stance:
                stance = "contradicts"
            else:
                stance = "neutral"

        data["stance"] = stance

        # Confidence
        try:
            conf = float(data.get("confidence", 0.5))
        except Exception:
            conf = 0.5

        conf = max(0.0, min(1.0, conf))
        data["confidence"] = conf

        # Reason
        reason = str(data.get("reason", "")).strip()
        if not reason:
            reason = "No clear reasoning provided"

        reason = ". ".join(reason.split(". ")[:2]).strip()
        data["reason"] = reason

        return data

    except Exception as e:
        logger.error("Parsing failed: %s | Response: %s", e, response)
        return None


def _apply_consistency_correction(parsed: Dict, claim: str) -> Dict:
    """
    Ensure consistency between reasoning and stance using general logic.
    """

    reason = parsed.get("reason", "").lower()
    stance = parsed.get("stance", "neutral")
    confidence = parsed.get("confidence", 0.5)

    # If model says neutral but still discusses relationship/effect → weak support
    implication_signals = ["affect", "impact", "effect", "influence", "change"]

    mentions_relationship = any(word in reason for word in implication_signals)

    # Upgrade neutral → supports if relationship is acknowledged
    if stance == "neutral" and mentions_relationship:
        parsed["stance"] = "supports"
        parsed["confidence"] = max(confidence, 0.55)

    return parsed


def classify_stance(paper: Paper, claim: str) -> Dict:
    """
    Classify stance of a single paper relative to a claim.
    """

    if not paper or not paper.abstract or not paper.abstract.strip():
        return _safe_fallback()

    abstract = paper.abstract[:1500]
    prompt = build_stance_prompt(claim, abstract)

    for attempt in range(2):
        try:
            response = call_llm(prompt)

            print("\n====================")
            print("RAW RESPONSE:\n", response)
            print("====================\n")

            if not response or not response.strip():
                raise ValueError("Empty response")

            parsed = _parse_response(response)

            if parsed:
                # ✅ Apply general consistency correction
                parsed = _apply_consistency_correction(parsed, claim)

                logger.info(
                    "Paper %s → %s (confidence=%.2f)",
                    paper.paper_id,
                    parsed["stance"],
                    parsed["confidence"],
                )

                return parsed

        except Exception as e:
            logger.error("LLM call failed: %s", e)

    return _safe_fallback()
