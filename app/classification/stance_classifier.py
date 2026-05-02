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
        "confidence": 0.0,
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
    Parse and validate LLM JSON response.
    """
    try:
        raw = _extract_json(response)
        data = json.loads(raw)

        stance = str(data.get("stance", "")).lower().strip()
        if stance not in ["supports", "contradicts", "neutral"]:
            raise ValueError("Invalid stance")
        data["stance"] = stance

        conf = float(data.get("confidence", 0.0))
        conf = max(0.0, min(1.0, conf))
        data["confidence"] = conf

        reason = str(data.get("reason", "")).strip()
        reason = ". ".join(reason.split(". ")[:2]).strip()
        data["reason"] = reason

        return data

    except Exception as e:
        logger.error("Parsing failed: %s | Response: %s", e, response)
        return None


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
            # NOTE: timeout support depends on router implementation
            response = call_llm(prompt)

            if not response or not response.strip():
                raise ValueError("Empty response")

            parsed = _parse_response(response)

            if parsed:
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
