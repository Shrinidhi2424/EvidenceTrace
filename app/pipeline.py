import logging
import time
from typing import Dict

from app.calibration import calibrate_confidence
from app.classification import classify_stance
from app.retrieval import fetch_papers
from app.retrieval.quality_scorer import score_papers

logger = logging.getLogger(__name__)


class NoPapersFound(Exception):
    """Both paper sources returned nothing for this claim."""


def _ms(start: float) -> float:
    return round((time.perf_counter() - start) * 1000, 1)


def analyze_claim(claim: str, max_papers: int = 8) -> Dict:
    """
    claim -> fetch papers -> score papers -> classify stance of the top
    max_papers -> calibrate confidence. Returns a JSON-serialisable report.
    """
    timings = {}

    t = time.perf_counter()
    papers = fetch_papers(claim)
    timings["retrieval"] = _ms(t)

    if not papers:
        raise NoPapersFound("Semantic Scholar and arXiv returned no papers for this claim.")

    t = time.perf_counter()
    scored = score_papers(papers, claim)
    timings["scoring"] = _ms(t)

    top = scored[:max_papers]

    t = time.perf_counter()
    paper_rows = []
    evidence = []
    for paper in top:
        result = classify_stance(paper, claim)
        paper_rows.append(
            {
                "title": paper.title,
                "year": paper.year,
                "source": paper.source,
                "citation_count": paper.citation_count or 0,
                "paper_id": paper.paper_id,
                "quality_score": paper.score,
                "score_components": paper.score_components,
                "stance": result["stance"],
                "stance_confidence": result["confidence"],
                "evidence_sentence": result["reason"],
            }
        )
        evidence.append(
            {
                "stance": result["stance"],
                "confidence": result["confidence"],
                "paper_id": paper.paper_id,
                "citation_count": paper.citation_count or 0,
                "year": paper.year,
                "score": paper.score,
                "source": paper.source,
            }
        )
    timings["classification"] = _ms(t)

    t = time.perf_counter()
    calibrated = calibrate_confidence(evidence)
    timings["calibration"] = _ms(t)

    return {
        "claim": claim,
        "verdict": calibrated["final_decision"],
        "confidence_score": calibrated["confidence_score"],
        "confidence_label": calibrated["confidence_label"],
        "evidence_distribution": calibrated["evidence_distribution"],
        "evidence_counts": calibrated["evidence_counts"],
        "conflicts": calibrated["conflicts"],
        "papers": paper_rows,
        "timings_ms": timings,
    }
