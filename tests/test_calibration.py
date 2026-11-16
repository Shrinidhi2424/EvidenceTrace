import pytest

from app.calibration import calibrate_confidence
from app.calibration.confidence import (
    compute_agreement,
    compute_confidence,
    compute_directional_scores,
    compute_evidence_breadth,
    detect_conflicts,
)
from app.calibration.models import EvidenceItem


def item(stance, conf=0.9, pid="p", cites=0, score=0.8, source="semantic_scholar"):
    return EvidenceItem(
        stance=stance,
        confidence=conf,
        paper_id=pid,
        citation_count=cites,
        year=2022,
        score=score,
        source=source,
    )


def test_directional_scores_sum_to_one():
    items = [
        item("supports", 0.9, "a", score=0.8),
        item("contradicts", 0.7, "b", score=0.6),
        item("neutral", 0.5, "c", score=0.4),
    ]
    scores = compute_directional_scores(items)
    assert sum(scores.values()) == pytest.approx(1.0)
    assert scores["supports"] > scores["contradicts"] > scores["neutral"]


def test_directional_scores_zero_weight_is_all_neutral():
    assert compute_directional_scores([item("supports", 0.0, score=0.0)]) == {
        "supports": 0.0,
        "contradicts": 0.0,
        "neutral": 1.0,
    }


def test_agreement_is_gap_between_support_and_contradiction():
    assert compute_agreement({"supports": 0.7, "contradicts": 0.2, "neutral": 0.1}) == pytest.approx(0.5)
    assert compute_agreement({"supports": 0.4, "contradicts": 0.4, "neutral": 0.2}) == 0.0


def test_evidence_breadth_counts_non_neutral_and_caps_at_one():
    assert compute_evidence_breadth([]) == 0.0
    assert compute_evidence_breadth([item("neutral")] * 4) == 0.0
    mixed = [item("supports"), item("contradicts"), item("neutral"), item("neutral")]
    assert compute_evidence_breadth(mixed) == pytest.approx(0.4)
    assert compute_evidence_breadth([item("supports")] * 5) == 1.0
    assert compute_evidence_breadth([item("supports")] * 9) == 1.0


def test_conflict_detected_for_two_credible_semantic_scholar_papers():
    items = [
        item("supports", 0.9, "a", cites=20),
        item("contradicts", 0.8, "b", cites=50),
    ]
    conflicts = detect_conflicts(items)
    assert len(conflicts) == 1
    assert {conflicts[0]["paper_a"], conflicts[0]["paper_b"]} == {"a", "b"}
    assert conflicts[0]["severity"] == "moderate"


def test_conflict_severity_high_when_both_very_confident():
    items = [item("supports", 0.9, "a", cites=30), item("contradicts", 0.9, "b", cites=30)]
    assert detect_conflicts(items)[0]["severity"] == "high"


def test_no_conflict_below_citation_threshold():
    items = [item("supports", 0.9, "a", cites=19), item("contradicts", 0.9, "b", cites=50)]
    assert detect_conflicts(items) == []


def test_no_conflict_at_or_below_confidence_threshold():
    items = [item("supports", 0.7, "a", cites=50), item("contradicts", 0.9, "b", cites=50)]
    assert detect_conflicts(items) == []


def test_no_conflict_when_same_stance():
    items = [item("supports", 0.9, "a", cites=50), item("supports", 0.9, "b", cites=50)]
    assert detect_conflicts(items) == []


def test_arxiv_papers_use_quality_floor_not_citations():
    # arXiv papers have 0 citations, so the quality score decides
    ok = [
        item("supports", 0.9, "a", source="arxiv", score=0.6),
        item("contradicts", 0.9, "b", cites=30, source="semantic_scholar"),
    ]
    assert len(detect_conflicts(ok)) == 1

    both_arxiv = [
        item("supports", 0.9, "a", source="arxiv", score=0.7),
        item("contradicts", 0.9, "b", source="arxiv", score=0.65),
    ]
    assert len(detect_conflicts(both_arxiv)) == 1

    too_low = [
        item("supports", 0.9, "a", source="arxiv", score=0.59),
        item("contradicts", 0.9, "b", cites=30, source="semantic_scholar"),
    ]
    assert detect_conflicts(too_low) == []


def test_conflict_penalty_is_085():
    items = [item("supports", 0.5, "a", score=1.0)]
    scores = compute_directional_scores(items)
    # 0.4*1 (agreement) + 0.3*0.5 (avg confidence) + 0.2*1 (supports) + 0.1*0.2 (breadth)
    assert compute_confidence(scores, items, []) == pytest.approx(0.77)
    assert compute_confidence(scores, items, [{"x": 1}]) == pytest.approx(0.77 * 0.85, abs=1e-4)


def test_confidence_is_capped_at_095():
    items = [item("supports", 1.0, str(n), score=1.0) for n in range(5)]
    scores = compute_directional_scores(items)
    assert compute_confidence(scores, items, []) == 0.95  # the raw formula gives 1.0


def test_empty_input():
    assert compute_confidence({"supports": 0, "contradicts": 0, "neutral": 1}, [], []) == 0.0
    result = calibrate_confidence([])
    assert result["final_decision"] == "neutral"
    assert result["confidence_score"] == 0.0
    assert result["confidence_label"] == "very low"
    assert result["evidence_counts"] == {"supports": 0, "contradicts": 0, "neutral": 0}
    assert result["conflicts"] == []


def test_calibrate_confidence_end_to_end():
    rows = [
        {"stance": "supports", "confidence": 0.9, "paper_id": "a", "citation_count": 50,
         "year": 2020, "score": 0.8, "source": "semantic_scholar"},
        {"stance": "contradicts", "confidence": 0.8, "paper_id": "b", "citation_count": 40,
         "year": 2019, "score": 0.7, "source": "semantic_scholar"},
        {"stance": "neutral", "confidence": 0.7, "paper_id": "c", "citation_count": 0,
         "year": 2021, "score": 0.5, "source": "arxiv"},
    ]
    result = calibrate_confidence(rows)
    assert result["evidence_counts"] == {"supports": 1, "contradicts": 1, "neutral": 1}
    assert result["evidence_breadth"] == pytest.approx(0.4)
    assert "diversity_score" not in result
    assert len(result["conflicts"]) == 1
    assert result["final_decision"] == "supports"
    assert sum(result["evidence_distribution"].values()) == pytest.approx(1.0)
