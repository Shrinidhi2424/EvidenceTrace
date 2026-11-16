import math
from datetime import datetime

import pytest

from app.retrieval.quality_scorer import (
    compute_final_score,
    compute_recency,
    compute_source_score,
    normalize_citations,
)

NOW = datetime.now().year


def test_normalize_citations_zero_is_zero():
    assert normalize_citations(0, 100) == 0.0
    assert normalize_citations(5, 0) == 0.0


def test_normalize_citations_max_is_one():
    assert normalize_citations(100, 100) == pytest.approx(1.0)


def test_normalize_citations_is_log_scaled():
    # 10 of 100 citations is 10% linearly, but log-scaled it is much higher
    expected = math.log1p(10) / math.log1p(100)
    assert normalize_citations(10, 100) == pytest.approx(expected)
    assert normalize_citations(10, 100) > 0.10
    assert normalize_citations(10, 100) < normalize_citations(50, 100)


def test_recency_current_year_is_one():
    assert compute_recency(NOW) == pytest.approx(1.0)


def test_recency_base_year_is_zero():
    assert compute_recency(2015) == pytest.approx(0.0)
    assert compute_recency(2000) == 0.0  # older than the base year is clamped to 0


def test_recency_future_or_missing_year_is_zero():
    assert compute_recency(NOW + 1) == 0.0
    assert compute_recency(None) == 0.0


def test_recency_is_linear_between_base_and_now():
    assert compute_recency(NOW - 1, base_year=NOW - 2) == pytest.approx(0.5)


def test_source_scores():
    assert compute_source_score("semantic_scholar") == 0.85
    assert compute_source_score("arxiv") == 0.70
    assert compute_source_score("something_else") == 0.50


def test_final_score_weights():
    assert compute_final_score(1, 0, 0, 0) == 0.40
    assert compute_final_score(0, 1, 0, 0) == 0.25
    assert compute_final_score(0, 0, 1, 0) == 0.20
    assert compute_final_score(0, 0, 0, 1) == 0.15
    assert compute_final_score(1, 1, 1, 1) == pytest.approx(1.0)
    assert compute_final_score(0.5, 0.5, 0.5, 0.5) == pytest.approx(0.5)
