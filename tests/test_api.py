import pytest
from fastapi.testclient import TestClient

from app.pipeline import NoPapersFound

FIXED_REPORT = {
    "claim": "Sleep deprivation impairs working memory.",
    "verdict": "supports",
    "confidence_score": 0.74,
    "confidence_label": "moderate",
    "evidence_distribution": {"supports": 0.8, "contradicts": 0.1, "neutral": 0.1},
    "evidence_counts": {"supports": 4, "contradicts": 1, "neutral": 3},
    "conflicts": [],
    "papers": [
        {
            "title": "A paper",
            "year": 2022,
            "source": "semantic_scholar",
            "citation_count": 12,
            "paper_id": "abc",
            "quality_score": 0.7,
            "score_components": {"sim": 0.8, "cit": 0.5, "rec": 0.9, "src": 0.85},
            "stance": "supports",
            "stance_confidence": 0.95,
            "evidence_sentence": "Sleep deprivation impaired working memory.",
        }
    ],
    "timings_ms": {"retrieval": 1.0, "scoring": 1.0, "classification": 1.0, "calibration": 1.0},
}

VALID = {"claim": "Sleep deprivation impairs working memory."}


@pytest.fixture
def client(monkeypatch):
    monkeypatch.setenv("SKIP_WARMUP", "1")
    monkeypatch.setattr("app.api.routes.analyze_claim", lambda claim, max_papers=8: FIXED_REPORT)
    from app.main import app

    with TestClient(app) as c:
        yield c


def test_valid_claim_returns_200(client):
    response = client.post("/api/analyze", json=VALID)
    assert response.status_code == 200
    body = response.json()
    assert body["verdict"] == "supports"
    assert body["papers"][0]["paper_id"] == "abc"


def test_claim_ending_in_question_mark_is_422(client):
    response = client.post("/api/analyze", json={"claim": "Does sleep deprivation impair memory?"})
    assert response.status_code == 422
    assert "Phrase the claim as a statement" in response.text


def test_too_short_claim_is_422(client):
    assert client.post("/api/analyze", json={"claim": "short"}).status_code == 422


def test_max_papers_out_of_range_is_422(client):
    assert client.post("/api/analyze", json={**VALID, "max_papers": 0}).status_code == 422
    assert client.post("/api/analyze", json={**VALID, "max_papers": 16}).status_code == 422


def test_no_papers_found_is_502(client, monkeypatch):
    def raise_none(claim, max_papers=8):
        raise NoPapersFound("no papers")

    monkeypatch.setattr("app.api.routes.analyze_claim", raise_none)
    assert client.post("/api/analyze", json=VALID).status_code == 502


def test_health(client):
    response = client.get("/health")
    assert response.status_code == 200
    body = response.json()
    assert body["status"] == "ok"
    assert "nli_model" in body and isinstance(body["loaded"], bool)
