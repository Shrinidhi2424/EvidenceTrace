import pytest

from app.classification.stance_classifier import classify_stance
from app.retrieval.models import Paper


def _paper(abstract: str) -> Paper:
    return Paper(title="t", abstract=abstract, year=2022, citation_count=0, paper_id="x", source="arxiv")


@pytest.mark.integration
def test_real_model_entailment():
    result = classify_stance(_paper("The cat is sleeping on the sofa."), "The cat is asleep.")
    assert result["stance"] == "supports"


@pytest.mark.integration
def test_real_model_contradiction():
    result = classify_stance(
        _paper("The cat is sleeping on the sofa."), "The cat is running around the garden."
    )
    assert result["stance"] == "contradicts"
