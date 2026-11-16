import numpy as np
import pytest

from app.classification.stance_classifier import (
    MAX_SENTENCES,
    aggregate_sentence_probs,
    classify_stance,
    split_sentences,
)
from app.retrieval.models import Paper

LABELS = ["contradiction", "entailment", "neutral"]  # same order as the real model
SENTENCES = ["First sentence of the abstract.", "Second sentence of the abstract."]


def test_entailment_wins():
    probs = np.array([[0.10, 0.80, 0.10], [0.05, 0.10, 0.85]])
    result = aggregate_sentence_probs(SENTENCES, probs, LABELS, 0.5)
    assert result["stance"] == "supports"
    assert result["confidence"] == pytest.approx(0.80)
    assert result["reason"] == SENTENCES[0]


def test_contradiction_wins():
    probs = np.array([[0.10, 0.20, 0.70], [0.90, 0.05, 0.05]])
    result = aggregate_sentence_probs(SENTENCES, probs, LABELS, 0.5)
    assert result["stance"] == "contradicts"
    assert result["confidence"] == pytest.approx(0.90)
    assert result["reason"] == SENTENCES[1]


def test_larger_of_the_two_wins_when_both_pass_threshold():
    probs = np.array([[0.10, 0.60, 0.30], [0.75, 0.05, 0.20]])
    assert aggregate_sentence_probs(SENTENCES, probs, LABELS, 0.5)["stance"] == "contradicts"


def test_below_threshold_is_neutral_with_mean_neutral_probability():
    probs = np.array([[0.20, 0.30, 0.50], [0.10, 0.20, 0.70]])
    result = aggregate_sentence_probs(SENTENCES, probs, LABELS, 0.5)
    assert result["stance"] == "neutral"
    assert result["confidence"] == pytest.approx(0.60)
    assert result["reason"] == "No sentence clearly supports or contradicts the claim."


def test_threshold_is_inclusive():
    probs = np.array([[0.10, 0.50, 0.40]])
    assert aggregate_sentence_probs(["A sentence long enough."], probs, LABELS, 0.5)["stance"] == "supports"


def test_label_order_comes_from_label_names():
    # same situation as test_entailment_wins, with columns shuffled together with their names
    shuffled = ["neutral", "contradiction", "entailment"]
    probs = np.array([[0.10, 0.10, 0.80], [0.85, 0.05, 0.10]])
    result = aggregate_sentence_probs(SENTENCES, probs, shuffled, 0.5)
    assert result["stance"] == "supports"
    assert result["confidence"] == pytest.approx(0.80)
    assert result["reason"] == SENTENCES[0]

    # and a contradiction case in yet another order
    other = ["entailment", "neutral", "contradiction"]
    probs = np.array([[0.10, 0.20, 0.70], [0.05, 0.05, 0.90]])
    result = aggregate_sentence_probs(SENTENCES, probs, other, 0.5)
    assert result["stance"] == "contradicts"
    assert result["reason"] == SENTENCES[1]


def test_reason_is_trimmed_to_300_characters():
    probs = np.array([[0.05, 0.90, 0.05]])
    assert len(aggregate_sentence_probs(["x" * 500], probs, LABELS, 0.5)["reason"]) == 300


def test_split_sentences_drops_short_and_caps_at_twelve():
    text = "Too short. " + " ".join(f"This is a long enough sentence number {i}." for i in range(20))
    sentences = split_sentences(text)
    assert "Too short." not in sentences
    assert len(sentences) == MAX_SENTENCES
    assert sentences[0] == "This is a long enough sentence number 0."


def test_split_sentences_collapses_newlines():
    assert split_sentences("BACKGROUND\nSleep loss harms memory.  Ok!") == [
        "BACKGROUND Sleep loss harms memory."
    ]


@pytest.mark.parametrize("abstract", ["", "   ", None])
def test_missing_abstract_falls_back_without_loading_a_model(abstract):
    paper = Paper(title="t", abstract=abstract, year=2020, citation_count=0, paper_id="x", source="arxiv")
    result = classify_stance(paper, "Sleep deprivation impairs working memory.")
    assert result["stance"] == "neutral"
    assert result["confidence"] == 0.3
