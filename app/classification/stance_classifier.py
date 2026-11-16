import logging
import os
import re
from typing import Dict, List, Sequence

import numpy as np
from sentence_transformers import CrossEncoder

from app.retrieval.models import Paper

logger = logging.getLogger(__name__)

DEFAULT_NLI_MODEL = "cross-encoder/nli-deberta-v3-small"

MIN_SENTENCE_CHARS = 20  # shorter sentences are dropped (headings, fragments)
MAX_SENTENCES = 12  # cap on sentences scored per abstract
REASON_MAX_CHARS = 300

_SENTENCE_SPLIT = re.compile(r"(?<=[.!?])\s+")

# Module-level cache: the model is loaded once, on first use (never at import).
_nli_model = None


def get_nli_model_name() -> str:
    return os.getenv("NLI_MODEL", DEFAULT_NLI_MODEL)


def get_nli_threshold() -> float:
    return float(os.getenv("NLI_THRESHOLD", "0.5"))


def is_nli_model_loaded() -> bool:
    return _nli_model is not None


def get_nli_model() -> CrossEncoder:
    """Load the NLI cross-encoder lazily and cache it."""
    global _nli_model
    if _nli_model is None:
        name = get_nli_model_name()
        logger.info("Loading NLI model %s", name)
        _nli_model = CrossEncoder(name)
    return _nli_model


def _safe_fallback() -> Dict:
    return {
        "stance": "neutral",
        "confidence": 0.3,
        "reason": "Abstract missing or insufficient for classification",
    }


def split_sentences(abstract: str) -> List[str]:
    """Split on . ! ? followed by whitespace, drop short pieces, keep the first 12."""
    parts = _SENTENCE_SPLIT.split(abstract.strip())
    parts = [" ".join(p.split()) for p in parts]  # collapse newlines/extra spaces
    kept = [p for p in parts if len(p) >= MIN_SENTENCE_CHARS]
    return kept[:MAX_SENTENCES]


def _softmax(logits: np.ndarray) -> np.ndarray:
    shifted = logits - logits.max(axis=1, keepdims=True)
    exp = np.exp(shifted)
    return exp / exp.sum(axis=1, keepdims=True)


def aggregate_sentence_probs(
    sentences: Sequence[str],
    probs: np.ndarray,
    label_names: Sequence[str],
    threshold: float,
) -> Dict:
    """
    Turn per-sentence NLI probabilities into one stance for the whole abstract.

    probs has shape (n_sentences, n_labels); label_names[i] names column i
    (taken from the model config, so the column order is never assumed).
    """
    probs = np.asarray(probs, dtype=float)
    names = [n.lower() for n in label_names]
    ent_col = names.index("entailment")
    con_col = names.index("contradiction")
    neu_col = names.index("neutral")

    best_ent = int(np.argmax(probs[:, ent_col]))
    best_con = int(np.argmax(probs[:, con_col]))
    ent_p = float(probs[best_ent, ent_col])
    con_p = float(probs[best_con, con_col])

    if max(ent_p, con_p) >= threshold:
        if ent_p >= con_p:
            stance, conf, idx = "supports", ent_p, best_ent
        else:
            stance, conf, idx = "contradicts", con_p, best_con
        return {
            "stance": stance,
            "confidence": conf,
            "reason": sentences[idx][:REASON_MAX_CHARS],
        }

    return {
        "stance": "neutral",
        "confidence": float(probs[:, neu_col].mean()),
        "reason": "No sentence clearly supports or contradicts the claim.",
    }


def classify_stance(paper: Paper, claim: str) -> Dict:
    """
    Classify the stance of a paper's abstract relative to a claim.

    Returns {"stance", "confidence", "reason"}; reason is the evidence sentence.
    """
    if not paper or not paper.abstract or not paper.abstract.strip():
        return _safe_fallback()

    sentences = split_sentences(paper.abstract)
    if not sentences:
        return _safe_fallback()

    model = get_nli_model()
    # premise = abstract sentence, hypothesis = claim; one batch for all sentences
    logits = model.predict([(s, claim) for s in sentences])
    probs = _softmax(np.asarray(logits, dtype=float))

    id2label = model.config.id2label
    label_names = [id2label[i] for i in range(len(id2label))]

    result = aggregate_sentence_probs(sentences, probs, label_names, get_nli_threshold())
    logger.info(
        "Paper %s -> %s (confidence=%.2f)",
        paper.paper_id,
        result["stance"],
        result["confidence"],
    )
    return result
