import logging
import math
from datetime import datetime
from typing import List

import numpy as np
from sentence_transformers import SentenceTransformer

from app.retrieval.models import Paper

logger = logging.getLogger(__name__)

# Load embedding model once (cached)
_model = SentenceTransformer("all-MiniLM-L6-v2")


def compute_similarity(
    query_emb: np.ndarray, abstract_emb: np.ndarray, paper_id: str
) -> float:
    """
    Compute cosine similarity between normalized embeddings.
    Returns value in [0, 1].
    """
    try:
        similarity = float(
            np.dot(query_emb, abstract_emb)
        )  # already normalized embeddings

        # normalize from [-1,1] → [0,1] with clamping
        sim = (similarity + 1) / 2
        return max(0.0, min(1.0, sim))

    except Exception as e:
        logger.error("Similarity computation error for paper %s: %s", paper_id, e)
        return 0.0


def normalize_citations(citation_count: int, max_citations: int) -> float:
    """
    Log-scaled citation normalization.
    """
    if citation_count <= 0 or max_citations <= 0:
        return 0.0

    try:
        return math.log1p(citation_count) / math.log1p(max_citations)
    except Exception as e:
        logger.error("Citation normalization error: %s", e)
        return 0.0


def compute_recency(year: int, base_year: int = 2015) -> float:
    """
    Compute recency score in [0,1].
    """
    current_year = datetime.now().year

    if not year or year > current_year:
        return 0.0

    window = current_year - base_year

    if window <= 0:
        return 0.0

    age = current_year - year
    rec = 1 - (age / window)

    return max(0.0, min(1.0, rec))


def compute_source_score(source: str) -> float:
    """
    Source reliability mapping.
    """
    if source == "semantic_scholar":
        return 0.85
    elif source == "arxiv":
        return 0.70
    return 0.50


def compute_final_score(sim: float, cit: float, rec: float, src: float) -> float:
    """
    Weighted final score.
    """
    score = 0.40 * sim + 0.25 * cit + 0.20 * rec + 0.15 * src
    return round(score, 4)


def score_papers(
    papers: List[Paper], query: str, threshold: float = 0.30, top_k: int = 20
) -> List[Paper]:
    """
    Score and rank papers based on quality.
    """
    if not papers:
        return []

    logger.info("Scoring %d papers", len(papers))

    # Precompute query embedding (normalized)
    try:
        query_emb = _model.encode(
            query, convert_to_numpy=True, normalize_embeddings=True
        )
    except Exception as e:
        logger.error("Query embedding failed: %s", e)
        return []

    # Prepare abstracts (truncate safely)
    abstracts = [
        (
            p.abstract[:1000]
            if p.abstract and len(p.abstract) > 1000
            else (p.abstract or "")
        )
        for p in papers
    ]

    # Batch encode abstracts (normalized)
    try:
        abstract_embeddings = _model.encode(
            abstracts, convert_to_numpy=True, normalize_embeddings=True
        )
    except Exception as e:
        logger.error("Abstract embedding batch failed: %s", e)
        # fallback: zero embeddings
        abstract_embeddings = [np.zeros_like(query_emb) for _ in papers]

    max_citations = max((p.citation_count or 0) for p in papers)
    logger.info("Max citations: %d", max_citations)

    scored_papers = []

    for paper, abstract_emb in zip(papers, abstract_embeddings):
        sim = compute_similarity(query_emb, abstract_emb, paper.paper_id)
        cit = normalize_citations(paper.citation_count or 0, max_citations)
        rec = compute_recency(paper.year)
        src = compute_source_score(paper.source)

        final_score = compute_final_score(sim, cit, rec, src)

        # attach score (assumes Paper has score field)
        paper.score = final_score

        # attach score breakdown (transparency)
        paper.score_components = {"sim": sim, "cit": cit, "rec": rec, "src": src}

        if final_score >= threshold:
            scored_papers.append(paper)

    logger.info("Papers after filtering: %d", len(scored_papers))

    scored_papers.sort(key=lambda p: p.score or 0, reverse=True)

    return scored_papers[:top_k]
