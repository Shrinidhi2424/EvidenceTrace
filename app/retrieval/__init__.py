from typing import List
from app.retrieval.models import Paper
from app.retrieval.semantic_scholar_client import fetch_semantic_scholar
from app.retrieval.arxiv_client import fetch_arxiv
from app.retrieval.utils import deduplicate_papers


def fetch_papers(query: str, limit: int = 20) -> List[Paper]:
    """
    Unified retrieval function with limit control.
    """

    semantic_papers = fetch_semantic_scholar(query)
    arxiv_papers = fetch_arxiv(query)

    combined = semantic_papers + arxiv_papers

    clean = deduplicate_papers(combined)

    return clean[:limit]
