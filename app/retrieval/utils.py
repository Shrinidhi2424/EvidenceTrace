from typing import List
from app.retrieval.models import Paper


def safe_get(data: dict, key: str, default=None):
    """Safely extract value from dict."""
    return data.get(key, default) if data else default


def normalize_text(text: str) -> str:
    """Basic normalization for deduplication."""
    if not text:
        return ""
    return text.strip().lower()


def deduplicate_papers(papers: List[Paper]) -> List[Paper]:
    """
    Deduplicate papers using (title, year) and paper_id.
    """
    seen_keys = set()
    seen_ids = set()

    unique = []

    for paper in papers:
        key = (normalize_text(paper.title), paper.year)
        id_key = paper.paper_id

        if key in seen_keys or id_key in seen_ids:
            continue

        seen_keys.add(key)
        seen_ids.add(id_key)
        unique.append(paper)

    return unique
