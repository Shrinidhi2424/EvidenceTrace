import requests
import time
import logging
from typing import List
from app.retrieval.models import Paper
from app.retrieval.utils import safe_get
from app.core.config import settings

logger = logging.getLogger(__name__)

BASE_URL = "https://api.semanticscholar.org/graph/v1/paper/search"


def fetch_semantic_scholar(query: str, limit: int = 10) -> List[Paper]:
    """
    Fetch papers from Semantic Scholar API with retry logic.
    """
    params = {
        "query": query,
        "limit": limit,
        "fields": "title,abstract,year,citationCount,paperId",
    }

    headers = (
        {"x-api-key": settings.SEMANTIC_SCHOLAR_API_KEY}
        if settings.SEMANTIC_SCHOLAR_API_KEY
        else {}
    )

    for _ in range(3):
        try:
            response = requests.get(
                BASE_URL, params=params, headers=headers, timeout=10
            )
            response.raise_for_status()
            data = response.json()
            break
        except Exception as e:
            logger.error("Semantic Scholar API error: %s", e)
            time.sleep(1)
    else:
        return []

    papers = []

    for item in data.get("data", []):
        abstract = safe_get(item, "abstract", "")

        if not abstract:
            continue

        paper = Paper(
            title=safe_get(item, "title", ""),
            abstract=abstract,
            year=safe_get(item, "year"),
            citation_count=safe_get(item, "citationCount", 0),
            paper_id=safe_get(item, "paperId", ""),
            source="semantic_scholar",
        )

        papers.append(paper)

    return papers
