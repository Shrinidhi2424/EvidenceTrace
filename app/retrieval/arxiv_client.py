import requests
import time
import logging
import xml.etree.ElementTree as ET
from typing import List
from app.retrieval.models import Paper

logger = logging.getLogger(__name__)

BASE_URL = "http://export.arxiv.org/api/query"


def fetch_arxiv(query: str, max_results: int = 10) -> List[Paper]:
    """
    Fetch papers from arXiv API with retry logic.
    """
    params = {
        "search_query": f"all:{query}",
        "start": 0,
        "max_results": max_results,
    }

    for _ in range(3):
        try:
            response = requests.get(BASE_URL, params=params, timeout=10)
            response.raise_for_status()
            break
        except Exception as e:
            logger.error("arXiv API error: %s", e)
            time.sleep(1)
    else:
        return []

    root = ET.fromstring(response.text)
    namespace = {"atom": "http://www.w3.org/2005/Atom"}

    papers = []

    for entry in root.findall("atom:entry", namespace):
        title = entry.find("atom:title", namespace)
        summary = entry.find("atom:summary", namespace)
        published = entry.find("atom:published", namespace)
        paper_id = entry.find("atom:id", namespace)

        try:
            year = int(published.text[:4]) if published is not None else None
        except Exception:
            year = None

        paper = Paper(
            title=title.text.strip() if title is not None else "",
            abstract=summary.text.strip() if summary is not None else "",
            year=year,
            citation_count=0,
            paper_id=paper_id.text if paper_id is not None else "",
            source="arxiv",
        )

        papers.append(paper)

    return papers
