import copy
import logging
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from typing import Dict, List, Tuple

from app.retrieval.arxiv_client import fetch_arxiv
from app.retrieval.models import Paper
from app.retrieval.semantic_scholar_client import fetch_semantic_scholar
from app.retrieval.utils import deduplicate_papers

logger = logging.getLogger(__name__)

CACHE_TTL_SECONDS = 600  # 10 minutes

# (query key, limit) -> (time stored, papers)
_cache: Dict[Tuple[str, int], Tuple[float, List[Paper]]] = {}
_cache_lock = threading.Lock()


def _cache_key(query: str, limit: int) -> Tuple[str, int]:
    """Normalise the query so 'Sleep  loss' and 'sleep loss' share an entry."""
    return (" ".join(query.lower().split()), limit)


def _safe_fetch(fetcher, query: str) -> List[Paper]:
    """Run one source; a failing source must not take the other one down."""
    try:
        return fetcher(query)
    except Exception as e:
        logger.error("%s failed: %s", fetcher.__name__, e)
        return []


def fetch_papers(query: str, limit: int = 20) -> List[Paper]:
    """
    Unified retrieval: Semantic Scholar and arXiv in parallel, deduplicated.

    Results are cached in memory for 10 minutes. Callers get deep copies, so
    scoring (which sets paper.score) can never change a cached object.
    """
    key = _cache_key(query, limit)

    with _cache_lock:
        entry = _cache.get(key)
        if entry and time.time() - entry[0] < CACHE_TTL_SECONDS:
            logger.info("Retrieval cache hit for %r", key[0])
            return copy.deepcopy(entry[1])

    with ThreadPoolExecutor(max_workers=2) as pool:
        semantic_future = pool.submit(_safe_fetch, fetch_semantic_scholar, query)
        arxiv_future = pool.submit(_safe_fetch, fetch_arxiv, query)
        semantic_papers = semantic_future.result()
        arxiv_papers = arxiv_future.result()

    clean = deduplicate_papers(semantic_papers + arxiv_papers)[:limit]

    # Do not cache empty results: they usually mean a transient API failure.
    if clean:
        with _cache_lock:
            _cache[key] = (time.time(), copy.deepcopy(clean))

    return clean
