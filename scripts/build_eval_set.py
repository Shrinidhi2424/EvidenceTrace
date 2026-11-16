"""
Build a CSV for hand-labelling stance.

Usage (from the repo root):
    python -m scripts.build_eval_set "Sleep deprivation impairs working memory." "Another claim as a statement."

Writes data/stance_eval.csv with columns: claim, paper_id, title, abstract, gold.
`gold` is left empty: fill it in yourself with supports / contradicts / neutral.
Existing rows (and your labels) are kept; only new (claim, paper_id) pairs are added.
"""
import csv
import os
import sys
from itertools import chain, zip_longest

from app.retrieval import fetch_papers

OUT_PATH = os.path.join("data", "stance_eval.csv")
COLUMNS = ["claim", "paper_id", "title", "abstract", "gold"]
PAPERS_PER_CLAIM = 5


def pick_papers(papers, n):
    """Take papers alternately from each source, so both Semantic Scholar and arXiv appear."""
    with_abstract = [p for p in papers if p.abstract and p.abstract.strip()]
    by_source = {}
    for p in with_abstract:
        by_source.setdefault(p.source, []).append(p)
    interleaved = chain.from_iterable(zip_longest(*by_source.values()))
    return [p for p in interleaved if p is not None][:n]


def main(claims):
    rows = []
    if os.path.exists(OUT_PATH):
        with open(OUT_PATH, newline="", encoding="utf-8") as f:
            rows = list(csv.DictReader(f))
    seen = {(r["claim"], r["paper_id"]) for r in rows}

    added = 0
    for claim in claims:
        for paper in pick_papers(fetch_papers(claim), PAPERS_PER_CLAIM):
            if (claim, paper.paper_id) in seen:
                continue
            rows.append(
                {
                    "claim": claim,
                    "paper_id": paper.paper_id,
                    "title": paper.title,
                    "abstract": " ".join(paper.abstract.split()),
                    "gold": "",
                }
            )
            added += 1

    os.makedirs("data", exist_ok=True)
    with open(OUT_PATH, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=COLUMNS)
        writer.writeheader()
        writer.writerows(rows)

    print(f"Added {added} rows ({len(rows)} total) to {OUT_PATH}. Fill in the 'gold' column.")


if __name__ == "__main__":
    if len(sys.argv) < 2:
        sys.exit(__doc__)
    main(sys.argv[1:])
