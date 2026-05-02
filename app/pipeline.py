from app.agents.decomposition import decompose_claim
from app.retrieval.semantic_scholar_client import fetch_papers

def run_pipeline(claim: str):
    sub_questions = decompose_claim(claim)

    papers = []
    for q in sub_questions:
        papers.extend(fetch_papers(q))

    return {
        "claim": claim,
        "sub_questions": sub_questions,
        "papers_found": len(papers)
    }