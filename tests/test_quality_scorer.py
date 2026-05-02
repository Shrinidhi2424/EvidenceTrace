from app.retrieval import fetch_papers
from app.retrieval.quality_scorer import score_papers

query = "intermittent fasting cognitive performance"

papers = fetch_papers(query)
scored = score_papers(papers, query)

print("\nTop 5 scored papers:\n")

for p in scored[:5]:
    print("=" * 60)
    print("Title:", p.title)
    print("Score:", p.score)
    print("Components:", p.score_components)
