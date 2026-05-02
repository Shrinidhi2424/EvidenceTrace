from app.retrieval import fetch_papers
from app.retrieval.quality_scorer import score_papers
from app.classification import classify_stance


def test_real_pipeline():
    query = "intermittent fasting cognitive performance"

    papers = fetch_papers(query)
    scored = score_papers(papers, query)

    print("\n=== TOP 3 PAPERS ===\n")

    for p in scored[:3]:
        result = classify_stance(p, query)

        print("=" * 60)
        print("Title:", p.title)
        print("Score:", p.score)
        print("Stance:", result["stance"])
        print("Confidence:", result["confidence"])
        print("Reason:", result["reason"])


if __name__ == "__main__":
    test_real_pipeline()
