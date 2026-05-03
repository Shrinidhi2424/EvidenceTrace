from app.retrieval import fetch_papers
from app.retrieval.quality_scorer import score_papers
from app.classification import classify_stance
from app.calibration import calibrate_confidence


def test_full_pipeline():
    query = "does sleep deprivation impair cognitive performance"

    print(f"\n=== QUERY ===\n{query}\n")

    papers = fetch_papers(query)
    print(f"Retrieved papers: {len(papers)}")

    if not papers:
        print("No papers retrieved. Exiting.")
        return

    scored = score_papers(papers, query)
    print(f"Scored papers: {len(scored)}")

    if not scored:
        print("No papers after scoring. Exiting.")
        return

    evidence = []

    print("\n=== TOP PAPERS (CLASSIFICATION) ===\n")

    for i, p in enumerate(scored[:3]):
        stance_result = classify_stance(p, query)

        print(f"{i+1}. {p.title}")
        print(f"   Score: {p.score}")
        print(f"   Stance: {stance_result['stance']}")
        print(f"   Confidence: {stance_result['confidence']}")
        print(f"   Reason: {stance_result['reason']}\n")

        evidence.append(
            {
                "stance": stance_result["stance"],
                "confidence": stance_result["confidence"],
                "paper_id": p.paper_id,
                "citation_count": p.citation_count,
                "year": p.year,
                "score": p.score,
            }
        )

    result = calibrate_confidence(evidence)

    print("\n=== FINAL CALIBRATED OUTPUT ===\n")
    for k, v in result.items():
        print(f"{k}: {v}")


if __name__ == "__main__":
    test_full_pipeline()
