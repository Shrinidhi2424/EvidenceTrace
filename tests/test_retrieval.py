from app.retrieval import fetch_papers

query = "intermittent fasting cognitive performance"

papers = fetch_papers(query)

print(f"\nTotal papers fetched: {len(papers)}\n")

for i, p in enumerate(papers[:5]):
    print(f"--- Paper {i+1} ---")
    print(p.to_dict())
    print()
