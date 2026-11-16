import argparse
import logging
import sys

from pydantic import ValidationError

from app.models.schemas import AnalyzeRequest
from app.pipeline import NoPapersFound, analyze_claim


def main() -> int:
    parser = argparse.ArgumentParser(description="Analyze a scientific claim against the literature.")
    parser.add_argument("claim", help="The claim, phrased as a statement.")
    parser.add_argument("--max-papers", type=int, default=8, help="Papers to classify (1-15).")
    args = parser.parse_args()

    logging.basicConfig(level=logging.WARNING)

    try:
        request = AnalyzeRequest(claim=args.claim, max_papers=args.max_papers)
    except ValidationError as e:
        print(e.errors()[0]["msg"], file=sys.stderr)
        return 2

    try:
        report = analyze_claim(request.claim, request.max_papers)
    except NoPapersFound as e:
        print(f"Error: {e}", file=sys.stderr)
        return 1

    print(f"Claim:   {report['claim']}")
    print(f"Verdict: {report['verdict']}  (confidence {report['confidence_score']} - {report['confidence_label']})")
    if report["conflicts"]:
        print(f"Conflicts: {len(report['conflicts'])}")
    print()
    for p in report["papers"]:
        print(f"{p['stance']:<11} {p['stance_confidence']:.2f}  {p['title']} | {p['evidence_sentence']}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
