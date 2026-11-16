"""
Evaluate classify_stance on rows of data/stance_eval.csv that you have labelled.

Usage (from the repo root):
    python -m scripts.eval_stance

Only rows with a filled `gold` column are used. Prints accuracy, per-class
precision/recall/F1, macro-F1 and a confusion matrix, and writes data/eval_results.md.
"""
import csv
import os
import sys

from app.classification import classify_stance
from app.classification.stance_classifier import get_nli_model_name, get_nli_threshold
from app.retrieval.models import Paper

CSV_PATH = os.path.join("data", "stance_eval.csv")
RESULTS_PATH = os.path.join("data", "eval_results.md")
LABELS = ["supports", "contradicts", "neutral"]
MIN_ROWS_WARNING = 10


def confusion_matrix(gold, pred):
    """matrix[g][p] = number of rows with gold label g predicted as p."""
    matrix = {g: {p: 0 for p in LABELS} for g in LABELS}
    for g, p in zip(gold, pred):
        matrix[g][p] += 1
    return matrix


def per_class_metrics(matrix):
    metrics = {}
    for label in LABELS:
        tp = matrix[label][label]
        fp = sum(matrix[g][label] for g in LABELS) - tp
        fn = sum(matrix[label].values()) - tp
        precision = tp / (tp + fp) if tp + fp else 0.0
        recall = tp / (tp + fn) if tp + fn else 0.0
        f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
        metrics[label] = {"precision": precision, "recall": recall, "f1": f1, "support": tp + fn}
    return metrics


def format_report(n, accuracy, metrics, macro_f1, matrix):
    lines = [
        "# Stance evaluation",
        "",
        f"- Labelled rows: {n}",
        f"- NLI model: {get_nli_model_name()}",
        f"- NLI threshold: {get_nli_threshold()}",
        f"- Accuracy: {accuracy:.3f}",
        f"- Macro-F1: {macro_f1:.3f}",
    ]
    if n < MIN_ROWS_WARNING:
        lines.append(f"- WARNING: only {n} labelled rows; these numbers are not reliable.")
    lines += ["", "| Class | Precision | Recall | F1 | Gold count |", "|---|---|---|---|---|"]
    for label in LABELS:
        m = metrics[label]
        lines.append(f"| {label} | {m['precision']:.3f} | {m['recall']:.3f} | {m['f1']:.3f} | {m['support']} |")
    lines += ["", "Confusion matrix (rows = gold, columns = predicted):", ""]
    lines.append("| gold \\ predicted | " + " | ".join(LABELS) + " |")
    lines.append("|---|" + "---|" * len(LABELS))
    for g in LABELS:
        lines.append(f"| {g} | " + " | ".join(str(matrix[g][p]) for p in LABELS) + " |")
    return "\n".join(lines) + "\n"


def main():
    if not os.path.exists(CSV_PATH):
        sys.exit(f"{CSV_PATH} not found. Run scripts/build_eval_set.py first and label it.")

    with open(CSV_PATH, newline="", encoding="utf-8") as f:
        rows = [r for r in csv.DictReader(f) if r["gold"].strip()]

    bad = [r["gold"] for r in rows if r["gold"].strip().lower() not in LABELS]
    if bad:
        sys.exit(f"Invalid gold labels {sorted(set(bad))}; use one of {LABELS}.")
    if not rows:
        sys.exit("No labelled rows (the 'gold' column is empty). Nothing to evaluate.")
    if len(rows) < MIN_ROWS_WARNING:
        print(f"WARNING: only {len(rows)} labelled rows; the numbers will not be reliable.")

    gold, pred = [], []
    for r in rows:
        paper = Paper(
            title=r["title"], abstract=r["abstract"], year=None,
            citation_count=0, paper_id=r["paper_id"], source="",
        )
        gold.append(r["gold"].strip().lower())
        pred.append(classify_stance(paper, r["claim"])["stance"])

    matrix = confusion_matrix(gold, pred)
    metrics = per_class_metrics(matrix)
    accuracy = sum(g == p for g, p in zip(gold, pred)) / len(gold)
    macro_f1 = sum(m["f1"] for m in metrics.values()) / len(LABELS)

    report = format_report(len(rows), accuracy, metrics, macro_f1, matrix)
    print(report)
    with open(RESULTS_PATH, "w", encoding="utf-8") as f:
        f.write(report)
    print(f"Written to {RESULTS_PATH}")


if __name__ == "__main__":
    main()
