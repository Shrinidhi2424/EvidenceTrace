# EvidenceTrace

Checking a scientific claim against the literature is slow: you search for papers, read abstracts, and try to reconcile studies that disagree. Many AI tools blend everything into one fluent answer and hide the disagreement. EvidenceTrace takes a claim written as a statement (for example "Sleep deprivation impairs working memory."), retrieves papers from Semantic Scholar and arXiv, decides for each paper whether its abstract supports, contradicts or says nothing clear about the claim, and combines that into a verdict with a confidence score. It shows the evidence sentence behind every decision and flags high-credibility papers that disagree.

## How the pipeline works

1. **Retrieve.** `fetch_papers` queries Semantic Scholar and arXiv in parallel, removes duplicates, and caches the result for 10 minutes.
2. **Score quality.** `score_papers` ranks papers by a weighted quality score (formula below) and drops weak ones.
3. **Classify stance.** `classify_stance` runs a local NLI model over the sentences of each top paper's abstract (up to `max_papers`, default 8).
4. **Calibrate.** `calibrate_confidence` combines the stances, model confidences and paper quality into a verdict, a confidence score and a list of conflicts.
5. **Report.** `analyze_claim` returns everything as JSON, including the evidence sentence per paper and timings per step.

No LLM is used anywhere. The only models are two small local ones: `all-MiniLM-L6-v2` (embeddings for retrieval similarity) and an NLI cross-encoder (stance).

## How stance classification works

Stance is a natural language inference (NLI) problem: does the abstract *entail* the claim, *contradict* it, or neither?

- The abstract is split into sentences (on `.`, `!`, `?` followed by whitespace). Sentences under 20 characters are dropped and at most 12 are kept.
- Each sentence is the premise and the claim is the hypothesis. The NLI model (`cross-encoder/nli-deberta-v3-small` by default, override with `NLI_MODEL`) scores all pairs in one batch, and softmax turns the scores into probabilities. The label order is read from the model config, not assumed.
- Take the sentence with the highest entailment probability and the one with the highest contradiction probability. If the larger of the two is at least the threshold (0.5, override with `NLI_THRESHOLD`), the stance is `supports` or `contradicts`, the confidence is that probability, and the evidence sentence is that sentence.
- Otherwise the stance is `neutral`, and the confidence is the mean neutral probability across sentences.
- Missing or empty abstracts are `neutral` with confidence 0.3.

## Formulas

**Quality score** (`app/retrieval/quality_scorer.py`):

```
quality = 0.40 * similarity + 0.25 * citations + 0.20 * recency + 0.15 * source
```

- similarity: cosine similarity of claim and abstract embeddings, mapped from [-1, 1] to [0, 1]
- citations: `log(1 + citations) / log(1 + max citations in the result set)`
- recency: linear from 0 (2015) to 1 (current year); missing or future year gives 0
- source: Semantic Scholar 0.85, arXiv 0.70, anything else 0.50
- papers scoring below 0.30 are dropped

**Confidence** (`app/calibration/confidence.py`):

Each paper's weight is its stance confidence times its quality score. The weights are summed per stance and normalised to give `supports`, `contradicts` and `neutral` shares. Then:

```
confidence = 0.4 * agreement + 0.3 * average stance confidence
           + 0.2 * supports share + 0.1 * evidence breadth
```

- agreement: `|supports share - contradicts share|`
- evidence breadth: `min(1, number of non-neutral papers / 5)`
- if any conflict is detected, confidence is multiplied by 0.85
- the result is capped at 0.95
- labels: at least 0.75 high, at least 0.5 moderate, at least 0.3 low, otherwise very low
- verdict: `supports` if the supports share is larger than the contradicts share, `contradicts` if smaller, otherwise `neutral`

**Conflicts:** a supporting and a contradicting paper, each with stance confidence above 0.7, form a conflict if both are credible. A Semantic Scholar paper is credible with at least 20 citations. An arXiv paper (which never has citation counts) is credible with quality score at least 0.6. A conflict is "high" severity if both confidences are above 0.85.

These weights and thresholds are hand-chosen design values, not fitted to data. The confidence score is a heuristic summary, not a calibrated probability.

## Setup

```bash
python -m venv venv
venv\Scripts\activate        # Windows; use `source venv/bin/activate` elsewhere
pip install -r requirements.txt
```

The first run downloads the two models from Hugging Face. Optionally copy `.env.example` to `.env` and set `SEMANTIC_SCHOLAR_API_KEY` for higher Semantic Scholar rate limits; it works without one.

## Run

Command line:

```bash
python -m app.cli "Sleep deprivation impairs working memory."
```

It prints the verdict and confidence, then one line per paper: stance, probability, title, evidence sentence.

API:

```bash
uvicorn app.main:app
```

- Interactive docs at http://localhost:8000/docs
- `POST /api/analyze` with `{"claim": "...", "max_papers": 8}`. The claim must be 10-300 characters and phrased as a statement (a trailing `?` gives a 422). `max_papers` is 1-15. If neither source returns papers, the response is 502.
- `GET /health` returns the NLI model name and whether it is loaded.

Both models are loaded at startup; set `SKIP_WARMUP=1` to skip that (used by the tests).

## Tests

```bash
pip install -r requirements-dev.txt
pytest                      # offline: no network, no model downloads
pytest -m integration       # runs the real NLI model on two sentence pairs
```

`scripts/smoke_*.py` are manual scripts that print results and use the network (run with `python -m scripts.smoke_retrieval`, etc.).

## Evaluating stance quality

The labels must come from you. Build a CSV, fill in the `gold` column by hand (`supports`, `contradicts` or `neutral`), then run the evaluation:

```bash
python -m scripts.build_eval_set "Sleep deprivation impairs working memory."
python -m scripts.eval_stance
```

`eval_stance` prints accuracy, per-class precision/recall/F1, macro-F1 and a confusion matrix, and writes `data/eval_results.md`. No evaluation results are included in this repository yet.

## Design decisions

- **NLI instead of an LLM.** A local NLI model is deterministic (same input, same output), free to run, runs on a CPU, and its decision is inspectable: you can point to the exact sentence and probability. The previous LLM version (tag `v1-llm`) depended on paid APIs, varied between runs, and needed a keyword hack to repair inconsistent answers.
- **Log-scaled citations.** Citation counts are heavily skewed: a few papers have thousands and most have a handful. A linear scale would make everything but the top paper look near zero; the log scale keeps differences among ordinary papers visible.
- **The 0.95 cap.** The inputs are abstracts only, classified by a general-purpose model, over at most 15 papers. Reporting near-certainty from that would overstate what the system knows, so the score never exceeds 0.95.
- **Sentence-level evidence.** Scoring sentences instead of the whole abstract keeps NLI inputs short (it is trained on sentence pairs) and gives a human-readable reason for each stance.

## Limitations

- NLI models are trained on general text, not scientific writing. They can misread hedged findings, negation, and sentences that state a hypothesis or background ("sleep loss is known to impair memory") as if they were results.
- Only abstracts are used, not full texts, so a paper's real finding may be missing from what the model sees.
- Claims must be statements. Questions are rejected.
- Compound claims ("X improves Y and Z") are not decomposed; the model sees them as one hypothesis.
- Quality weights and confidence weights are hand-chosen and not validated against ground truth.
- Retrieval is keyword search on the claim text against two sources, so relevant papers can be missed, and the result depends on those services being reachable.
- Evaluation is not done yet: the scripts exist, but no labelled data or results are included.
