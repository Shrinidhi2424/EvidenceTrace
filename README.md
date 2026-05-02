# 🧠 EvidenceTrace: Agentic Scientific Claim Analyzer

> A multi-agent AI system that analyzes scientific claims by retrieving research papers, detecting evidence stance (support / contradict / neutral), and producing **confidence-calibrated evidence reports**.

---

## 🚨 Problem

Literature review is slow and error-prone.

Given a claim like:

> *"Does intermittent fasting improve cognitive performance?"*

A researcher must:

* search papers manually
* read multiple abstracts
* reconcile conflicting results

Most AI systems (RAG) fail here by:

* blending evidence into a single answer
* hiding disagreements
* producing misleading “consensus”

---

## 💡 Solution

**EvidenceTrace** solves this by:

* Decomposing claims into sub-questions
* Retrieving papers from real sources
* Classifying each piece of evidence as:

  * ✅ supports
  * ❌ contradicts
  * ⚖️ neutral
* Aggregating results using a **confidence calibration layer**
* Producing structured, uncertainty-aware reports

---

## 🔥 Key Features

### 🧩 Multi-Agent Pipeline

* Claim decomposition using LLM
* Retrieval from arXiv + Semantic Scholar
* Stance classification per evidence chunk
* Confidence calibration (no LLM, pure scoring)

---

### ⚖️ Stance-Aware Reasoning

Instead of naive summarization:

* Detects **agreement vs contradiction**
* Surfaces **conflicting evidence explicitly**

---

### 📊 Confidence Calibration (Core Innovation)

Computes a **final confidence score** using:

* agreement between sources
* model confidence
* evidence quality
* source diversity

👉 Moves beyond classification → introduces **probabilistic reasoning**

---

### 📉 Retrieval Quality Scoring

Each chunk is scored using:

* semantic similarity
* citation count (log-scaled)
* recency
* source reliability

👉 Ensures only high-quality evidence is used downstream

---

### ⚠️ Conflict Detection

* Identifies high-confidence contradictions
* Flags them in final output
* Reduces confidence accordingly

---

## 🏗️ System Architecture

```
User Claim
   ↓
Decomposition Agent
   ↓
Retrieval Engine (arXiv + Semantic Scholar)
   ↓
Quality Scorer
   ↓
Stance Classifier
   ↓
Confidence Calibration Layer
   ↓
Synthesis Agent
   ↓
Structured Report (UI)
```

---

## 🧪 Example Output

```json
{
  "claim": "Does intermittent fasting improve cognitive performance?",
  "confidence_score": 0.43,
  "confidence_label": "Low — conflicting evidence",
  "support_score": 0.54,
  "contradiction_score": 0.34,
  "conflicts": ["High-confidence disagreement between 2 papers"],
  "summary": "Evidence is mixed. Multiple high-citation papers disagree."
}
```

---

## 🛠️ Tech Stack

| Layer           | Technology                       |
| --------------- | -------------------------------- |
| Backend         | FastAPI                          |
| Agent Framework | LangGraph                        |
| LLM (Primary)   | Gemini 2.5 Flash                 |
| LLM (Fallback)  | Groq (LLaMA / DeepSeek)          |
| Vector DB       | ChromaDB                         |
| Embeddings      | Sentence Transformers            |
| Data Sources    | arXiv API + Semantic Scholar API |
| Frontend        | Streamlit                        |

---

## 🔁 Multi-LLM Design

```python
if Gemini available:
    use Gemini
else:
    fallback to Groq
```

👉 Ensures:

* reliability
* cost efficiency
* fault tolerance

---

## ⚙️ Setup Instructions

### 1. Clone repo

```bash
git clone https://github.com/yourusername/evidence-trace.git
cd evidence-trace
```

---

### 2. Create virtual environment

```bash
python -m venv venv
venv\Scripts\activate   # Windows
```

---

### 3. Install dependencies

```bash
pip install -r requirements.txt
```

---

### 4. Setup environment variables

Create `.env`:

```
GEMINI_API_KEY=
GROQ_API_KEY=
SEMANTIC_SCHOLAR_API_KEY=
```

---

### 5. Run backend

```bash
uvicorn app.main:app --reload
```

---

### 6. Run frontend

```bash
streamlit run frontend/app.py
```

---

## 📊 Evaluation

The system is evaluated using:

* **Stance Classification Accuracy (F1 score)**
* **Retrieval Ranking (Spearman correlation)**
* **Confidence calibration sanity tests**
* **Baseline comparison vs naive RAG**

---

## 🧠 What This Project Demonstrates

* Advanced **RAG system design**
* **Multi-agent orchestration (LangGraph)**
* Handling **uncertainty and conflicting evidence**
* **Quantitative evaluation of AI systems**
* Real-world API integration + system design thinking

---

## ⚠️ Limitations

* Dependent on external APIs
* LLM-based stance classification may introduce bias
* Citation count ≠ ground truth

---

## 🚀 Future Improvements

* Temporal evidence analysis
* Citation graph visualization
* Improved stance classification models
* Better domain-specific tuning

---

## 🧑‍💻 Author

**Shrinidhi Shetty**

---

## ⭐ Why This Project Matters

Most AI systems answer questions.

**EvidenceTrace evaluates evidence.**

That difference is what makes it closer to real-world decision systems.
