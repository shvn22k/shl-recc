<div align="center">

# SHL Assessment Recommender

**A conversational AI agent that recommends SHL assessments for any hiring role.**

[![Python](https://img.shields.io/badge/python-3.11-blue.svg)](https://python.org)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.111-009688.svg)](https://fastapi.tiangolo.com)
[![License](https://img.shields.io/badge/license-MIT-green.svg)](LICENSE)
[![Tests](https://img.shields.io/badge/tests-45%2F45%20passing-brightgreen.svg)]()

Built by [Shiven Shandil](https://github.com/shvn22k) as a hiring assessment for SHL Labs.

</div>

---

## Live API

**Endpoint:** `https://shl-recc.onrender.com`

| Route | Method | Description |
|---|---|---|
| `/health` | GET | Readiness probe — returns `{"status":"ok"}` |
| `/chat` | POST | Conversational assessment recommendation |

---

## What It Does

Give it a hiring need in plain English. It asks the right questions, then recommends the exact SHL assessments from the product catalog — grounded, specific, and explainable.

```
Hiring Manager: "I'm hiring a senior Java backend engineer."

Agent: "For a senior IC backend role, here's a focused battery:
        Core Java (Advanced), Spring, SQL, AWS, Verify G+, OPQ32r.
        Say the word if you'd prefer to drop the OPQ32r."
```

The agent handles multi-turn refinement ("add Docker, drop REST"), comparison questions ("what's the difference between DSI and the Safety 8.0?"), legal refusals, and graceful catalog gap handling ("no Rust-specific test exists — here are the closest proxies").

---

## Architecture

The system implements **Conversational Slot-Guided RAG (CSG-RAG)** — a custom retrieval approach designed for explicit-intent, small-catalog recommendation.

```
User message (full conversation history)
             │
             ▼
    ┌─────────────────┐
    │   Guardrails    │──── injection / legal / off-topic / gibberish
    └────────┬────────┘
             │
             ▼
    ┌─────────────────┐
    │ Slot Extractor  │──── LLM Call 1 (Gemini 2.5 Flash, JSON mode)
    │                 │     role · seniority · purpose · constraints
    │                 │     explicit adds/drops · conversation phase
    └────────┬────────┘
             │
      ┌──────┴──────┐
      ▼             ▼
  clarify?     retrieve?
  ask one      │
  question     ▼
             ┌─────────────────────────────────────┐
             │         Hybrid Retrieval            │
             │  1. FAISS semantic search  (top-40) │
             │  2. Job level filter (grad/entry)   │
             │  3. BM25 keyword re-rank            │
             │  4. Explicit addition injection     │
             └──────────────┬──────────────────────┘
                            │ top-20 candidates
                            ▼
             ┌─────────────────────────────────────┐
             │          LLM Ranker                 │──── LLM Call 2
             │  Chain-of-thought selection 1-10    │     (GPT-4o-mini)
             │  OPQ32r + Verify G+ defaults        │
             │  Explicit drop enforcement          │
             └──────────────┬──────────────────────┘
                            │
                            ▼
             ┌─────────────────────────────────────┐
             │         Post-Processing             │
             │  Semantic deduplication             │
             │  URL whitelist enforcement          │
             └──────────────┬──────────────────────┘
                            │
                            ▼
                      ChatResponse
             { reply, recommendations[], end_of_conversation }
```

### Why CSG-RAG instead of standard RAG?

Standard RAG pipelines embed the query and retrieve top-k documents. This works poorly for our problem because:

- The hiring manager's query often contains implicit constraints that change *which type* of assessment to retrieve, not just *which specific* one
- The SHL catalog has sparse metadata (no job families, inconsistent job levels on technical tests)
- The agent must decide whether to clarify, recommend, compare, or close — standard RAG has no notion of conversation phases

CSG-RAG separates **intent understanding** (LLM 1, deterministic extraction) from **assessment selection** (LLM 2, reasoning-based ranking) with a structured `SlotState` intermediate that drives both retrieval filtering and ranker prompting.

### Retrieval Strategy

```
Query text construction:
  Tech roles  → [tech keywords] + role + "knowledge skills technical assessment"
  Other roles → role + seniority + purpose + domain

                     ┌──────────────────┐
  Query text  ──────►│  MiniLM-L6-v2   │──► 384-dim normalized vector
                     └──────────────────┘
                              │
                              ▼
                     ┌──────────────────┐
                     │  FAISS FlatIP    │──► top-40 by cosine similarity
                     │  (377 vectors)   │
                     └──────────────────┘
                              │
                    job level filter (grad/entry only)
                              │
                              ▼
                     ┌──────────────────┐
  Query + role ─────►│   BM25 Okapi    │──► re-ranked candidates
  keywords           └──────────────────┘
                              │
              combined score = 0.6 × semantic + 0.4 × BM25
                              │
                    explicit_additions injected (score=1.0)
                              │
                              ▼
                         top-20 → LLM ranker
```

**Why restrict job level filter to graduate/entry-level?**

Technical knowledge tests (Java, Spring, SQL, AWS etc.) have absent or inconsistent job level metadata in SHL's catalog. Applying the filter for senior+ roles silently eliminates the most relevant candidates. The ranker handles seniority framing through its prompt instead.

**Why lead with tech keywords for tech roles?**

"Senior Rust engineer for high-performance networking" embeds closest to management/leadership assessments when the query leads with seniority framing — because those dominate the senior-IC embedding space. Leading with domain keywords ("rust linux networking systems programming") anchors the query in the technical assessment cluster.

---

## API Reference

### `GET /health`

Readiness probe. Returns immediately once the catalog is loaded.

```json
{"status": "ok"}
```

### `POST /chat`

Stateless conversational endpoint. Send the full conversation history on every request.

**Request:**
```json
{
  "messages": [
    {"role": "user", "content": "I'm hiring a senior Java developer."},
    {"role": "assistant", "content": "What's the primary stack — backend-heavy or full-stack?"},
    {"role": "user", "content": "Backend. Java, Spring, SQL, and AWS."}
  ]
}
```

**Response:**
```json
{
  "reply": "For a senior backend engineer owning Java/Spring/SQL/AWS...",
  "recommendations": [
    {
      "name": "Core Java (Advanced Level) (New)",
      "url": "https://www.shl.com/products/product-catalog/view/core-java-advanced-level-new/",
      "test_type": "K"
    },
    {
      "name": "Spring (New)",
      "url": "https://www.shl.com/products/product-catalog/view/spring-new/",
      "test_type": "K"
    }
  ],
  "end_of_conversation": false
}
```

**Schema rules:**
- `recommendations` is always a list, never null — empty `[]` when clarifying or refusing
- `recommendations` contains 1–10 items when committed to a shortlist
- Every URL is from the scraped SHL catalog (`/products/product-catalog/view/`)
- `test_type` is a clean code string: `"K"`, `"A"`, `"P"`, `"B"`, `"S"`, `"C"`, `"D"`, or comma-joined for multi-type
- `end_of_conversation: true` only after explicit user confirmation

**Test type codes:**
| Code | Label |
|---|---|
| `A` | Ability & Aptitude |
| `P` | Personality & Behavior |
| `K` | Knowledge & Skills |
| `B` | Biodata & Situational Judgment |
| `S` | Simulations |
| `C` | Competencies |
| `D` | Development & 360 |

---

## Evaluation Results

Measured against 10 public conversation traces using a local evaluation harness.

| Metric | Score |
|---|---|
| Schema compliance | **100%** (0 failures) |
| Average Recall@10 | **0.652** |
| Behavior probes | **25 / 25** passing |
| Schema tests | **20 / 20** passing |
| Average latency | **10.4s** |
| P90 latency | **16.3s** |

Per-conversation Recall@10:

```
C1  CXO leadership selection    ████████░░  0.50
C2  Senior Rust engineer        ████████░░  0.50
C3  Contact center (500 agents) ████████████  0.75
C4  Graduate financial analysts ████████████████  0.80
C5  Sales org reskilling        ████████░░  0.50
C6  Chemical plant operators    ████████░░  0.50
C7  Bilingual healthcare admin  ████████░░  0.50
C8  Admin MS Office screening   ████████████████████  1.00
C9  Full-stack engineer (7-turn) ████████████████  0.80
C10 Graduate management trainee ████████████  0.67
```

---

## Project Structure

```
shl-recc/
├── app/
│   ├── main.py          # FastAPI app, lifespan handler, endpoints
│   ├── models.py        # Pydantic schema (API contract + internal types)
│   ├── agent.py         # ChatHandler — CSG-RAG pipeline orchestration
│   ├── retriever.py     # Hybrid retrieval (FAISS + BM25 + metadata filter)
│   ├── catalog.py       # CatalogStore — single point of access for all data
│   ├── prompts.py       # All LLM prompts (slot extractor, ranker, comparison)
│   ├── llm_client.py    # Gemini primary + OpenAI fallback with retry logic
│   └── guardrails.py    # Input validation and URL whitelist enforcement
├── scraper/
│   ├── catalog_scraper.py   # SHL catalog scraper (httpx + BeautifulSoup)
│   ├── embed_catalog.py     # Embedding pipeline (MiniLM → FAISS index)
│   └── validate_catalog.py  # Data quality checks and statistics
├── data/
│   ├── catalog.json         # 377 Individual Test Solutions (scraped)
│   └── vector_index/
│       ├── index.faiss      # FAISS IndexFlatIP, 384-dim, 377 vectors
│       └── metadata.json    # Parallel metadata array
├── tests/
│   ├── test_schema.py       # 20 schema compliance tests
│   ├── test_behaviors.py    # 25 behavior probe tests
│   ├── test_traces.py       # Slot extractor replay on 10 conversation traces
│   ├── eval_harness.py      # End-to-end Recall@10 evaluation
│   └── traces/              # 10 public conversation trace files
├── docs/
│   └── APPROACH.md          # 2-page approach document
├── Dockerfile
├── render.yaml
├── requirements.txt
└── .env.example
```

---

## Setup

### Prerequisites
- Conda (Anaconda or Miniconda)
- Google Gemini API key ([get one here](https://aistudio.google.com/))
- OpenAI API key ([get one here](https://platform.openai.com/))

### Installation

```bash
# Clone the repository
git clone https://github.com/shvn22k/shl-recc.git
cd shl-recc

# Create and activate conda environment
conda create -n shl-recommender python=3.11 -y
conda activate shl-recommender

# Install dependencies
pip install -r requirements.txt

# Configure environment
cp .env.example .env
# Edit .env and add your GEMINI_API_KEY and OPENAI_API_KEY
```

### Running Locally

```bash
# The catalog and vector index are already committed to the repo.
# No scraping needed — just start the server.
uvicorn app.main:app --reload --port 8000
```

The service is ready when you see:
```
INFO     CatalogStore ready
INFO     Startup complete. Ready to serve requests.
```

### Rebuilding the Catalog (Optional)

Only needed if you want to re-scrape SHL's catalog:

```bash
# Scrape (takes ~45-60 minutes, polite 1.5s delay between requests)
python scraper/catalog_scraper.py

# Validate
python scraper/validate_catalog.py

# Rebuild embeddings and FAISS index
python scraper/embed_catalog.py
```

---

## Running Tests

Requires the server running on `localhost:8000`.

```bash
# Start server in one terminal
uvicorn app.main:app --port 8000

# In a second terminal:

# Schema compliance (fast, ~2 min)
pytest tests/test_schema.py -v

# Behavior probes (LLM calls, ~9 min)
pytest tests/test_behaviors.py -v

# Slot extractor traces (LLM calls, no HTTP server needed)
pytest tests/test_traces.py -v

# Full evaluation harness with Recall@10 (~20-30 min)
python tests/eval_harness.py
```

---

## Deployment

The service is containerized and configured for [Render.com](https://render.com) free tier.

**Environment variables required:**
```
GEMINI_API_KEY      Your Gemini API key
OPENAI_API_KEY      Your OpenAI API key
LLM_PROVIDER        openai  (recommended for consistent latency)
```

All other variables have sensible defaults in `.env.example`.

The Docker image includes the pre-scraped catalog and FAISS index — no runtime scraping or index building.

---

## Technical Stack

| Component | Choice | Reason |
|---|---|---|
| Framework | FastAPI + Pydantic v2 | Schema enforcement at the API boundary |
| LLM (primary) | OpenAI GPT-4o-mini | Consistent latency, JSON mode support |
| LLM (fallback) | Google Gemini 2.5 Flash | Fast, generous context window |
| Embeddings | MiniLM-L6-v2 (local) | No API cost, no rate limits, ~80MB |
| Vector search | FAISS IndexFlatIP | Exact cosine similarity, zero overhead at 377 vectors |
| Keyword search | BM25 Okapi | Boosts exact tech keyword matches |
| Scraping | httpx + BeautifulSoup | SHL catalog is server-rendered HTML, no JS needed |
| Deployment | Render.com (Docker) | Free tier, Docker-based, persistent process |

---

## Approach Document

See [`docs/APPROACH.md`](docs/APPROACH.md) for the full technical approach document covering:
- Problem framing
- Architecture decisions and trade-offs
- Challenges encountered and how they were fixed
- Evaluation results analysis
- What would improve with more time

---

## Development Phases

- [x] Phase 0: Scaffold & environment
- [x] Phase 1: Catalog scraper (377 Individual Test Solutions)
- [x] Phase 2: Embedding pipeline (FAISS + MiniLM)
- [x] Phase 3: FastAPI core + schema enforcement
- [x] Phase 4: Slot extractor + agent logic
- [x] Phase 5+6: Hybrid retriever + LLM ranker
- [x] Phase 7+8: Guardrails + full test suite (45/45 passing)
- [x] Phase 9: Deployment (Render.com)
- [x] Phase 10: README + repo finalized

---

## Author

**Shiven Shandil**  
[@shvn22k](https://github.com/shvn22k)  

Built as a hiring assessment submission for SHL Labs AI Intern role.
