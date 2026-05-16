# Phase 0 — Project Scaffold & Environment Setup

## Read This First — Full Project Context

You are setting up a project for a **hiring assessment submission to SHL Labs** for an AI Intern role. Understanding the full context is critical so every file you create, every name you choose, and every config you write is aligned with the end goal.

### What We Are Building

A **conversational SHL Assessment Recommender** — a publicly deployed FastAPI service that acts as an intelligent agent for hiring managers. The agent takes a recruiter's vague hiring intent (e.g. "I need something for senior engineers") and guides them through a natural multi-turn conversation to arrive at a grounded shortlist of SHL assessments from SHL's official product catalog.

This is **not a simple search tool**. It is a stateless conversational agent with:
- Multi-turn dialogue management
- Intelligent clarification before recommending
- Real-time refinement when the user changes constraints
- Grounded comparison of assessments using catalog data only
- Hard refusal of off-topic, legal, and out-of-scope questions

### The Evaluation Criteria

SHL will run an **automated evaluation harness** against our deployed API. It scores us on:

1. **Hard evals (must pass 100%)** — Schema compliance, all URLs from scraped catalog only, max 8 conversation turns honored
2. **Recall@10** — How many of the truly relevant assessments appear in our top-10 recommendations
3. **Behavior probes** — Refusal works, no premature recommendations, user edits honored, no hallucination

Failing any hard eval = automatic disqualification. This means schema correctness is non-negotiable.

### The API Contract (Non-Negotiable)

```
GET  /health  →  {"status": "ok"}

POST /chat
  Request:  { "messages": [{"role": "user"|"assistant", "content": "string"}, ...] }
  Response: { "reply": "string", "recommendations": [...], "end_of_conversation": bool }

recommendations item: { "name": "string", "url": "string", "test_type": "string" }
```

- `recommendations` is always `[]` (empty array) when clarifying, comparing, or refusing — never null, never omitted
- `end_of_conversation: true` only when the user has explicitly confirmed they are done
- Every URL must exist in our scraped SHL catalog — no exceptions

### The Retrieval Architecture — CSG-RAG

We are implementing a custom approach called **Conversational Slot-Guided RAG (CSG-RAG)**:

1. **Slot Extractor** (LLM Call 1) — Reads the full conversation history, extracts structured intent: role, seniority, purpose, job family, language, constraints, explicit adds/drops
2. **Metadata pre-filter** — Hard filter on catalog by job level, job family (no LLM, pure logic)
3. **Semantic search** — FAISS vector similarity on filtered catalog (MiniLM embeddings)
4. **BM25 keyword re-rank** — Boosts exact tech/domain keyword matches within semantic results
5. **LLM Ranker** (LLM Call 2) — Chain-of-thought selection of final 1–10 assessments from top-20 candidates

Total LLM calls per request: **2**. This keeps us well under the 30-second hard timeout.

### LLM Stack

- **Primary**: Google Gemini 2.0 Flash (`gemini-2.0-flash`) — fast, free tier available
- **Fallback**: OpenAI GPT-4o-mini — used if Gemini fails or rate-limits
- **Embeddings**: `sentence-transformers/all-MiniLM-L6-v2` — local, free, CPU-compatible

### Deployment Target

- Platform: **Render.com** (free tier web service, Docker-based)
- The service must be publicly reachable, cold start within 2 minutes
- All catalog data and FAISS index committed to the repo (no runtime scraping)

### Data Source

- SHL product catalog: `https://www.shl.com/solutions/products/product-catalog/`
- Scope: **Individual Test Solutions only** (not Pre-packaged Job Solutions)
- Fields to capture: name, URL, test_type, description, job_levels, job_families, languages, duration, remote_testing, adaptive_irt

---

## Your Task — Phase 0

Set up the complete project scaffold on a **Windows 11** machine using **Conda** for environment management. The developer uses **Antigravity IDE**.

Create everything listed below exactly as specified. Do not add extra files, do not rename things, do not reorganize. Follow the structure precisely — later phases depend on these exact paths.

---

## Step 1 — Create the Conda Environment

Open **Anaconda Prompt** (not PowerShell, not CMD) and run:

```bash
conda create -n shl-recommender python=3.11 -y
conda activate shl-recommender
```

Verify Python version:
```bash
python --version
# Must output: Python 3.11.x
```

---

## Step 2 — Create the Project Root

Choose a location for the project (e.g. `C:\Projects\`) and create the folder:

```bash
mkdir shl-recommender
cd shl-recommender
```

---

## Step 3 — Create the Full Folder Structure

Run these commands from inside the `shl-recommender/` root:

```bash
mkdir app
mkdir scraper
mkdir data
mkdir data\vector_index
mkdir tests
mkdir tests\traces
mkdir docs
mkdir docs\diagrams
```

Your folder tree should look exactly like this:

```
shl-recommender/
├── app/
├── scraper/
├── data/
│   └── vector_index/
├── tests/
│   └── traces/
└── docs/
    └── diagrams/
```

---

## Step 4 — Create All Python Package Files

Create empty `__init__.py` files to make directories into packages:

```bash
type nul > app\__init__.py
type nul > scraper\__init__.py
type nul > tests\__init__.py
```

---

## Step 5 — Create All Source Files (Empty Stubs)

Create every source file as an empty stub. Later phases will fill these in. Run from project root:

```bash
type nul > app\main.py
type nul > app\models.py
type nul > app\agent.py
type nul > app\retriever.py
type nul > app\prompts.py
type nul > app\guardrails.py
type nul > app\catalog.py
type nul > scraper\catalog_scraper.py
type nul > scraper\embed_catalog.py
type nul > scraper\validate_catalog.py
type nul > tests\test_schema.py
type nul > tests\test_traces.py
type nul > tests\test_behaviors.py
type nul > tests\eval_harness.py
```

---

## Step 6 — Create `requirements.txt`

Create `requirements.txt` in the project root with this exact content:

```
fastapi==0.111.0
uvicorn[standard]==0.29.0
pydantic==2.7.1
httpx==0.27.0
beautifulsoup4==4.12.3
lxml==5.2.2
playwright==1.44.0
sentence-transformers==2.7.0
faiss-cpu==1.8.0
rank-bm25==0.2.2
google-generativeai==0.7.2
openai==1.30.1
python-dotenv==1.0.1
pytest==8.2.0
pytest-asyncio==0.23.6
tenacity==8.3.0
```

**Why each dependency:**
- `fastapi` + `uvicorn` — the API framework, as specified by the assessment
- `pydantic` — schema enforcement at the API boundary (hard eval protection)
- `httpx` — async HTTP for LLM API calls and scraping
- `beautifulsoup4` + `lxml` — HTML parsing for the catalog scraper
- `playwright` — headless browser for scraping JavaScript-rendered catalog pages
- `sentence-transformers` — local MiniLM embeddings, no API cost
- `faiss-cpu` — vector similarity search for the catalog index
- `rank-bm25` — keyword re-ranking layer in our hybrid retrieval
- `google-generativeai` — Gemini API client
- `openai` — OpenAI fallback client
- `python-dotenv` — environment variable loading
- `pytest` + `pytest-asyncio` — test framework
- `tenacity` — retry logic with backoff for LLM API calls

---

## Step 7 — Install Dependencies

```bash
pip install -r requirements.txt
```

Then install Playwright browsers (needed for JavaScript-rendered catalog scraping):

```bash
playwright install chromium
```

Verify key installations:
```bash
python -c "import fastapi; print('fastapi OK')"
python -c "import faiss; print('faiss OK')"
python -c "from sentence_transformers import SentenceTransformer; print('sentence-transformers OK')"
python -c "import google.generativeai; print('gemini OK')"
```

All four must print OK. If any fail, install that package individually with `pip install <package>`.

---

## Step 8 — Create `.env.example`

Create `.env.example` in the project root:

```
# Primary LLM — Google Gemini
GEMINI_API_KEY=your_gemini_api_key_here

# Fallback LLM — OpenAI
OPENAI_API_KEY=your_openai_api_key_here

# LLM provider to use (gemini or openai)
LLM_PROVIDER=gemini

# Model names
GEMINI_MODEL=gemini-2.0-flash
OPENAI_MODEL=gpt-4o-mini

# Embedding model (local, no key needed)
EMBEDDING_MODEL=all-MiniLM-L6-v2

# Data paths
CATALOG_PATH=data/catalog.json
INDEX_PATH=data/vector_index

# Logging
LOG_LEVEL=INFO

# App settings
MAX_TURNS=8
MAX_RECOMMENDATIONS=10
REQUEST_TIMEOUT_SECONDS=25
```

---

## Step 9 — Create `.env` (Your Actual Keys)

Create `.env` in the project root by copying `.env.example` and filling in your real keys:

```bash
copy .env.example .env
```

Then open `.env` and replace:
- `your_gemini_api_key_here` → your actual Gemini API key
- `your_openai_api_key_here` → your actual OpenAI API key

**This file must never be committed to git.**

---

## Step 10 — Create `.gitignore`

Create `.gitignore` in the project root with this content:

```gitignore
# Environment
.env
.venv/
__pycache__/
*.pyc
*.pyo
*.pyd
*.egg-info/
dist/
build/
.eggs/

# Conda
conda-meta/

# IDE
.antigravity/
.cursor/
.vscode/
*.swp
*.swo

# Testing
.pytest_cache/
htmlcov/
.coverage
coverage.xml

# OS
.DS_Store
Thumbs.db
desktop.ini

# Data — keep catalog.json and vector index committed
# (they are built once and should be version controlled)
# Do NOT add data/ to gitignore

# Logs
*.log
logs/
```

---

## Step 11 — Create `Dockerfile`

Create `Dockerfile` in the project root:

```dockerfile
FROM python:3.11-slim

WORKDIR /app

# Install system dependencies for lxml and playwright
RUN apt-get update && apt-get install -y \
    gcc \
    g++ \
    libxml2-dev \
    libxslt-dev \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Copy and install Python dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy application code and data
COPY app/ ./app/
COPY data/ ./data/
COPY .env.example ./.env.example

# Expose port
EXPOSE 8000

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=60s --retries=3 \
    CMD curl -f http://localhost:8000/health || exit 1

# Run the application
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]
```

---

## Step 12 — Create `render.yaml`

Create `render.yaml` in the project root (Render.com deployment config):

```yaml
services:
  - type: web
    name: shl-recommender
    runtime: docker
    dockerfilePath: ./Dockerfile
    plan: free
    envVars:
      - key: GEMINI_API_KEY
        sync: false
      - key: OPENAI_API_KEY
        sync: false
      - key: LLM_PROVIDER
        value: gemini
      - key: GEMINI_MODEL
        value: gemini-2.0-flash
      - key: OPENAI_MODEL
        value: gpt-4o-mini
      - key: EMBEDDING_MODEL
        value: all-MiniLM-L6-v2
      - key: CATALOG_PATH
        value: data/catalog.json
      - key: INDEX_PATH
        value: data/vector_index
      - key: LOG_LEVEL
        value: INFO
      - key: MAX_TURNS
        value: "8"
      - key: MAX_RECOMMENDATIONS
        value: "10"
      - key: REQUEST_TIMEOUT_SECONDS
        value: "25"
    healthCheckPath: /health
```

---

## Step 13 — Create `app/main.py` (Working Stub)

This is not an empty stub — write a working health endpoint so we can verify the app runs:

```python
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from dotenv import load_dotenv
import os

load_dotenv()

app = FastAPI(
    title="SHL Assessment Recommender",
    description="Conversational agent for recommending SHL assessments",
    version="0.1.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/health")
async def health():
    return {"status": "ok"}


@app.post("/chat")
async def chat_stub():
    # Stub — will be implemented in Phase 3
    return {
        "reply": "Service is up. Full agent coming soon.",
        "recommendations": [],
        "end_of_conversation": False,
    }
```

---

## Step 14 — Create `app/models.py` (Full Schema)

Write the complete Pydantic schema now — this never changes and protects us from schema violations:

```python
from pydantic import BaseModel, Field, field_validator
from typing import Optional


class Message(BaseModel):
    role: str = Field(..., pattern="^(user|assistant)$")
    content: str = Field(..., min_length=1)


class Recommendation(BaseModel):
    name: str = Field(..., min_length=1)
    url: str = Field(..., min_length=1)
    test_type: str = Field(..., min_length=1)


class ChatRequest(BaseModel):
    messages: list[Message] = Field(..., min_length=1)


class ChatResponse(BaseModel):
    reply: str = Field(..., min_length=1)
    recommendations: list[Recommendation] = Field(default_factory=list)
    end_of_conversation: bool = Field(default=False)

    @field_validator("recommendations")
    @classmethod
    def recommendations_max_ten(cls, v):
        if len(v) > 10:
            raise ValueError("recommendations must not exceed 10 items")
        return v
```

---

## Step 15 — Create `data/.gitkeep` Files

Ensure the data directories are tracked by git even when empty:

```bash
type nul > data\.gitkeep
type nul > data\vector_index\.gitkeep
type nul > docs\diagrams\.gitkeep
type nul > tests\traces\.gitkeep
```

---

## Step 16 — Create `pytest.ini`

```ini
[pytest]
asyncio_mode = auto
testpaths = tests
python_files = test_*.py
python_classes = Test*
python_functions = test_*
```

---

## Step 17 — Create `README.md`

```markdown
# SHL Assessment Recommender

A conversational AI agent that recommends SHL assessments based on hiring manager requirements.

## Architecture

This project implements **Conversational Slot-Guided RAG (CSG-RAG)** — a custom retrieval approach that:
1. Extracts structured intent (slots) from conversation history via LLM
2. Pre-filters the SHL catalog by metadata (hard constraints)
3. Retrieves candidates via FAISS semantic search + BM25 keyword re-ranking
4. Re-ranks and selects final recommendations via chain-of-thought LLM reasoning

## API

```
GET  /health          → {"status": "ok"}
POST /chat            → {"reply": "...", "recommendations": [...], "end_of_conversation": bool}
```

## Setup

```bash
conda create -n shl-recommender python=3.11 -y
conda activate shl-recommender
pip install -r requirements.txt
cp .env.example .env
# Fill in your API keys in .env
```

## Running Locally

```bash
uvicorn app.main:app --reload --port 8000
```

## Project Structure

```
shl-recommender/
├── app/              # FastAPI application
├── scraper/          # Catalog scraper and embedding pipeline
├── data/             # Catalog JSON and FAISS vector index
├── tests/            # Schema, trace, and behavior tests
└── docs/             # Architecture diagrams and approach document
```

## Development Phases

- [x] Phase 0: Scaffold & environment
- [ ] Phase 1: Catalog scraper
- [ ] Phase 2: Embedding pipeline
- [ ] Phase 3: FastAPI core + schema
- [ ] Phase 4: Slot extractor + agent logic
- [ ] Phase 5: Hybrid retriever
- [ ] Phase 6: LLM ranker + full /chat endpoint
- [ ] Phase 7: Guardrails + refusal handling
- [ ] Phase 8: Test suite
- [ ] Phase 9: Deployment
- [ ] Phase 10: Approach document
```

---

## Step 18 — Verify the App Runs

Start the development server:

```bash
uvicorn app.main:app --reload --port 8000
```

In a new terminal, test both endpoints:

```bash
curl http://localhost:8000/health
# Expected: {"status":"ok"}

curl -X POST http://localhost:8000/chat ^
  -H "Content-Type: application/json" ^
  -d "{\"messages\":[{\"role\":\"user\",\"content\":\"Hello\"}]}"
# Expected: {"reply":"Service is up...","recommendations":[],"end_of_conversation":false}
```

Both must return the expected responses. If either fails, do not proceed.

---

## Step 19 — Initialize Git

From the project root:

```bash
git init
git add .
git commit -m "feat: phase 0 — project scaffold and environment setup

- Conda environment with Python 3.11
- Full folder structure: app, scraper, data, tests, docs
- Pydantic schema models (non-negotiable API contract)
- Working /health endpoint and /chat stub
- Dockerfile for Render.com deployment
- render.yaml deployment config
- requirements.txt with all dependencies pinned
- .env.example with all configuration variables documented
- pytest.ini for test configuration
- .gitignore excluding secrets and build artifacts"
```

Then connect to your GitHub repo:

```bash
git remote add origin https://github.com/YOUR_USERNAME/shl-recommender.git
git branch -M main
git push -u origin main
```

---

## Final Verification Checklist

Before marking Phase 0 complete, confirm every item:

- [ ] `conda activate shl-recommender` works without error
- [ ] `python --version` shows 3.11.x
- [ ] All 4 import checks pass (fastapi, faiss, sentence_transformers, google.generativeai)
- [ ] Folder structure matches the tree exactly
- [ ] `.env` exists with real API keys (not committed to git)
- [ ] `.env` is listed in `.gitignore`
- [ ] `uvicorn app.main:app --reload` starts without error
- [ ] `GET /health` returns `{"status":"ok"}`
- [ ] `POST /chat` returns valid stub response matching schema
- [ ] `git log` shows the initial commit
- [ ] Repo is pushed to GitHub

**Do not proceed to Phase 1 until every box is checked.**

---

## What Comes Next

**Phase 1** will implement the catalog scraper (`scraper/catalog_scraper.py`). It will:
- Use Playwright to load the SHL catalog pages (JavaScript-rendered)
- Paginate through all Individual Test Solutions
- Extract: name, URL, test_type, description, job_levels, job_families, languages, duration
- Output a clean `data/catalog.json`

Everything we build from Phase 1 onward depends on the catalog data, so it must be thorough and correct.
