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
