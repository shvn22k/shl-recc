# Approach Document — SHL Assessment Recommender

**Author:** Shiven Shandil  
**Submission:** SHL Labs AI Intern Assessment

---

## Problem Framing

The task is to build a conversational agent that takes a hiring manager from a
vague hiring intent to a grounded shortlist of SHL Individual Test Solution
assessments. The core challenges are:

1. **Grounding** — every recommendation must come from the scraped SHL catalog.
   Hallucinated URLs or invented assessments are automatic disqualifiers.
2. **Conversational coherence** — the agent must track state across turns
   (add/drop instructions, clarification answers) without server-side session storage.
3. **Retrieval quality** — the catalog has 377 assessments across very different
   categories. A naive keyword search misses semantic matches; pure semantic
   search misses exact technology names.
4. **Latency** — a 30-second hard timeout with a cold start of ~30 seconds on
   free-tier deployment leaves very little headroom.

---

## Architecture — Conversational Slot-Guided RAG (CSG-RAG)

We designed a custom retrieval approach called CSG-RAG instead of using a
standard RAG pipeline. The key insight is that this is an *explicit intent*
problem, not an *implicit preference* problem. The hiring manager tells us
exactly what they need — our job is to parse that intent precisely and use
it to filter and rank a small, well-defined catalog.

The pipeline runs on every `/chat` request:

```
Full conversation history
        │
        ▼
┌─────────────────────────────┐
│     Guardrail Pre-checks    │  Injection / legal / off-topic / gibberish
└──────────────┬──────────────┘
               │
               ▼
┌─────────────────────────────┐
│    Slot Extractor (LLM 1)   │  Gemini 2.5 Flash with JSON mode
│  role · seniority · purpose │  Extracts structured intent from history
│  language · constraints     │  Decides: clarify / retrieve / compare / close
│  explicit adds/drops        │
└──────────────┬──────────────┘
               │
       ┌───────┴────────┐
       ▼                ▼
  needs_clarify    ready_to_retrieve
  → ask question   │
  → return []      ▼
            ┌─────────────────────────────┐
            │     Hybrid Retrieval        │
            │  1. FAISS semantic (top-40) │
            │  2. Metadata pre-filter     │
            │  3. BM25 keyword re-rank    │
            │  4. Explicit injection      │
            └──────────────┬──────────────┘
                           │ top-20 candidates
                           ▼
            ┌─────────────────────────────┐
            │    LLM Ranker (LLM 2)       │  Chain-of-thought selection
            │  Selects 1-10 assessments   │  Default OPQ32r / Verify G+
            │  Writes terse reply         │  Honors explicit add/drop
            └──────────────┬──────────────┘
                           │
                           ▼
            ┌─────────────────────────────┐
            │    Post-Processing          │
            │  Deduplication              │
            │  Default injection          │
            │  URL whitelist enforcement  │
            └──────────────┬──────────────┘
                           │
                           ▼
                    ChatResponse
```

### Why not standard RAG?

Standard RAG pipelines embed the user's query and retrieve the top-k documents.
This works poorly here because:

- A query like "I need an assessment for senior engineers" retrieves generic
  management tools, not Java/Python/SQL knowledge tests
- The catalog has sparse metadata (no job families, inconsistent job levels)
  so pure vector search degrades on under-described assessments
- The LLM must know *not to recommend* on the first turn for vague queries —
  standard RAG has no notion of conversation phases

CSG-RAG solves this by separating intent understanding (LLM 1) from
assessment selection (LLM 2), with a structured intermediate representation
(SlotState) that drives both retrieval and ranking.

---

## Key Design Decisions

### 1. Two LLM calls per request, not one

We use one LLM call for slot extraction and a second for ranking. This
separation is critical:

- The extractor runs at temperature 0.1 for deterministic parsing
- The ranker runs at temperature 0.2 for nuanced selection
- Each prompt is focused on one task — extraction prompts don't rank,
  ranking prompts don't parse

Combining both into one call would require a much longer, more complex prompt
with worse reliability on both tasks.

### 2. Hybrid retrieval: semantic + BM25

Neither pure semantic search nor pure keyword search works alone:

- Semantic: "I need problem-solving tests" → correctly surfaces Verify G+
  even though the description doesn't contain "problem-solving"
- BM25: "Java developer" → correctly surfaces Core Java (Advanced) above
  generic programming tests
- Combined score: 0.6 × semantic + 0.4 × BM25

The 60/40 split was calibrated empirically. Semantic dominates because most
queries are expressed in natural language, but BM25 provides enough signal
to surface exact technology matches.

### 3. Job level filter restricted to graduate/entry-level only

Early testing showed that filtering by job level for senior+ roles silently
eliminated most technical knowledge tests from the candidate pool. These tests
(Java, Spring, SQL, AWS etc.) have inconsistent or absent job level metadata
in SHL's catalog — they're general-purpose instruments. The filter was
restricted to graduate and entry-level where it genuinely discriminates.

### 4. OPQ32r and Verify G+ as programmatic defaults

The 10 conversation traces revealed consistent patterns: OPQ32r appears in
every mid/senior selection battery; Verify G+ appears in every graduate and
senior-IC battery. Rather than relying on the LLM to remember these rules
under prompt pressure, we inject them programmatically after the ranker runs
and before URL whitelist enforcement. The LLM can focus on domain-specific
selection; defaults are handled deterministically.

### 5. URL whitelist as the last line of defense

Every response goes through `enforce_url_whitelist()` before reaching the
client. This strips any URL not in the scraped catalog regardless of how it
got there — LLM hallucination, prompt injection, or edge case bugs. It is
the single most important guardrail for the hard evaluation criteria.

---

## Challenges and How We Fixed Them

### Challenge 1: Retrieval failure on tech roles with catalog gaps

**Problem:** Querying for "senior Rust engineer" returned leadership/management
reports because the `senior-ic` job level filter reduced candidates to 4 items,
triggering a safety net that dumped all 377 unfiltered vectors. The FAISS
query led with seniority framing, which embedded closest to management tools.

**Fix:** Two-part solution:
1. Restrict job level filter to graduate/entry-level only
2. For tech roles (detected by extracting known technology keywords from the
   role field), lead the FAISS query with domain keywords + "knowledge skills
   technical assessment" instead of seniority framing

Result: C2 (senior Rust engineer) now correctly surfaces Linux, Networking,
and Smart Interview Live Coding as the top candidates.

### Challenge 2: Duplicate Verify G+ variants in recommendations

**Problem:** The ranker selected "Verify G+" (an ability test report variant)
from candidates while `_inject_defaults` independently added "SHL Verify
Interactive G+" — the actual assessment instrument. Both appeared in the
final list occupying 2 of 10 slots.

**Fix:** Semantic deduplication using shared-term counting. Two assessments
sharing 2+ meaningful terms (length ≥ 2, excluding stopwords like "test",
"report", "assessment") are considered duplicates. The `_inject_defaults`
method now replaces the weaker variant rather than skipping the injection.

### Challenge 3: C7 (bilingual healthcare) recall near zero

**Problem:** The query included language="Spanish" which caused FAISS to
surface Spanish-language personality assessments instead of HIPAA and
Medical Terminology tests.

**Fix:** Language is excluded from the FAISS query text entirely. The ranker
already sees the language slot in its structured input and can apply it during
selection. Including it in the semantic query was actively harmful because
"Spanish" is a stronger semantic signal than "HIPAA" or "Medical Terminology".

### Challenge 4: P90 latency at 24s (over 20s target)

**Problem:** Gemini 429 rate limit errors caused 12s delays per call as the
client waited for the retry window.

**Fix:** Set `LLM_PROVIDER=openai` as the deployment default. OpenAI
GPT-4o-mini has more consistent latency under the usage patterns of this
system. Gemini remains configured as a fallback.

Result: Average latency dropped from 16.0s to 10.4s, P90 from 24.0s to 16.3s.

---

## Evaluation Results

| Metric | Score |
|---|---|
| Schema compliance | 100% (0 failures across all test paths) |
| Average Recall@10 | 0.652 (65.2%) |
| Behavior probes | 25/25 passing |
| Schema tests | 20/20 passing |
| P90 latency | 16.3s |

Per-conversation Recall@10:

| Trace | Recall | Notes |
|---|---|---|
| C1 — CXO leadership | 0.50 | Report variants (UCR, Leadership) hard to retrieve |
| C2 — Senior Rust engineer | 0.50 | No Rust test exists; proxies surfaced correctly |
| C3 — Contact center | 0.75 | SVAR accent variant correctly clarified |
| C4 — Graduate finance | 0.80 | All key instruments retrieved |
| C5 — Sales audit | 0.50 | Sales Transformation 2.0 is a niche catalog item |
| C6 — Plant operator safety | 0.50 | DSI vs 8.0 comparison correctly handled |
| C7 — Bilingual healthcare | 0.50 | Fixed with language exclusion from FAISS query |
| C8 — Admin MS Office | 1.00 | Full recall after explicit_additions fix |
| C9 — Full-stack engineer | 0.80 | Multi-turn refinement works correctly |
| C10 — Graduate trainee | 0.67 | Stable |

---

## What Would Improve With More Time

1. **Re-scrape with Playwright** — SHL's detail pages may have JavaScript-rendered
   content not captured by the current httpx scraper. Job families in particular
   were 0% populated. With job families, the metadata filter would be more precise.

2. **Fine-tuned embeddings** — MiniLM-L6-v2 is a general-purpose model.
   Fine-tuning on SHL assessment descriptions and HR job titles would improve
   semantic retrieval quality, particularly for niche instruments like
   Sales Transformation 2.0.

3. **Conversation-level caching** — The two LLM calls add 8-12s per request.
   Caching slot extraction results when the conversation hasn't changed would
   eliminate the extractor call on turn 2+ for most refinement patterns.

4. **Expanded ground truth** — The Recall@10 measurement uses 10 manually
   curated ground truth sets. A larger labeled dataset would give more
   reliable signal for retrieval tuning.
