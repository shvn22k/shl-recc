"""
Hybrid retriever for SHL Assessment Recommender.

Implements the retrieval stage of CSG-RAG:
  1. Query construction from SlotState
  2. FAISS semantic search (top-40 candidates)
  3. Metadata pre-filter (job level, test type preferences)
  4. BM25 keyword re-rank
  5. Direct lookup for explicit_additions

Returns a ranked list of CandidateAssessment objects for the LLM ranker.
"""

import logging
import re
from rank_bm25 import BM25Okapi
from app.models import SlotState, CandidateAssessment

logger = logging.getLogger(__name__)

# Minimum candidates to pass to ranker even after aggressive filtering
MIN_CANDIDATES = 15
# Maximum candidates passed to ranker (controls ranker prompt length)
MAX_CANDIDATES = 20
# FAISS initial retrieval pool — wide net for recall
FAISS_TOP_K = 40


def build_retrieval_query(slots: SlotState) -> str:
    """
    Build a natural language query string from the SlotState.

    This string is embedded and used for FAISS semantic search.
    The order of components matters — earlier terms have slightly
    more influence on the embedding.

    Strategy:
    - Lead with role + seniority (most discriminating)
    - Add explicit test type preferences
    - Add industry/domain context
    - Add purpose framing
    - Add constraint signals
    """
    parts = []

    # Core identity — role + seniority
    if slots.role:
        parts.append(slots.role)
    if slots.seniority:
        seniority_phrases = {
            "graduate": "graduate entry level recent graduate",
            "entry-level": "entry level junior",
            "mid-professional": "mid level professional",
            "senior-ic": "senior individual contributor experienced",
            "manager": "manager team lead people manager",
            "director": "director head of department senior leader",
            "executive": "executive CXO C-suite senior leadership",
        }
        parts.append(seniority_phrases.get(slots.seniority, slots.seniority))

    # Explicit test type preferences
    test_type_phrases = {
        "cognitive": "cognitive ability reasoning numerical verbal inductive",
        "personality": "personality behavior workplace OPQ",
        "situational-judgment": "situational judgment SJT scenarios",
        "knowledge": "knowledge skills technical domain specific",
        "simulation": "simulation realistic job preview work sample",
        "behavioral": "behavioral competency assessment",
    }
    for tt in slots.explicit_test_types:
        if tt in test_type_phrases:
            parts.append(test_type_phrases[tt])

    # Industry / domain
    if slots.industry:
        parts.append(slots.industry)

    # Purpose framing
    purpose_phrases = {
        "selection": "hiring selection assessment battery",
        "development": "development feedback coaching growth",
        "screening": "high volume screening filter",
        "audit": "talent audit review workforce assessment",
        "reskilling": "reskilling upskilling development learning",
    }
    if slots.purpose:
        parts.append(purpose_phrases.get(slots.purpose, slots.purpose))

    # Language constraint
    if slots.language and slots.language.lower() != "english":
        parts.append(f"{slots.language} language assessment")

    # Time constraint
    if slots.time_constraint == "short":
        parts.append("short quick brief assessment under 20 minutes")

    # Volume
    if slots.volume == "high":
        parts.append("high volume screening large scale")

    query = " ".join(parts)
    logger.debug(f"Retrieval query: '{query}'")
    return query


def tokenize_for_bm25(text: str) -> list[str]:
    """
    Tokenize text for BM25 indexing.
    Lowercase, split on non-alphanumeric, filter short tokens.
    """
    tokens = re.findall(r"[a-zA-Z0-9]+", text.lower())
    return [t for t in tokens if len(t) > 2]


def bm25_rerank(
    candidates: list[dict],
    query: str,
    slots: SlotState,
) -> list[dict]:
    """
    Re-rank candidates using BM25 keyword matching.

    BM25 boosts assessments whose name/description contain exact
    keywords from the query and role. This catches cases where
    semantic search ranks a general test above a specific one
    (e.g. "Java Advanced" should score higher than "General Coding"
    when the query mentions Java specifically).

    The final score is a weighted combination:
        final_score = 0.6 * semantic_score + 0.4 * bm25_score_normalized
    """
    if not candidates:
        return candidates

    # Build BM25 corpus from candidate name + description
    corpus = []
    for c in candidates:
        doc_text = f"{c.get('name', '')} {c.get('description', '')}"
        corpus.append(tokenize_for_bm25(doc_text))

    # Build BM25 query — role + explicit terms + technology keywords
    query_parts = [query]
    if slots.role:
        query_parts.append(slots.role)
    # Extract technology keywords from role (e.g. "Java", "Python", "AWS")
    tech_keywords = extract_tech_keywords(slots.role or "")
    query_parts.extend(tech_keywords)

    bm25_query = tokenize_for_bm25(" ".join(query_parts))

    if not bm25_query:
        return candidates  # No keywords to match, skip BM25

    bm25 = BM25Okapi(corpus)
    bm25_scores = bm25.get_scores(bm25_query)

    # Normalize BM25 scores to [0, 1]
    max_bm25 = max(bm25_scores) if max(bm25_scores) > 0 else 1.0
    bm25_scores_normalized = [s / max_bm25 for s in bm25_scores]

    # Combine semantic + BM25 scores
    for i, candidate in enumerate(candidates):
        semantic_score = candidate.get("score", 0.0)
        bm25_score = bm25_scores_normalized[i]
        candidate["combined_score"] = 0.6 * semantic_score + 0.4 * bm25_score

    # Sort by combined score descending
    candidates.sort(key=lambda x: x.get("combined_score", 0.0), reverse=True)
    return candidates


def extract_tech_keywords(role: str) -> list[str]:
    """
    Extract technology-specific keywords from a role string.
    These are used to boost BM25 scores for exact tech matches.
    """
    if not role:
        return []

    # Known technologies that have specific SHL catalog tests
    TECH_TERMS = [
        "java", "python", "sql", "javascript", "typescript", "angular",
        "react", "spring", "aws", "docker", "kubernetes", "linux",
        "networking", "rest", "restful", "html", "css", "php", "ruby",
        "scala", "kotlin", "swift", "excel", "word", "powerpoint",
        "salesforce", "sap", "oracle", "hipaa", "medical", "accounting",
        "statistics", "finance", "safety", "rust", "golang", "go",
        "c++", "cpp", "dotnet", ".net", "azure", "gcp",
    ]

    role_lower = role.lower()
    found = []
    for term in TECH_TERMS:
        if term in role_lower:
            found.append(term)
    return found


async def retrieve_candidates(
    slots: SlotState,
    catalog_store,
) -> list[CandidateAssessment]:
    """
    Main retrieval function. Implements the full hybrid pipeline.

    Steps:
    1. Build query from slots
    2. FAISS semantic search (top-40)
    3. Metadata pre-filter (job level, test type)
    4. BM25 re-rank
    5. Inject explicit_additions directly
    6. Return top-20 as CandidateAssessment objects

    Args:
        slots: Populated SlotState from Phase 4
        catalog_store: The CatalogStore singleton

    Returns:
        List of up to MAX_CANDIDATES CandidateAssessment objects,
        ranked by combined semantic + BM25 score.
    """
    # ── Step 1: Build retrieval query ────────────────────────────────────────
    query = build_retrieval_query(slots)

    # ── Step 2: FAISS semantic search ────────────────────────────────────────
    raw_candidates = catalog_store.search_by_text(query, k=FAISS_TOP_K)
    logger.info(f"FAISS retrieved {len(raw_candidates)} candidates")

    # ── Step 3: Metadata pre-filter ──────────────────────────────────────────
    filtered = raw_candidates

    # Job level filter (with synonym expansion built into CatalogStore)
    if slots.seniority:
        filtered = catalog_store.filter_by_job_level(filtered, [slots.seniority])
        logger.info(f"After job level filter ({slots.seniority}): {len(filtered)} candidates")

        # Safety net: if filter is too aggressive, relax it
        if len(filtered) < MIN_CANDIDATES:
            logger.warning(
                f"Job level filter too aggressive ({len(filtered)} < {MIN_CANDIDATES}). "
                f"Relaxing to unfiltered results."
            )
            filtered = raw_candidates

    # Test type filter — only apply if user explicitly requested specific types
    if slots.explicit_test_types:
        type_code_map = {
            "cognitive": ["A"],
            "personality": ["P"],
            "situational-judgment": ["B"],
            "knowledge": ["K"],
            "simulation": ["S"],
            "behavioral": ["B", "C"],
        }
        type_codes = []
        for tt in slots.explicit_test_types:
            type_codes.extend(type_code_map.get(tt, []))

        if type_codes:
            type_filtered = catalog_store.filter_by_test_type(filtered, type_codes)
            logger.info(
                f"After test type filter {type_codes}: {len(type_filtered)} candidates"
            )
            # Only apply if result is not too restrictive
            if len(type_filtered) >= MIN_CANDIDATES:
                filtered = type_filtered
            else:
                logger.warning("Test type filter too restrictive, skipping")

    # ── Step 4: BM25 re-rank ─────────────────────────────────────────────────
    reranked = bm25_rerank(list(filtered), query, slots)
    logger.info(f"After BM25 re-rank: {len(reranked)} candidates")

    # Take top MAX_CANDIDATES
    top_candidates = reranked[:MAX_CANDIDATES]

    # ── Step 5: Inject explicit_additions ────────────────────────────────────
    # These are assessments the user explicitly asked to add.
    # Fetch them directly from catalog and prepend to candidate list.
    # They bypass retrieval scoring — they are ALWAYS included.
    if slots.explicit_additions:
        existing_names_lower = {
            c.get("name", "").lower() for c in top_candidates
        }
        for addition_term in slots.explicit_additions:
            # Try to find in catalog by name
            match = catalog_store.get_by_name(addition_term)
            if match and match["name"].lower() not in existing_names_lower:
                logger.info(f"Injecting explicit addition: {match['name']}")
                # Give it a high score so ranker sees it prominently
                match_with_score = dict(match)
                match_with_score["combined_score"] = 1.0
                top_candidates.insert(0, match_with_score)
                existing_names_lower.add(match["name"].lower())
            elif not match:
                logger.warning(
                    f"Explicit addition '{addition_term}' not found in catalog"
                )

    # ── Step 6: Convert to CandidateAssessment objects ───────────────────────
    result = []
    for c in top_candidates:
        try:
            result.append(CandidateAssessment(
                name=c.get("name", ""),
                url=c.get("url", ""),
                test_type=c.get("test_type", ""),
                test_type_label=c.get("test_type_label", ""),
                description=c.get("description", ""),
                job_levels=c.get("job_levels", []),
                languages=c.get("languages", []),
                duration=c.get("duration", ""),
                remote_testing=c.get("remote_testing", False),
                adaptive_irt=c.get("adaptive_irt", False),
                score=c.get("combined_score", c.get("score", 0.0)),
            ))
        except Exception as e:
            logger.warning(f"Failed to build CandidateAssessment for {c.get('name')}: {e}")

    logger.info(f"Returning {len(result)} candidates to ranker")
    return result
