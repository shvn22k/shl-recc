"""
Hybrid retrieval for the SHL Assessment Recommender.

Implements the retrieval stage of CSG-RAG:

  1. Query construction  — builds a semantic query from structured slot state
  2. FAISS search        — top-40 candidates by cosine similarity
  3. Metadata filter     — job level filter for graduate/entry-level roles only
  4. BM25 re-rank        — boosts exact keyword matches (critical for tech roles)
  5. Explicit injection  — user-requested additions bypass retrieval scoring

The job level filter is intentionally restricted to graduate and entry-level
seniorities. Senior and above roles have sparse job level metadata in the SHL
catalog, and filtering aggressively eliminates valid technical knowledge tests.
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

    Strategy differs by role type:
    - Tech/knowledge roles: lead with domain keywords + "knowledge test"
      so FAISS surfaces coding/technical assessments first
    - Non-tech roles: lead with role + seniority framing
    """
    parts = []

    # Detect if this is a tech/knowledge role — check role AND explicit_additions
    # so "screen for Excel and Word" surfaces the right assessments even when the
    # LLM extracts role="admin assistant" without the tool names embedded.
    tech_keywords = extract_tech_keywords(slots.role or "")
    for addition in slots.explicit_additions:
        tech_keywords.extend(extract_tech_keywords(addition))
    tech_keywords = list(dict.fromkeys(tech_keywords))  # deduplicate, preserve order
    is_tech_role = len(tech_keywords) > 0

    if is_tech_role:
        # For tech roles: domain keywords first, then "knowledge skills test"
        # This anchors the embedding in the technical assessment space
        parts.extend(tech_keywords)
        if slots.role:
            parts.append(slots.role)
        parts.append("knowledge skills technical assessment programming")

        # Add explicit test type preferences
        test_type_phrases = {
            "cognitive": "cognitive ability reasoning",
            "personality": "personality behavior OPQ",
            "situational-judgment": "situational judgment SJT",
            "knowledge": "knowledge skills domain specific test",
            "simulation": "simulation work sample",
            "behavioral": "behavioral competency",
        }
        for tt in slots.explicit_test_types:
            if tt in test_type_phrases:
                parts.append(test_type_phrases[tt])

        # Seniority appended after domain (less influential position)
        if slots.seniority:
            seniority_phrases = {
                "graduate": "graduate entry level",
                "entry-level": "entry level junior",
                "mid-professional": "mid level professional",
                "senior-ic": "senior advanced expert",
                "manager": "manager lead",
                "director": "director senior",
                "executive": "executive leadership",
            }
            parts.append(seniority_phrases.get(slots.seniority, slots.seniority))

    else:
        # For non-tech roles: role + seniority first, then purpose/domain
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

    # NOTE: language is intentionally excluded from the FAISS query.
    # Including it (e.g. "Spanish language assessment") overweights language-
    # proficiency tests and crowds out the domain-specific tools that are the
    # primary signal for most roles (e.g. HIPAA, Medical Terminology for a
    # bilingual healthcare-admin hire).  The ranker sees slots.language in its
    # prompt and can select language-appropriate catalog variants from the
    # correctly-surfaced domain candidates.

    # Time constraint
    if slots.time_constraint == "short":
        parts.append("short quick brief assessment under 20 minutes")

    # Volume
    if slots.volume == "high":
        parts.append("high volume screening large scale")

    query = " ".join(parts)
    logger.debug(f"Retrieval query (tech_role={is_tech_role}): '{query}'")
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
    catalog_store: "CatalogStore",
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
        slots: Populated SlotState from the slot extractor
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

    # Job level filter — only apply for graduate and entry-level
    # For senior+ roles, most technical knowledge tests have no job level set,
    # so filtering by level eliminates valid candidates. Ranker handles seniority
    # framing via prompt instructions instead.
    LEVEL_FILTER_APPLICABLE = {"graduate", "entry-level"}
    if slots.seniority and slots.seniority in LEVEL_FILTER_APPLICABLE:
        filtered = catalog_store.filter_by_job_level(filtered, [slots.seniority])
        logger.info(
            f"After job level filter ({slots.seniority}): {len(filtered)} candidates"
        )
        # Safety net: if filter is too aggressive, relax it
        if len(filtered) < MIN_CANDIDATES:
            logger.warning(
                f"Job level filter too aggressive ({len(filtered)} < {MIN_CANDIDATES}). "
                f"Relaxing to unfiltered results."
            )
            filtered = raw_candidates
    else:
        logger.info(
            f"Skipping job level filter for seniority='{slots.seniority}' "
            f"(technical knowledge tests have sparse job level metadata)"
        )

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

    # ── Step 4b: Re-inject current shortlist items (before BM25 cutoff) ─────────
    # On refinement turns the agreed shortlist must survive a fresh FAISS search.
    # A query biased toward newly-requested items (e.g. "AWS Docker") can push
    # agreed-upon items (e.g. "Core Java", "Spring", "SQL") out of the top-40.
    # We inject them directly into the filtered pool before BM25 so they
    # participate in re-ranking and the top-20 cutoff.
    if slots.current_shortlist_urls:
        existing_urls_in_pool = {c.get("url", "") for c in filtered}
        injected = 0
        for url in slots.current_shortlist_urls:
            if url in existing_urls_in_pool:
                continue
            item = catalog_store.get_by_url(url)
            if item:
                shortlist_item = dict(item)
                shortlist_item["score"] = 0.85  # high-but-not-max; BM25 will further sort
                filtered.append(shortlist_item)
                existing_urls_in_pool.add(url)
                injected += 1
        if injected:
            logger.info(f"Shortlist continuity: injected {injected} items into BM25 pool")

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
