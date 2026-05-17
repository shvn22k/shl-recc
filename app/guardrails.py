"""
Guardrails for the SHL Assessment Recommender.

Two responsibilities:

1. Input classification — detects injection attempts, legal questions, and
   off-topic messages before they reach the LLM, so the agent can refuse
   gracefully with a canned response.

2. URL whitelist enforcement — strips any recommendation whose URL is not in
   the scraped catalog before the response leaves the API. This is the hard
   guard against hallucinated URLs that would fail the evaluator's hard evals.
"""

import logging
import re

logger = logging.getLogger(__name__)


# ── Prompt injection patterns ─────────────────────────────────────────────────
# Catches common injection attempts at the regex layer. Not exhaustive —
# the LLM system prompt also instructs the model to ignore such attempts.

_INJECTION_PATTERNS = [
    r"ignore\s+(all\s+)?(previous|prior|above|your)\s+instructions",
    r"forget\s+(everything|all|your)\s+(you|above|previous)",
    r"you\s+are\s+now\s+a?\s*(new|different|another|unrestricted)",
    r"(pretend|act|behave)\s+(like|as\s+if|as though)\s+you",
    r"(disregard|override|bypass|ignore)\s+(your\s+)?(system|prompt|guidelines|rules)",
    r"jailbreak",
    r"dan\s+mode",
    r"developer\s+mode",
    r"prompt\s+inject",
]

# ── Off-topic keywords ────────────────────────────────────────────────────────
# Topics explicitly out of scope per the assessment spec. Conservative —
# only triggers on unambiguous non-assessment signals. When in doubt, the
# LLM handles it gracefully rather than over-refusing.

_OFF_TOPIC_SIGNALS = [
    "salary", "compensation", "pay", "wages",
    "interview questions", "how to interview",
    "resume", "cv", "cover letter",
    "background check", "reference check",
    "onboarding", "training program",
    "performance review", "kpi",
    "layoff", "redundancy", "terminate",
    "stock options", "equity",
    "visa", "work permit", "immigration",
]

# ── Legal / compliance patterns ───────────────────────────────────────────────

_LEGAL_PATTERNS = [
    r"legally\s+required",
    r"legal\s+requirement",
    r"comply\s+with",
    r"hipaa\s+require",
    r"gdpr\s+require",
    r"eeoc",
    r"ada\s+compliance",
    r"discriminat",
    r"protected\s+class",
    r"lawsuit",
    r"regulation\s+require",
    r"must\s+we",
    r"are\s+we\s+required",
    r"satisfy.*requirement",
    r"fulfil.*requirement",
]

# Pre-compile everything once at import time
_compiled_injection = [re.compile(p, re.IGNORECASE) for p in _INJECTION_PATTERNS]
_compiled_legal = [re.compile(p, re.IGNORECASE) for p in _LEGAL_PATTERNS]


# ── Input classifiers ─────────────────────────────────────────────────────────

def is_injection_attempt(text: str) -> bool:
    """Return True if the text contains prompt injection signals."""
    for pattern in _compiled_injection:
        if pattern.search(text):
            logger.warning(f"Injection attempt detected: '{text[:120]}'")
            return True
    return False


def is_legal_question(text: str) -> bool:
    """Return True if the text is asking a legal or compliance question."""
    for pattern in _compiled_legal:
        if pattern.search(text):
            logger.info(f"Legal question detected: '{text[:120]}'")
            return True
    return False


def is_off_topic(text: str) -> bool:
    """
    Return True if the text is clearly off-topic (not about SHL assessments).
    Conservative — only triggers on explicit non-assessment keywords.
    """
    text_lower = text.lower()
    matched = [s for s in _OFF_TOPIC_SIGNALS if s in text_lower]
    if matched:
        logger.info(f"Off-topic signals detected: {matched}")
        return True
    return False


# ── URL whitelist enforcer ────────────────────────────────────────────────────

def enforce_url_whitelist(recommendations: list, catalog_store) -> list:
    """
    Strip any recommendation whose URL is not in the catalog whitelist.

    Applied as a post-processing step on every response before it leaves
    the API — regardless of where the recommendation came from. Protects
    against LLM hallucinations and any other source of invalid URLs.

    Args:
        recommendations: list of Recommendation objects or raw dicts.
        catalog_store:   the CatalogStore singleton (has is_valid_url()).

    Returns:
        Filtered list containing only catalog-valid URLs.
    """
    if not recommendations:
        return recommendations

    valid = []
    for rec in recommendations:
        url = rec.url if hasattr(rec, "url") else rec.get("url", "")
        if catalog_store.is_valid_url(url):
            valid.append(rec)
        else:
            name = rec.name if hasattr(rec, "name") else rec.get("name", "unknown")
            logger.error(
                f"WHITELIST VIOLATION — stripping recommendation: "
                f"name='{name}' url='{url}'"
            )

    stripped = len(recommendations) - len(valid)
    if stripped:
        logger.warning(f"Whitelist enforcement: {stripped} recommendation(s) stripped")

    return valid


# ── Canned refusal messages ───────────────────────────────────────────────────

def get_injection_refusal() -> str:
    return (
        "I'm here to help you find the right SHL assessments for your hiring needs. "
        "I can't follow instructions that ask me to change my role or ignore my guidelines. "
        "What role or position are you looking to assess?"
    )


def get_legal_refusal() -> str:
    return (
        "That's a legal or compliance question that falls outside what I can advise on. "
        "I can help you select SHL assessments based on role requirements, but whether "
        "a specific assessment satisfies a regulatory obligation is a question for your "
        "legal or compliance team. "
        "Is there anything else I can help you with regarding assessment selection?"
    )


def get_off_topic_refusal() -> str:
    return (
        "I'm focused on helping you choose the right SHL assessments for your roles. "
        "I'm not able to help with that topic, but I'd be happy to recommend assessments "
        "if you share the role or position you're hiring for."
    )
