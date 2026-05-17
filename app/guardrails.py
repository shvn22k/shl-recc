"""
Input validation and output enforcement for the SHL Assessment Recommender.

Two distinct responsibilities:

Pre-request checks (run before any LLM call):
  - Injection detection  — regex patterns for direct and indirect prompt injection
  - Legal detection      — compliance/discrimination/regulatory questions
  - Off-topic detection  — clearly non-assessment queries
  - Gibberish detection  — messages with no meaningful content

Post-response enforcement (run before every API response):
  - URL whitelist        — strips any recommendation URL not in the scraped catalog

Guardrails are conservative by design. When a message is ambiguous,
it passes through to the LLM rather than triggering a false-positive refusal.
"""

import logging
import re
from typing import Any

logger = logging.getLogger(__name__)


# ── Prompt Injection Patterns ─────────────────────────────────────────────────
# Catches direct and indirect injection attempts.
# Conservative — only fire on clear manipulation signals.

INJECTION_PATTERNS = [
    # Direct instruction override
    r"ignore\s+(all\s+)?(previous|prior|above|your)\s+instructions",
    r"forget\s+(everything|all|your)\s*(you|above|previous|prior)?",
    r"disregard\s+(your\s+)?(system|prompt|guidelines|rules|instructions)",
    r"override\s+(your\s+)?(system|prompt|guidelines|rules|instructions)",
    r"bypass\s+(your\s+)?(system|prompt|guidelines|rules|restrictions)",

    # Identity replacement
    r"you\s+are\s+now\s+a?\s*(new|different|another|unrestricted|free)",
    r"(pretend|act|behave)\s+(like|as\s+if|as\s+though)\s+you('re|\s+are)",
    r"from\s+now\s+on\s+(you\s+are|your\s+role|act\s+as)",
    r"your\s+new\s+(instructions|role|persona|purpose)\s+(are|is)",
    r"disregard\s+your\s+role\s+as",

    # System prompt extraction
    r"(output|print|show|reveal|display|repeat)\s+(your\s+)?(system\s+prompt|instructions|guidelines)",
    r"what\s+(are\s+your|is\s+your)\s+(system\s+prompt|instructions|guidelines|rules)",

    # Known jailbreak terminology
    r"jailbreak",
    r"\bdan\s+mode\b",
    r"developer\s+mode",
    r"prompt\s+inject",
    r"(ignore|skip)\s+all\s+(safety|content)\s+(filters|guidelines|checks)",
]

COMPILED_INJECTION = [re.compile(p, re.IGNORECASE) for p in INJECTION_PATTERNS]


# ── Legal / Compliance Patterns ───────────────────────────────────────────────
# Catches questions about regulatory obligations, legal validity,
# discrimination risk, and compliance requirements.
# Conservative — "compliance" alone does not trigger (too common in HR context).

LEGAL_PATTERNS = [
    # Regulatory requirements
    r"(legally|legally\s+required|required\s+by\s+law)\s+(to\s+)?(test|assess|use)",
    r"are\s+we\s+(legally\s+)?(required|obligated|mandated)\s+(under|by|to)",
    r"(does\s+this|will\s+this)\s+(satisfy|fulfil|meet|comply\s+with)\s+.{0,30}(requirement|regulation|law|standard)",
    r"(hipaa|gdpr|eeoc|ada|ccpa|fcra)\s+(require|compliance|compliant|regulated)",
    r"(compliant|compliance)\s+with\s+(hipaa|gdpr|eeoc|ada|law|regulation)",

    # Legal validity and risk
    r"(hold\s+up|admissible|valid|defensible)\s+in\s+(court|litigation|tribunal)",
    r"(lawsuit|legal\s+action|litigation)\s+risk",
    r"(legal|liability)\s+(exposure|risk|implications?)",

    # Discrimination and bias risk
    r"(does\s+this|will\s+this|could\s+this)\s+(discriminat|creat|introduc|result\s+in)\s+.{0,20}(bias|discriminat)",
    r"(protected\s+class|protected\s+characteristic|disparate\s+impact)",
    r"(race|gender|age|disability|religion|national\s+origin).{0,30}(bias|discriminat|test)",

    # EU/regional compliance
    r"(use\s+this|valid|allowed|permitted)\s+(in|for)\s+(the\s+)?(eu|europe|uk\s+gdpr)",
    r"(eu|gdpr|uk)\s+(data\s+protection|privacy\s+law|regulation)",
]

COMPILED_LEGAL = [re.compile(p, re.IGNORECASE) for p in LEGAL_PATTERNS]


# ── Off-Topic Signals ─────────────────────────────────────────────────────────
# Only fire on clearly non-assessment topics.
# Do NOT include terms common in legitimate HR/assessment context.

OFF_TOPIC_KEYWORDS = [
    # Compensation and benefits
    "salary range", "compensation package", "pay scale", "wage",
    "stock options", "equity grant", "bonus structure",

    # Non-assessment HR processes
    "background check", "reference check", "credit check",
    "drug test", "drug screening",
    "cover letter", "resume review", "cv screening",
    "job posting", "job advertisement", "write a job description",
    "offer letter", "employment contract",

    # Post-hire processes (clearly not assessment selection)
    "onboarding process", "new hire orientation",
    "termination process", "layoff", "redundancy process",
    "performance improvement plan", "pip process",

    # Completely unrelated
    "stock market", "cryptocurrency", "recipe", "weather",
    "sports score", "movie recommendation",
]


# ── Gibberish Detection ───────────────────────────────────────────────────────
# Minimum signal required to attempt slot extraction.
# Very conservative — only catch clear non-text inputs.

MIN_MEANINGFUL_WORDS = 2
GIBBERISH_PATTERN = re.compile(r"^[\W\d_\s]+$")  # Only non-word chars


def is_injection_attempt(text: str) -> bool:
    """Returns True if text contains prompt injection signals."""
    for pattern in COMPILED_INJECTION:
        if pattern.search(text):
            logger.warning(f"Injection detected: '{text[:80]}'")
            return True
    return False


def is_legal_question(text: str) -> bool:
    """Returns True if text asks a legal/compliance/discrimination question."""
    for pattern in COMPILED_LEGAL:
        if pattern.search(text):
            logger.info(f"Legal question detected: '{text[:80]}'")
            return True
    return False


def is_off_topic(text: str) -> bool:
    """
    Returns True only if text is clearly about a non-assessment topic.
    Conservative — when in doubt, returns False and lets LLM handle it.
    """
    text_lower = text.lower()
    matched = [kw for kw in OFF_TOPIC_KEYWORDS if kw in text_lower]
    if matched:
        # Require at least one match AND no assessment-related terms
        # to avoid false positives on hybrid queries
        assessment_signals = [
            "assess", "test", "evaluation", "hiring", "recruit",
            "candidate", "shl", "skill", "aptitude", "personality"
        ]
        has_assessment_signal = any(s in text_lower for s in assessment_signals)
        if not has_assessment_signal:
            logger.info(f"Off-topic detected: {matched}")
            return True
    return False


def is_gibberish(text: str) -> bool:
    """
    Returns True if the message has no meaningful content.
    Only catches clear non-text: symbols only, single char, pure numbers.
    """
    stripped = text.strip()
    if len(stripped) < 2:
        return True
    if GIBBERISH_PATTERN.match(stripped):
        return True
    # Count real words (alphabetic sequences of 2+ chars)
    words = re.findall(r"[a-zA-Z]{2,}", stripped)
    if len(words) < MIN_MEANINGFUL_WORDS:
        return True
    return False


def enforce_url_whitelist(recommendations: list, catalog_store: Any) -> list:
    """
    Remove any recommendation whose URL is not in the catalog whitelist.
    Applied to every response before it leaves the API.
    Protects against LLM URL hallucination.
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
                f"WHITELIST VIOLATION — stripping: name='{name}' url='{url}'"
            )

    stripped = len(recommendations) - len(valid)
    if stripped > 0:
        logger.warning(f"Whitelist enforcement stripped {stripped} recommendation(s)")

    return valid


# ── Refusal Message Factory ───────────────────────────────────────────────────

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
        "a specific assessment satisfies a regulatory obligation, creates legal risk, "
        "or meets discrimination standards is a question for your legal or compliance team. "
        "Is there anything else I can help you with regarding assessment selection?"
    )


def get_off_topic_refusal() -> str:
    return (
        "I'm focused on helping you choose the right SHL assessments for your roles. "
        "I'm not able to help with that topic, but I'd be happy to recommend assessments "
        "if you share the role or position you're hiring for."
    )


def get_gibberish_redirect() -> str:
    return (
        "I didn't quite catch that. "
        "I'm here to help you find the right SHL assessments — "
        "could you tell me the role or position you're looking to hire for?"
    )
