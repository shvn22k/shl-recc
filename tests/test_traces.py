"""
Trace replay tests for the slot extractor.

Replays representative conversation traces through _extract_slots() and
asserts the agent makes the correct clarify/recommend/compare/close decision.

Run with:
    pytest tests/test_traces.py -v
"""

import pytest
from app.models import Message
from app.catalog import catalog_store
from app.agent import ChatHandler


@pytest.fixture(scope="module", autouse=True)
def load_catalog():
    """Load the catalog once for the entire test module."""
    catalog_store.load()


def msgs(*turns: tuple[str, str]) -> list[Message]:
    """Build a list of Message objects from (role, content) tuples."""
    return [Message(role=r, content=c) for r, c in turns]


# ── Turn 1 clarify / recommend decisions ──────────────────────────────────────

@pytest.mark.asyncio
async def test_c1_vague_clarifies():
    """C1: 'We need a solution for senior leadership' — must clarify."""
    handler = ChatHandler(catalog_store)
    slots = await handler._extract_slots(msgs(
        ("user", "We need a solution for senior leadership."),
    ))
    assert slots.needs_clarification is True
    assert slots.ready_to_retrieve is False
    assert slots.clarification_question is not None
    assert len(slots.clarification_question) > 10


@pytest.mark.asyncio
async def test_c2_rich_query_recommends():
    """C2: Senior Rust engineer with named stack — recommend immediately."""
    handler = ChatHandler(catalog_store)
    slots = await handler._extract_slots(msgs(
        ("user", "I'm hiring a senior Rust engineer for high-performance networking infrastructure. What assessments should I use?"),
    ))
    assert slots.ready_to_retrieve is True
    assert slots.seniority in ("senior-ic", "manager")
    assert slots.role is not None


@pytest.mark.asyncio
async def test_c3_english_accent_clarifies():
    """C3: Contact center, English stated — must ask which accent variant."""
    handler = ChatHandler(catalog_store)
    slots = await handler._extract_slots(msgs(
        ("user", "We're screening 500 entry-level contact centre agents. Inbound calls, customer service focus. What should we use?"),
        ("assistant", "Before I shape the stack — what language are the calls in?"),
        ("user", "English."),
    ))
    assert slots.needs_clarification is True
    assert slots.accent_variant is None


@pytest.mark.asyncio
async def test_c4_graduate_finance_recommends():
    """C4: Graduate financial analysts with explicit test types — recommend turn 1."""
    handler = ChatHandler(catalog_store)
    slots = await handler._extract_slots(msgs(
        ("user", "Hiring graduate financial analysts — final-year students, no work experience. We need numerical reasoning and a finance knowledge test."),
    ))
    assert slots.ready_to_retrieve is True
    assert slots.seniority == "graduate"
    assert "cognitive" in slots.explicit_test_types or slots.role is not None


@pytest.mark.asyncio
async def test_c5_sales_reskilling_recommends():
    """C5: Sales reskilling / audit — recommend immediately."""
    handler = ChatHandler(catalog_store)
    slots = await handler._extract_slots(msgs(
        ("user", "As part of our restructuring and annual talent audit, we need to re-skill our Sales organization. What solutions do you recommend?"),
    ))
    assert slots.ready_to_retrieve is True
    assert slots.purpose in ("reskilling", "audit", "development")


@pytest.mark.asyncio
async def test_c6_safety_recommends():
    """C6: Plant operators, safety-critical — recommend immediately."""
    handler = ChatHandler(catalog_store)
    slots = await handler._extract_slots(msgs(
        ("user", "We're hiring plant operators for a chemical facility. Safety is absolute top priority — reliability, procedure compliance, never cutting corners."),
    ))
    assert slots.ready_to_retrieve is True
    assert slots.industry in ("manufacturing", "chemical", "oil and gas", None)


@pytest.mark.asyncio
async def test_c7_language_constraint_surfaces():
    """C7: Bilingual healthcare admin, Spanish — slots capture language."""
    handler = ChatHandler(catalog_store)
    slots = await handler._extract_slots(msgs(
        ("user", "We're hiring bilingual healthcare admin staff in South Texas — they handle patient records and need to be assessed in Spanish. HIPAA compliance is critical."),
    ))
    assert slots.language is not None
    assert "spanish" in slots.language.lower()


@pytest.mark.asyncio
async def test_c8_time_constraint_captured():
    """C8: 'Quickly screen admin assistants' — time_constraint=short."""
    handler = ChatHandler(catalog_store)
    slots = await handler._extract_slots(msgs(
        ("user", "I need to quickly screen admin assistants for Excel and Word daily."),
    ))
    assert slots.ready_to_retrieve is True
    assert slots.time_constraint == "short"


@pytest.mark.asyncio
async def test_c9_complex_jd_clarifies():
    """C9: JD with 7 technologies — must clarify backend vs frontend ownership."""
    handler = ChatHandler(catalog_store)
    slots = await handler._extract_slots(msgs(
        ("user", 'Here\'s the JD for an engineer we need to fill. Can you recommend an assessment battery?\n\n"Senior Full-Stack Engineer — 5+ years across Core Java, Spring, REST API design, Angular, SQL/relational databases, AWS deployment, and Docker."'),
    ))
    assert slots.needs_clarification is True
    assert slots.clarification_question is not None


@pytest.mark.asyncio
async def test_c10_graduate_trainee_recommends():
    """C10: Graduate management trainee, explicit three test types — recommend turn 1."""
    handler = ChatHandler(catalog_store)
    slots = await handler._extract_slots(msgs(
        ("user", "We run a graduate management trainee scheme. We need a full battery — cognitive, personality, and situational judgement. All recent graduates."),
    ))
    assert slots.ready_to_retrieve is True
    assert slots.seniority == "graduate"


# ── Refinement behaviour ──────────────────────────────────────────────────────

@pytest.mark.asyncio
async def test_explicit_add_captured():
    """Explicit add instruction must populate explicit_additions."""
    handler = ChatHandler(catalog_store)
    slots = await handler._extract_slots(msgs(
        ("user", "I'm hiring a senior Java developer."),
        ("assistant", "Here are my recommendations: [Java, Spring, SQL, Verify G+, OPQ32r]"),
        ("user", "Add AWS and Docker. Drop REST."),
    ))
    assert len(slots.explicit_additions) > 0
    assert any("aws" in a.lower() or "docker" in a.lower() for a in slots.explicit_additions)


@pytest.mark.asyncio
async def test_explicit_drop_captured():
    """Explicit drop instruction must populate explicit_drops."""
    handler = ChatHandler(catalog_store)
    slots = await handler._extract_slots(msgs(
        ("user", "I'm hiring a senior Java developer."),
        ("assistant", "Here are my recommendations: [Java, Spring, SQL, Verify G+, OPQ32r]"),
        ("user", "Drop REST."),
    ))
    assert len(slots.explicit_drops) > 0
    assert any("rest" in d.lower() for d in slots.explicit_drops)


# ── Special turn detection ────────────────────────────────────────────────────

@pytest.mark.asyncio
async def test_comparison_turn_detected():
    """Comparison question must set is_comparison_turn=True."""
    handler = ChatHandler(catalog_store)
    slots = await handler._extract_slots(msgs(
        ("user", "We're hiring plant operators for a chemical facility."),
        ("assistant", "Here are my recommendations: [DSI, Safety 8.0, WHS]"),
        ("user", "What's the difference between the DSI and the Safety & Dependability 8.0?"),
    ))
    assert slots.is_comparison_turn is True
    assert slots.conversation_phase == "comparing"


@pytest.mark.asyncio
async def test_closing_turn_detected():
    """Closing confirmation must set end_of_conversation=True."""
    handler = ChatHandler(catalog_store)
    slots = await handler._extract_slots(msgs(
        ("user", "We're hiring plant operators for a chemical facility."),
        ("assistant", "Here are my recommendations: [Safety 8.0, WHS]"),
        ("user", "We're industrial. The 8.0 bundle is the right fit. Confirmed."),
    ))
    assert slots.end_of_conversation is True
    assert slots.conversation_phase == "closing"
