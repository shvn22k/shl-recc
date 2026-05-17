"""
Agent logic for the SHL Assessment Recommender.

ChatHandler orchestrates the full CSG-RAG pipeline for each /chat request:

  1. Guardrail pre-check  — injection, legal, off-topic (this phase)
  2. Turn cap check       — max 8 user turns (this phase)
  3. Slot extraction      — LLM Call 1, extracts structured intent (Phase 4)
  4. Metadata pre-filter  — job level / test type filter on candidates (Phase 5)
  5. Hybrid retrieval     — FAISS semantic + BM25 keyword re-rank (Phase 5)
  6. LLM ranker           — LLM Call 2, selects final 1-10 assessments (Phase 6)
  7. URL whitelist        — post-processing strip of any invalid URLs (this phase)

Phases 4-6 replace the stub methods below with real implementations.
Everything else here is final and production-ready.
"""

import logging
import os

from app.models import AgentDecision, Message, Recommendation, SlotState
from app.retriever import retrieve_candidates as _do_retrieve
from app.prompts import build_slot_extractor_prompt, build_ranker_prompt, build_comparison_prompt
from app.llm_client import call_llm_json, LLMError
from app.guardrails import (
    enforce_url_whitelist,
    get_injection_refusal,
    get_legal_refusal,
    get_off_topic_refusal,
    is_injection_attempt,
    is_legal_question,
    is_off_topic,
)

logger = logging.getLogger(__name__)

MAX_TURNS = int(os.getenv("MAX_TURNS", "8"))


class ChatHandler:
    """
    Stateless handler for a single /chat request.

    Instantiated fresh per request — no state is shared between calls.
    Takes the full message history and runs the CSG-RAG pipeline,
    returning an AgentDecision that the endpoint serialises into ChatResponse.

    Usage:
        handler = ChatHandler(catalog_store)
        decision = await handler.handle(request.messages)
    """

    def __init__(self, catalog_store):
        self.catalog_store = catalog_store

    async def handle(self, messages: list[Message]) -> AgentDecision:
        """
        Main entry point. Runs the full pipeline for one conversation turn.
        Always returns an AgentDecision — never raises.
        """

        # ── 1. Turn cap ───────────────────────────────────────────────────────
        # Count user messages only; assistant turns don't count toward the cap.
        user_turns = sum(1 for m in messages if m.role == "user")
        if user_turns > MAX_TURNS:
            logger.warning(f"Turn cap exceeded: {user_turns} user turns")
            return AgentDecision(
                reply=(
                    "We've reached the end of our session. Based on everything "
                    "you've shared, I recommend reviewing the assessments we "
                    "discussed. Feel free to start a new conversation to explore "
                    "further options."
                ),
                recommendations=[],
                end_of_conversation=True,
            )

        # ── 2. Guardrail pre-checks on the latest user message ────────────────
        latest = self._get_latest_user_message(messages)

        if is_injection_attempt(latest):
            return AgentDecision(
                reply=get_injection_refusal(),
                recommendations=[],
                end_of_conversation=False,
            )

        if is_legal_question(latest):
            return AgentDecision(
                reply=get_legal_refusal(),
                recommendations=[],
                end_of_conversation=False,
            )

        if is_off_topic(latest):
            return AgentDecision(
                reply=get_off_topic_refusal(),
                recommendations=[],
                end_of_conversation=False,
            )

        # ── 3. Slot extraction (Phase 4) ──────────────────────────────────────
        slots = await self._extract_slots(messages)

        # ── 4. Clarification turn — return question, skip retrieval ───────────
        if slots.needs_clarification and not slots.ready_to_retrieve:
            return AgentDecision(
                reply=slots.clarification_question or (
                    "Could you tell me more about the role you're hiring for?"
                ),
                recommendations=[],
                end_of_conversation=False,
            )

        # ── 5. Comparison turn — explain differences, no new recommendations ──
        if slots.is_comparison_turn:
            reply = await self._handle_comparison(messages, slots)
            return AgentDecision(
                reply=reply,
                recommendations=[],
                end_of_conversation=False,
            )

        # ── 6. Closing turn — user confirmed they're done ─────────────────────
        if slots.end_of_conversation:
            current_recs = await self._get_current_recommendations(slots)
            safe_recs = enforce_url_whitelist(current_recs, self.catalog_store)
            return AgentDecision(
                reply=self._build_closing_reply(slots),
                recommendations=safe_recs,
                end_of_conversation=True,
            )

        # ── 7. Retrieval + ranking (Phases 5 & 6) ────────────────────────────
        candidates = await self._retrieve_candidates(slots)
        decision = await self._rank_and_respond(messages, slots, candidates)

        # ── 8. Whitelist enforcement — always the last step ───────────────────
        decision.recommendations = enforce_url_whitelist(
            decision.recommendations, self.catalog_store
        )

        return decision

    # ── Helpers ───────────────────────────────────────────────────────────────

    def _get_latest_user_message(self, messages: list[Message]) -> str:
        """Return the content of the most recent user message, or empty string."""
        for msg in reversed(messages):
            if msg.role == "user":
                return msg.content
        return ""

    def _build_closing_reply(self, slots: SlotState) -> str:
        """Build a warm closing confirmation message."""
        role_phrase = f" for the {slots.role} role" if slots.role else ""
        return (
            f"Great — here's your final assessment shortlist{role_phrase}. "
            "Good luck with your hiring process. "
            "Feel free to come back if you need to reassess or explore other roles."
        )

    # ── Stub methods — replaced in Phases 4, 5, 6 ────────────────────────────

    async def _extract_slots(self, messages: list[Message]) -> SlotState:
        """
        LLM Call 1 — Slot Extractor.

        Reads the full conversation history and extracts structured intent
        into a SlotState. Uses Gemini 2.5 Flash with JSON mode, falls back
        to OpenAI GPT-4o-mini automatically.

        The SlotState drives all downstream decisions:
          needs_clarification  → return question, skip retrieval
          ready_to_retrieve    → proceed to Phase 5 retrieval
          is_comparison_turn   → return explanation, no new retrieval
          end_of_conversation  → repeat shortlist, set end=true
          explicit_additions/drops → honoured absolutely in Phase 6
        """
        system_prompt, user_prompt = build_slot_extractor_prompt(messages)

        try:
            raw = await call_llm_json(
                system_prompt=system_prompt,
                user_prompt=user_prompt,
                temperature=0.1,
            )
            logger.debug(f"Slot extractor raw output: {raw}")
        except LLMError as e:
            logger.error(f"Slot extractor LLM call failed: {e}")
            return SlotState(
                needs_clarification=True,
                clarification_question=(
                    "I'm having trouble processing your request right now. "
                    "Could you describe the role you're hiring for?"
                ),
            )
        except ValueError as e:
            logger.error(f"Slot extractor JSON parse failed: {e}")
            return SlotState(
                needs_clarification=True,
                clarification_question=(
                    "Could you tell me more about the role and level "
                    "you're looking to assess?"
                ),
            )

        try:
            sanitised = _sanitise_slot_output(raw)
            return SlotState(**sanitised)
        except Exception as e:
            logger.error(f"SlotState construction failed: {e}\nRaw: {raw}")
            return SlotState(
                needs_clarification=True,
                clarification_question=(
                    "What role are you hiring for, and what level of seniority?"
                ),
            )

    async def _retrieve_candidates(self, slots: SlotState) -> list:
        """
        Phase 5 — Hybrid retrieval.
        Delegates to app/retriever.py which implements:
        FAISS semantic search → metadata filter → BM25 re-rank → explicit injection
        """
        return await _do_retrieve(slots, self.catalog_store)

    async def _rank_and_respond(
        self,
        messages: list[Message],
        slots: SlotState,
        candidates: list,
    ) -> AgentDecision:
        """
        Phase 6 — LLM Ranker (Call 2).

        Selects final 1-10 assessments from candidates using chain-of-thought
        reasoning. Applies explicit_drops as a post-processing step.
        """
        from app.llm_client import call_llm_json, call_llm, LLMError
        from app.models import Recommendation

        # Edge case: no candidates retrieved
        if not candidates:
            return AgentDecision(
                reply=(
                    "I wasn't able to find specific assessments matching those "
                    "criteria in the SHL catalog. Could you tell me more about "
                    "the role or the skills you need to assess?"
                ),
                recommendations=[],
                end_of_conversation=False,
            )

        system_prompt, user_prompt = build_ranker_prompt(messages, slots, candidates)

        try:
            raw = await call_llm_json(
                system_prompt=system_prompt,
                user_prompt=user_prompt,
                temperature=0.2,  # Slightly higher than extractor — allows nuanced selection
            )
        except LLMError as e:
            logger.error(f"Ranker LLM call failed: {e}")
            # Fallback: return top-5 candidates directly without LLM selection
            fallback_recs = []
            for c in candidates[:5]:
                fallback_recs.append(Recommendation(
                    name=c.name,
                    url=c.url,
                    test_type=c.test_type,
                ))
            return AgentDecision(
                reply=(
                    "Here are the most relevant assessments I found for your needs."
                ),
                recommendations=fallback_recs,
                end_of_conversation=False,
            )

        # Parse ranker output
        try:
            selected_raw = raw.get("selected_assessments", [])
            reply = raw.get("reply", "").strip()
            end_conv = bool(raw.get("end_of_conversation", False))

            if not reply:
                reply = "Here are the assessments I recommend for this role."

            # Build Recommendation objects from ranker output
            # Validate each against candidates to prevent URL hallucination
            candidate_url_map = {
                (c.name if hasattr(c, "name") else c.get("name", "")).lower(): c
                for c in candidates
            }
            candidate_urls = {
                c.url if hasattr(c, "url") else c.get("url", "")
                for c in candidates
            }

            recommendations = []
            seen_urls = set()

            for item in selected_raw:
                name = item.get("name", "").strip()
                url = item.get("url", "").strip()
                test_type = item.get("test_type", "").strip()

                # Skip duplicates
                if url in seen_urls:
                    continue

                # Validate URL is from candidates or catalog whitelist
                if url not in candidate_urls:
                    # Try to find the correct URL from catalog by name
                    match = self.catalog_store.get_by_name(name)
                    if match:
                        url = match["url"]
                        test_type = match["test_type"]
                        logger.warning(
                            f"Ranker returned wrong URL for '{name}', "
                            f"corrected to catalog URL"
                        )
                    else:
                        logger.error(
                            f"Ranker hallucinated assessment not in candidates: "
                            f"name='{name}' url='{url}' — skipping"
                        )
                        continue

                # Apply explicit_drops — remove if name matches any drop term
                if self._should_drop(name, slots.explicit_drops):
                    logger.info(f"Dropping '{name}' per explicit_drops instruction")
                    continue

                if not name or not url or not test_type:
                    continue

                recommendations.append(Recommendation(
                    name=name,
                    url=url,
                    test_type=test_type,
                ))
                seen_urls.add(url)

                if len(recommendations) >= 10:
                    break

        except Exception as e:
            logger.error(f"Failed to parse ranker output: {e}\nRaw: {raw}")
            recommendations = []
            reply = "I found relevant assessments but encountered an issue formatting them. Please try again."
            end_conv = False

        return AgentDecision(
            reply=reply,
            recommendations=recommendations,
            end_of_conversation=end_conv,
        )

    async def _handle_comparison(
        self, messages: list[Message], slots: SlotState
    ) -> str:
        """
        Phase 6 — Comparison handler.

        Fetches the relevant assessments from the catalog by URL
        (from current_shortlist_urls) and answers the comparison
        question grounded in catalog data only.

        Returns plain text reply. recommendations=[] on comparison turns.
        """
        from app.llm_client import call_llm, LLMError

        # Gather details for assessments currently in the shortlist
        assessment_details_parts = []
        urls_to_fetch = slots.current_shortlist_urls or []

        # If no shortlist URLs, try fetching the two most recently mentioned names
        # from the conversation (the LLM will figure it out from context)
        for url in urls_to_fetch[:6]:  # Cap at 6 to keep prompt manageable
            item = self.catalog_store.get_by_url(url)
            if item:
                desc = item.get("description", "")[:400]
                assessment_details_parts.append(
                    f"Assessment: {item['name']}\n"
                    f"Type: {item['test_type']} ({item.get('test_type_label', '')})\n"
                    f"Duration: {item.get('duration', 'Not specified')}\n"
                    f"Job Levels: {', '.join(item.get('job_levels', [])) or 'General'}\n"
                    f"Description: {desc}"
                )

        assessment_details = "\n\n---\n\n".join(assessment_details_parts)
        if not assessment_details:
            assessment_details = "No specific assessment details available for comparison."

        system_prompt, user_prompt = build_comparison_prompt(messages, assessment_details)

        try:
            reply = await call_llm(
                system_prompt=system_prompt,
                user_prompt=user_prompt,
                json_mode=False,  # Plain text for comparison answers
                temperature=0.2,
            )
            return reply.strip()
        except LLMError as e:
            logger.error(f"Comparison LLM call failed: {e}")
            return (
                "I can help compare those assessments. "
                "Both are available in the SHL catalog — "
                "could you clarify what specific dimension you'd like to compare?"
            )

    @staticmethod
    def _should_drop(name: str, explicit_drops: list[str]) -> bool:
        """
        Check if an assessment name matches any explicit drop instruction.
        Case-insensitive, partial match allowed.
        """
        if not explicit_drops:
            return False
        name_lower = name.lower()
        for drop_term in explicit_drops:
            drop_lower = drop_term.lower()
            if drop_lower in name_lower or name_lower in drop_lower:
                return True
        return False

    async def _get_current_recommendations(self, slots: SlotState) -> list[Recommendation]:
        """
        Reconstruct the current shortlist from URLs stored in slot state.
        Used on closing turns to echo back the agreed list.
        """
        recs = []
        for url in slots.current_shortlist_urls:
            item = self.catalog_store.get_by_url(url)
            if item:
                recs.append(Recommendation(
                    name=item["name"],
                    url=item["url"],
                    test_type=item["test_type"],
                ))
        return recs


# ── Module-level helpers ──────────────────────────────────────────────────────

def _sanitise_slot_output(raw: dict) -> dict:
    """
    Normalise the raw LLM JSON output before passing it to SlotState.

    The LLM is instructed to return specific types and enum values, but
    defensive parsing here protects against:
      - Extra fields the LLM invented (stripped via explicit key selection)
      - String "true"/"false" instead of bool True/False
      - Invalid enum values (replaced with None so Pydantic doesn't reject)
      - Missing fields (filled with safe defaults)
    """
    VALID_SENIORITY = {
        "graduate", "entry-level", "mid-professional",
        "senior-ic", "manager", "director", "executive",
    }
    VALID_PURPOSE = {"selection", "development", "screening", "audit", "reskilling"}
    VALID_PHASE = {"clarifying", "recommending", "refining", "comparing", "closing"}
    VALID_ACCENT = {"US", "UK", "AU", "IN"}
    VALID_TIME = {"short", "normal"}
    VALID_VOLUME = {"high", "normal"}

    def to_bool(v) -> bool:
        if isinstance(v, bool):
            return v
        if isinstance(v, str):
            return v.lower() in ("true", "1", "yes")
        return bool(v)

    def to_list_of_str(v) -> list:
        if v is None:
            return []
        if isinstance(v, list):
            return [str(i) for i in v if i]
        return []

    def validated_enum(v, valid_set):
        if v in valid_set:
            return v
        if isinstance(v, str) and v.lower() in valid_set:
            return v.lower()
        return None

    return {
        "role":                  raw.get("role") or None,
        "seniority":             validated_enum(raw.get("seniority"), VALID_SENIORITY),
        "purpose":               validated_enum(raw.get("purpose"), VALID_PURPOSE),
        "industry":              raw.get("industry") or None,
        "language":              raw.get("language") or None,
        "accent_variant":        validated_enum(raw.get("accent_variant"), VALID_ACCENT),
        "time_constraint":       validated_enum(raw.get("time_constraint"), VALID_TIME),
        "volume":                validated_enum(raw.get("volume"), VALID_VOLUME),
        "explicit_additions":    to_list_of_str(raw.get("explicit_additions")),
        "explicit_drops":        to_list_of_str(raw.get("explicit_drops")),
        "explicit_test_types":   to_list_of_str(raw.get("explicit_test_types")),
        "current_shortlist_urls": to_list_of_str(raw.get("current_shortlist_urls")),
        "conversation_phase":    validated_enum(raw.get("conversation_phase"), VALID_PHASE) or "clarifying",
        "needs_clarification":   to_bool(raw.get("needs_clarification", True)),
        "clarification_question": raw.get("clarification_question") or None,
        "ready_to_retrieve":     to_bool(raw.get("ready_to_retrieve", False)),
        "is_comparison_turn":    to_bool(raw.get("is_comparison_turn", False)),
        "is_legal_question":     to_bool(raw.get("is_legal_question", False)),
        "is_off_topic":          False,  # Always handled by guardrails, never by LLM
        "end_of_conversation":   to_bool(raw.get("end_of_conversation", False)),
    }

