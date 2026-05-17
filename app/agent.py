"""
Core agent logic for the SHL Assessment Recommender.

ChatHandler orchestrates the Conversational Slot-Guided RAG (CSG-RAG) pipeline:

  1. Guardrail pre-checks  — injection, legal, off-topic, gibberish
  2. Slot extraction       — LLM reads conversation, outputs structured intent
  3. Routing               — clarify, retrieve, compare, or close
  4. Hybrid retrieval      — FAISS semantic + BM25 keyword + metadata filter
  5. LLM ranking           — chain-of-thought selection of final 1-10 assessments
  6. Post-processing       — default injection, deduplication, URL whitelist

The handler is stateless — instantiated fresh per request, all context
comes from the messages[] array in the request body.
"""

import logging
import os
import re

from app.models import AgentDecision, Message, Recommendation, SlotState
from app.retriever import retrieve_candidates as _do_retrieve
from app.prompts import build_slot_extractor_prompt, build_ranker_prompt, build_comparison_prompt
from app.llm_client import call_llm_json, LLMError
from app.guardrails import (
    enforce_url_whitelist,
    get_injection_refusal,
    get_legal_refusal,
    get_off_topic_refusal,
    get_gibberish_redirect,
    is_injection_attempt,
    is_legal_question,
    is_off_topic,
    is_gibberish,
)

logger = logging.getLogger(__name__)

MAX_TURNS = int(os.getenv("MAX_TURNS", "8"))

SHL_CATALOG_URL_RE = re.compile(
    r"https://www\.shl\.com/products/product-catalog/view/[^\s\)>\]\"']+",
    re.IGNORECASE,
)
SHORTLIST_FOOTER_RE = re.compile(
    r"\[Shortlist:\s*([^\]]+)\]",
    re.IGNORECASE,
)
COMPARISON_SIGNAL_RE = re.compile(
    r"\b(difference|compare|vs\.?|versus|how does|which is better|explain)\b",
    re.IGNORECASE,
)
CLOSING_SIGNAL_RE = re.compile(
    r"\b(confirmed|that works|that covers it|perfect|locking it in|"
    r"thanks|thank you|final list|keep the shortlist|shortlist as-is|"
    r"that.s good|sounds good|that.ll work|works for me|go ahead)\b",
    re.IGNORECASE,
)
CONTINUATION_SIGNAL_RE = re.compile(
    r"\b(keep (the )?shortlist|as-is|understood|go with (the )?hybrid|"
    r"clear\.|we'll use|final list)\b",
    re.IGNORECASE,
)
REFINEMENT_SIGNAL_RE = re.compile(
    r"\b(add|drop|remove|exclude|also add|without)\b",
    re.IGNORECASE,
)


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

        # Check for gibberish / empty-intent messages
        if is_gibberish(latest):
            return AgentDecision(
                reply=get_gibberish_redirect(),
                recommendations=[],
                end_of_conversation=False,
            )

        # ── 3. Slot extraction ────────────────────────────────────────────────
        slots = await self._extract_slots(messages)
        slots = self._normalize_slots(slots, latest)
        slots = self._merge_recovered_shortlist(slots, messages)

        # ── 4. Clarification turn — return question, skip retrieval ───────────
        if slots.needs_clarification and not slots.ready_to_retrieve:
            return AgentDecision(
                reply=slots.clarification_question or (
                    "Could you tell me more about the role you're hiring for?"
                ),
                recommendations=[],
                end_of_conversation=False,
            )

        # ── 5. Closing turn — BEFORE comparison (sticky compare flag fix) ─────
        if slots.end_of_conversation:
            return await self._handle_closing_turn(messages, slots)

        # ── 6. Comparison turn — explain differences, no new recommendations ──
        if slots.is_comparison_turn:
            reply = await self._handle_comparison(messages, slots)
            return AgentDecision(
                reply=reply,
                recommendations=[],
                end_of_conversation=False,
            )

        # ── 7. Retrieval + ranking (Phases 5 & 6) ────────────────────────────
        candidates = await self._retrieve_candidates(slots)
        decision = await self._rank_and_respond(messages, slots, candidates)
        return self._finalize_recommendation_decision(decision, slots)

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

    def _normalize_slots(self, slots: SlotState, latest_user_message: str) -> SlotState:
        """Correct common slot-extractor mistakes using the latest user message."""
        latest = latest_user_message.lower()
        updates: dict = {}

        if slots.is_comparison_turn and not COMPARISON_SIGNAL_RE.search(latest):
            updates["is_comparison_turn"] = False

        if CLOSING_SIGNAL_RE.search(latest):
            updates.update({
                "end_of_conversation": True,
                "conversation_phase": "closing",
                "is_comparison_turn": False,
                "needs_clarification": False,
            })

        if slots.conversation_phase in ("closing", "refining"):
            updates["is_comparison_turn"] = False

        if slots.end_of_conversation:
            updates.update({
                "is_comparison_turn": False,
                "needs_clarification": False,
            })

        if (
            not is_legal_question(latest_user_message)
            and CONTINUATION_SIGNAL_RE.search(latest)
        ):
            updates.update({
                "needs_clarification": False,
                "ready_to_retrieve": True,
                "is_legal_question": False,
                "is_comparison_turn": False,
            })

        if REFINEMENT_SIGNAL_RE.search(latest) or slots.explicit_additions or slots.explicit_drops:
            updates.update({
                "needs_clarification": False,
                "ready_to_retrieve": True,
                "is_comparison_turn": False,
            })

        if updates:
            slots = slots.model_copy(update=updates)
        return slots

    def _recover_shortlist_urls(self, messages: list[Message]) -> list[str]:
        """Extract catalog URLs from prior assistant replies (footer + inline)."""
        seen: set[str] = set()
        ordered: list[str] = []

        for msg in reversed(messages):
            if msg.role != "assistant":
                continue
            footer = SHORTLIST_FOOTER_RE.search(msg.content)
            if footer:
                for part in footer.group(1).split(","):
                    url = part.strip().rstrip("/")
                    if self.catalog_store.is_valid_url(url) and url not in seen:
                        seen.add(url)
                        ordered.append(url)
            for url in SHL_CATALOG_URL_RE.findall(msg.content):
                url = url.rstrip("/.,;")
                if self.catalog_store.is_valid_url(url) and url not in seen:
                    seen.add(url)
                    ordered.append(url)

        ordered.reverse()
        return ordered

    def _merge_recovered_shortlist(self, slots: SlotState, messages: list[Message]) -> SlotState:
        """Merge extractor URLs with URLs recovered from conversation history."""
        recovered = self._recover_shortlist_urls(messages)
        merged: list[str] = []
        seen: set[str] = set()
        for url in list(slots.current_shortlist_urls) + recovered:
            if url and url not in seen and self.catalog_store.is_valid_url(url):
                seen.add(url)
                merged.append(url)
        if merged != slots.current_shortlist_urls:
            return slots.model_copy(update={"current_shortlist_urls": merged})
        return slots

    def _recommendations_from_urls(
        self, urls: list[str], explicit_drops: list[str]
    ) -> list[Recommendation]:
        """Build recommendations from catalog URLs; honour explicit drops."""
        recs: list[Recommendation] = []
        for url in urls:
            item = self.catalog_store.get_by_url(url)
            if not item:
                continue
            if self._should_drop(item["name"], explicit_drops):
                continue
            recs.append(Recommendation(
                name=item["name"],
                url=item["url"],
                test_type=self._clean_test_type(item["test_type"]),
            ))
        return recs

    def _recover_recommendations_from_assistant_text(
        self, messages: list[Message], explicit_drops: list[str]
    ) -> list[Recommendation]:
        """Match catalog assessment names mentioned in prior assistant replies."""
        for msg in reversed(messages):
            if msg.role != "assistant":
                continue
            text_lower = msg.content.lower()
            recs: list[Recommendation] = []
            seen_urls: set[str] = set()
            for item in self.catalog_store.metadata:
                name = item["name"]
                if len(name) < 10:
                    continue
                if name.lower() not in text_lower:
                    continue
                if self._should_drop(name, explicit_drops):
                    continue
                if item["url"] in seen_urls:
                    continue
                seen_urls.add(item["url"])
                recs.append(Recommendation(
                    name=name,
                    url=item["url"],
                    test_type=self._clean_test_type(item["test_type"]),
                ))
            if recs:
                logger.info(
                    f"Recovered {len(recs)} recommendations from assistant text"
                )
                return recs[:10]
        return []

    async def _handle_closing_turn(
        self, messages: list[Message], slots: SlotState
    ) -> AgentDecision:
        """
        Echo the agreed shortlist on closing — no full re-rank that collapses the list.
        """
        latest = self._get_latest_user_message(messages)
        if re.search(r"keep (the )?shortlist|as-is", latest, re.IGNORECASE):
            recall_slots = slots.model_copy(
                update={
                    "end_of_conversation": False,
                    "needs_clarification": False,
                    "ready_to_retrieve": True,
                    "is_comparison_turn": False,
                }
            )
            candidates = await self._retrieve_candidates(recall_slots)
            decision = await self._rank_and_respond(messages, recall_slots, candidates)
            decision = self._finalize_recommendation_decision(decision, recall_slots)
            return AgentDecision(
                reply=decision.reply,
                recommendations=decision.recommendations,
                end_of_conversation=True,
            )

        current_recs = self._recommendations_from_urls(
            slots.current_shortlist_urls, slots.explicit_drops
        )

        if not current_recs:
            current_recs = self._recover_recommendations_from_assistant_text(
                messages, slots.explicit_drops
            )

        if not current_recs:
            logger.warning("Closing turn: no shortlist recovered — retrieval fallback")
            recall_slots = slots.model_copy(
                update={
                    "end_of_conversation": False,
                    "needs_clarification": False,
                    "ready_to_retrieve": True,
                }
            )
            candidates = await self._retrieve_candidates(recall_slots)
            decision = await self._rank_and_respond(messages, recall_slots, candidates)
            current_recs = decision.recommendations

        current_recs = self._inject_defaults(
            current_recs, slots, self.catalog_store
        )
        current_recs = enforce_url_whitelist(current_recs, self.catalog_store)

        reply = self._append_shortlist_footer(
            self._build_closing_reply(slots), current_recs
        )
        return AgentDecision(
            reply=reply,
            recommendations=current_recs,
            end_of_conversation=True,
        )

    def _finalize_recommendation_decision(
        self, decision: AgentDecision, slots: SlotState
    ) -> AgentDecision:
        """Defaults, whitelist, and shortlist footer for recommending/refining turns."""
        decision.recommendations = self._inject_defaults(
            decision.recommendations, slots, self.catalog_store
        )
        decision.recommendations = enforce_url_whitelist(
            decision.recommendations, self.catalog_store
        )
        decision.reply = self._append_shortlist_footer(
            decision.reply, decision.recommendations
        )
        return decision

    @staticmethod
    def _append_shortlist_footer(reply: str, recommendations: list[Recommendation]) -> str:
        """Embed catalog URLs in reply so the next turn can recover the shortlist."""
        if not recommendations:
            return reply
        urls = [r.url for r in recommendations]
        footer = "[Shortlist: " + ", ".join(urls) + "]"
        if footer in reply:
            return reply
        return reply.rstrip() + "\n\n" + footer

    # ── Stub methods — replaced in Phases 4, 5, 6 ────────────────────────────

    async def _extract_slots(self, messages: list[Message]) -> SlotState:
        """
        Extract structured hiring intent from the full conversation history.

        Sends the conversation to the LLM with a structured extraction prompt.
        Returns a SlotState describing what the hiring manager needs and what
        the agent should do next.

        The SlotState drives all downstream decisions:
          needs_clarification  → return question, skip retrieval
          ready_to_retrieve    → proceed to hybrid retrieval
          is_comparison_turn   → return explanation, no new retrieval
          end_of_conversation  → repeat shortlist, set end=true
          explicit_additions/drops → honoured absolutely by the ranker

        Low temperature (0.1) keeps extraction deterministic across repeated
        calls on the same conversation.
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
        Run hybrid retrieval for the given SlotState.

        Delegates to app/retriever.py:
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
        LLM Call 2 — select the final 1-10 assessments from retrieved candidates.

        Uses chain-of-thought reasoning to pick the best-fit assessments,
        then applies explicit_drops as a hard post-processing filter.
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
            # end_of_conversation is controlled exclusively by the slot extractor
            # (closing signals) and turn cap — never by the ranker.
            end_conv = False

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
                test_type = self._clean_test_type(item.get("test_type", ""))

                # Skip duplicates
                if url in seen_urls:
                    continue

                # Validate URL is from candidates or catalog whitelist
                if url not in candidate_urls:
                    # Try to find the correct URL from catalog by name
                    match = self.catalog_store.get_by_name(name)
                    if match:
                        url = match["url"]
                        test_type = self._clean_test_type(match["test_type"])
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

        # Final deduplication — remove semantic duplicates keeping first occurrence
        # Catches cases where ranker selects both an instrument and its report variant
        deduplicated = []
        seen_key_terms = []

        for rec in recommendations:
            rec_terms = [
                t for t in rec.name.lower().split() 
                if len(t) >= 2 and t not in {"test", "report", "assessment", "shl", "the", "for", "new", "(new)"}
            ]
            is_dup = False
            for seen_terms in seen_key_terms:
                overlap = sum(1 for t in rec_terms if t in seen_terms)
                if overlap >= 2:
                    logger.info(
                        f"Dedup: removing '{rec.name}' as duplicate of earlier entry"
                    )
                    is_dup = True
                    break
            if not is_dup:
                deduplicated.append(rec)
                seen_key_terms.append(set(rec_terms))

        recommendations = deduplicated

        return AgentDecision(
            reply=reply,
            recommendations=recommendations,
            end_of_conversation=end_conv,
        )

    async def _handle_comparison(
        self, messages: list[Message], slots: SlotState
    ) -> str:
        """
        Answer a comparison or explanation question grounded in catalog data.

        Fetches catalog records for assessments in current_shortlist_urls,
        injects their structured metadata into the prompt, and asks the LLM
        to compare them without inventing details.

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

    def _clean_test_type(self, raw: str) -> str:
        """
        Strip labels from test_type, return only the code(s).
        'K (Knowledge & Skills)' → 'K'
        'A,C,P (Ability & Aptitude, Competencies, Personality & Behavior)' → 'A,C,P'
        'P' → 'P'
        """
        if not raw:
            return ""
        # Take everything before the first '(' and strip whitespace/commas
        code_part = raw.split("(")[0].strip().rstrip(",").strip()
        return code_part

    def _inject_defaults(
        self,
        recommendations: list,
        slots: SlotState,
        catalog_store,
    ) -> list:
        """
        Inject OPQ32r and Verify G+ as defaults when applicable,
        unless the user explicitly dropped them.

        This runs AFTER the ranker so defaults are always present
        regardless of whether they ranked highly in retrieval.
        """
        from app.models import Recommendation

        existing_urls = {r.url for r in recommendations}
        existing_names_lower = {r.name.lower() for r in recommendations}

        drops_lower = [d.lower() for d in slots.explicit_drops]

        def is_dropped(name: str) -> bool:
            name_l = name.lower()
            return any(d in name_l or name_l in d for d in drops_lower)

        def maybe_append(assessment_name: str):
            """Fetch from catalog and append if not already present and not dropped."""
            if is_dropped(assessment_name):
                return
            item = catalog_store.get_by_name(assessment_name)
            if not item:
                return
            if item["url"] in existing_urls:
                return
            
            item_name_lower = item["name"].lower()
            key_terms = [
                t for t in item_name_lower.split() 
                if len(t) >= 2 and t not in {"test", "report", "assessment", "shl", "the", "for", "new", "(new)"}
            ]
            import logging
            logger = logging.getLogger(__name__)
            
            # If the ranker hallucinated a variant (like a report instead of the instrument),
            # replace it with the correct default
            for existing_name in list(existing_names_lower):
                matches = sum(1 for t in key_terms if t in existing_name)
                if matches >= 2:  # 2+ shared meaningful words = semantic duplicate
                    logger.info(
                        f"Default injection '{item['name']}' replaces "
                        f"semantic duplicate '{existing_name}'"
                    )
                    for idx, r in enumerate(recommendations):
                        if r.name.lower() == existing_name:
                            recommendations[idx] = Recommendation(
                                name=item["name"],
                                url=item["url"],
                                test_type=self._clean_test_type(item["test_type"]),
                            )
                            existing_urls.add(item["url"])
                            existing_names_lower.add(item_name_lower)
                            return
            
            if len(recommendations) >= 10:
                return
            
            recommendations.append(Recommendation(
                name=item["name"],
                url=item["url"],
                test_type=self._clean_test_type(item["test_type"]),
            ))
            existing_urls.add(item["url"])
            existing_names_lower.add(item_name_lower)
            logger.info(f"Default injection: {item['name']}")

        # OPQ32r — inject for mid/senior/executive SELECTION roles and for
        # general screening where seniority is unspecified (the ranker prompt
        # already instructs "include OPQ32r for any mid/senior selection role").
        OPQ_APPLICABLE_SENIORITY = {
            "mid-professional", "senior-ic", "manager", "director", "executive",
        }
        OPQ_APPLICABLE_PURPOSE = {"selection", "screening", None}

        seniority_ok = (
            slots.seniority in OPQ_APPLICABLE_SENIORITY
            or slots.seniority is None  # inject when seniority unknown + non-high-volume
        )
        if (
            seniority_ok
            and slots.purpose in OPQ_APPLICABLE_PURPOSE
            and slots.volume != "high"
        ):
            maybe_append("OPQ32r")

        # Verify G+ — inject for graduate and senior-IC selection
        VERIFY_APPLICABLE_SENIORITY = {"graduate", "senior-ic"}
        if (
            slots.seniority in VERIFY_APPLICABLE_SENIORITY
            and slots.purpose in {"selection", "screening", None}
        ):
            maybe_append("Verify Interactive G+")

        return recommendations

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
        """Reconstruct the current shortlist from URLs stored in slot state."""
        return self._recommendations_from_urls(
            slots.current_shortlist_urls, slots.explicit_drops
        )


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

