"""
Pydantic models for the SHL Assessment Recommender.

Two layers:
  1. API schema types  (Message, Recommendation, ChatRequest, ChatResponse)
     — the non-negotiable contract with the SHL evaluator. Field names and
     types here must never change without a corresponding spec update.

  2. Internal pipeline types  (SlotState, CandidateAssessment, AgentDecision)
     — used within the agent pipeline and never serialised directly to the
     API response.
"""

from typing import Literal, Optional
from pydantic import BaseModel, Field, field_validator, model_validator


# ─────────────────────────────────────────────────────────────────────────────
# API Schema Types — DO NOT MODIFY FIELD NAMES
# These map directly to the SHL evaluator contract.
# ─────────────────────────────────────────────────────────────────────────────

class Message(BaseModel):
    role: Literal["user", "assistant"]
    content: str = Field(..., min_length=1)

    @field_validator("content")
    @classmethod
    def content_not_whitespace(cls, v: str) -> str:
        if not v.strip():
            raise ValueError("content must not be empty or whitespace only")
        return v


class Recommendation(BaseModel):
    name: str = Field(..., min_length=1)
    url: str = Field(..., min_length=1)
    test_type: str = Field(..., min_length=1)

    @field_validator("url")
    @classmethod
    def url_must_be_shl(cls, v: str) -> str:
        if not v.startswith("https://www.shl.com/"):
            raise ValueError(f"URL must be an SHL catalog URL, got: {v}")
        return v


class ChatRequest(BaseModel):
    messages: list[Message] = Field(..., min_length=1)

    @field_validator("messages")
    @classmethod
    def last_message_must_be_user(cls, v: list[Message]) -> list[Message]:
        if v and v[-1].role != "user":
            raise ValueError("Last message in the conversation must be from the user")
        return v


class ChatResponse(BaseModel):
    reply: str = Field(..., min_length=1)
    recommendations: list[Recommendation] = Field(default_factory=list)
    end_of_conversation: bool = Field(default=False)

    @field_validator("recommendations")
    @classmethod
    def recommendations_bounded(cls, v: list[Recommendation]) -> list[Recommendation]:
        if len(v) > 10:
            raise ValueError(
                f"recommendations must not exceed 10 items, got {len(v)}"
            )
        return v

    @model_validator(mode="after")
    def reply_not_whitespace(self) -> "ChatResponse":
        if not self.reply.strip():
            raise ValueError("reply must not be empty or whitespace only")
        return self


# ─────────────────────────────────────────────────────────────────────────────
# Internal Pipeline Types
# Used by agent / retriever / ranker — never returned directly to the caller.
# ─────────────────────────────────────────────────────────────────────────────

class SlotState(BaseModel):
    """
    Structured intent extracted from the conversation history by the Slot
    Extractor (LLM Call 1). All fields are optional — slots fill progressively
    across turns as the user provides more context.
    """

    # Core identification slots
    role: Optional[str] = None
    seniority: Optional[Literal[
        "graduate", "entry-level", "mid-professional",
        "senior-ic", "manager", "director", "executive"
    ]] = None
    purpose: Optional[Literal[
        "selection", "development", "screening", "audit", "reskilling"
    ]] = None
    industry: Optional[str] = None
    language: Optional[str] = None
    accent_variant: Optional[Literal["US", "UK", "AU", "IN"]] = None

    # Constraint slots
    time_constraint: Optional[Literal["short", "normal"]] = None
    volume: Optional[Literal["high", "normal"]] = None

    # Explicit user instructions — ABSOLUTE, the agent must honour these
    explicit_additions: list[str] = Field(default_factory=list)
    explicit_drops: list[str] = Field(default_factory=list)
    explicit_test_types: list[str] = Field(default_factory=list)

    # Current agreed shortlist (carried forward across refinement turns)
    current_shortlist_urls: list[str] = Field(default_factory=list)

    # Conversation phase
    conversation_phase: Literal[
        "clarifying", "recommending", "refining", "comparing", "closing"
    ] = "clarifying"

    # Agent decision flags
    needs_clarification: bool = True
    clarification_question: Optional[str] = None
    ready_to_retrieve: bool = False
    is_comparison_turn: bool = False
    is_legal_question: bool = False
    is_off_topic: bool = False
    end_of_conversation: bool = False


class CandidateAssessment(BaseModel):
    """A catalog assessment enriched with its retrieval score, used inside the pipeline."""
    name: str
    url: str
    test_type: str
    test_type_label: str
    description: str
    job_levels: list[str] = Field(default_factory=list)
    languages: list[str] = Field(default_factory=list)
    duration: str = ""
    remote_testing: bool = False
    adaptive_irt: bool = False
    score: float = 0.0  # cosine similarity from FAISS


class AgentDecision(BaseModel):
    """
    The fully resolved decision for one conversation turn.
    Produced by ChatHandler, consumed by the /chat endpoint to build ChatResponse.
    """
    reply: str
    recommendations: list[Recommendation] = Field(default_factory=list)
    end_of_conversation: bool = False
