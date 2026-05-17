"""
LLM client for the SHL Assessment Recommender.

Provides a unified async interface over two providers:
  Primary:  Google Gemini 2.5 Flash  — fast, generous context window
  Fallback: OpenAI GPT-4o-mini       — reliable, predictable latency

Both providers support JSON mode (response_mime_type / response_format),
which is used for all structured extraction calls to guarantee parseable output.

Retry logic uses tenacity with exponential backoff. If the primary provider
fails after 2 attempts, the fallback is tried automatically. LLMError is
raised only if both providers fail.
"""

import asyncio
import json
import logging
import os
from typing import Optional

import google.generativeai as genai
from openai import AsyncOpenAI
from tenacity import (
    retry,
    retry_if_exception_type,
    stop_after_attempt,
    wait_exponential,
)
from dotenv import load_dotenv

load_dotenv()

logger = logging.getLogger(__name__)

# ── Configuration ─────────────────────────────────────────────────────────────

GEMINI_API_KEY = os.getenv("GEMINI_API_KEY", "")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "")
OPENAI_ORG = os.getenv("OPENAI_ORGANIZATION", "")
OPENAI_PROJECT = os.getenv("OPENAI_PROJECT", "")
GEMINI_MODEL = os.getenv("GEMINI_MODEL", "gemini-2.5-flash")
OPENAI_MODEL = os.getenv("OPENAI_MODEL", "gpt-4o-mini")
LLM_PROVIDER = os.getenv("LLM_PROVIDER", "gemini")

if GEMINI_API_KEY:
    genai.configure(api_key=GEMINI_API_KEY)

openai_client = (
    AsyncOpenAI(
        api_key=OPENAI_API_KEY,
        organization=OPENAI_ORG or None,
        project=OPENAI_PROJECT or None,
    )
    if OPENAI_API_KEY
    else None
)


class LLMError(Exception):
    """Raised when all LLM providers fail after retries."""
    pass


# ── Gemini Call ───────────────────────────────────────────────────────────────

@retry(
    retry=retry_if_exception_type(Exception),
    stop=stop_after_attempt(2),
    wait=wait_exponential(multiplier=1, min=1, max=4),
    reraise=True,
)
async def _call_gemini(
    system_prompt: str,
    user_prompt: str,
    json_mode: bool = False,
    temperature: float = 0.1,
) -> str:
    """
    Call Gemini 2.5 Flash. Returns the raw text response.
    Raises on failure — the caller handles fallback to OpenAI.

    The Gemini SDK is synchronous, so we run it in a thread pool
    to avoid blocking the async event loop.
    """
    if not GEMINI_API_KEY:
        raise LLMError("Gemini not configured — missing GEMINI_API_KEY")

    generation_config: dict = {
        "temperature": temperature,
        "max_output_tokens": 1500,
    }
    if json_mode:
        generation_config["response_mime_type"] = "application/json"

    model = genai.GenerativeModel(
        model_name=GEMINI_MODEL,
        system_instruction=system_prompt,
        generation_config=generation_config,
    )

    response = await asyncio.get_event_loop().run_in_executor(
        None,
        lambda: model.generate_content(user_prompt),
    )

    if not response.text:
        raise LLMError("Gemini returned an empty response")

    logger.debug(f"Gemini ({GEMINI_MODEL}) responded: {response.text[:120]}...")
    return response.text.strip()


# ── OpenAI Call ───────────────────────────────────────────────────────────────

@retry(
    retry=retry_if_exception_type(Exception),
    stop=stop_after_attempt(2),
    wait=wait_exponential(multiplier=1, min=1, max=4),
    reraise=True,
)
async def _call_openai(
    system_prompt: str,
    user_prompt: str,
    json_mode: bool = False,
    temperature: float = 0.1,
) -> str:
    """
    Call OpenAI GPT-4o-mini. Returns the raw text response.
    """
    if not openai_client:
        raise LLMError("OpenAI not configured — missing OPENAI_API_KEY")

    kwargs: dict = {
        "model": OPENAI_MODEL,
        "temperature": temperature,
        "max_tokens": 1500,
        "messages": [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ],
    }
    if json_mode:
        kwargs["response_format"] = {"type": "json_object"}

    response = await openai_client.chat.completions.create(**kwargs)
    text = response.choices[0].message.content

    if not text:
        raise LLMError("OpenAI returned an empty response")

    logger.debug(f"OpenAI ({OPENAI_MODEL}) responded: {text[:120]}...")
    return text.strip()


# ── Public Interface ──────────────────────────────────────────────────────────

async def call_llm(
    system_prompt: str,
    user_prompt: str,
    json_mode: bool = False,
    temperature: float = 0.1,
    force_provider: Optional[str] = None,
) -> str:
    """
    Call the LLM with automatic Gemini → OpenAI fallback.

    Args:
        system_prompt:   The system/instruction prompt.
        user_prompt:     The user-facing input (conversation + task).
        json_mode:       Force JSON output — no markdown fences, directly parseable.
        temperature:     Sampling temperature. Low (0.1) = deterministic and consistent.
        force_provider:  Override default provider: 'gemini' or 'openai'.

    Returns:
        Raw text response string from the LLM.

    Raises:
        LLMError: If all providers fail after retries.
    """
    provider = force_provider or LLM_PROVIDER
    primary_error: Optional[Exception] = None

    # Primary attempt
    try:
        if provider == "gemini":
            logger.debug(f"Calling primary LLM: Gemini ({GEMINI_MODEL})")
            return await _call_gemini(system_prompt, user_prompt, json_mode, temperature)
        else:
            logger.debug(f"Calling primary LLM: OpenAI ({OPENAI_MODEL})")
            return await _call_openai(system_prompt, user_prompt, json_mode, temperature)
    except Exception as e:
        primary_error = e
        logger.warning(f"Primary LLM ({provider}) failed after retries: {e}. Trying fallback.")

    # Fallback attempt
    try:
        if provider == "gemini":
            logger.info("Falling back to OpenAI")
            return await _call_openai(system_prompt, user_prompt, json_mode, temperature)
        else:
            logger.info("Falling back to Gemini")
            return await _call_gemini(system_prompt, user_prompt, json_mode, temperature)
    except Exception as fallback_error:
        raise LLMError(
            f"All LLM providers failed. "
            f"Primary ({provider}) error: {primary_error}. "
            f"Fallback error: {fallback_error}"
        )


async def call_llm_json(
    system_prompt: str,
    user_prompt: str,
    temperature: float = 0.1,
) -> dict:
    """
    Call the LLM with JSON mode enabled and parse the response.

    Returns:
        Parsed dict from the JSON response.

    Raises:
        LLMError:   If the LLM call fails.
        ValueError: If the response is not valid JSON.
    """
    raw = await call_llm(
        system_prompt=system_prompt,
        user_prompt=user_prompt,
        json_mode=True,
        temperature=temperature,
    )

    # Strip markdown fences as a safety net — shouldn't occur in json_mode
    # but some models/versions still wrap output in ```json ... ```
    cleaned = raw.strip()
    if cleaned.startswith("```"):
        lines = cleaned.split("\n")
        cleaned = "\n".join(lines[1:-1]) if len(lines) > 2 else cleaned

    try:
        return json.loads(cleaned)
    except json.JSONDecodeError as e:
        logger.error(f"Failed to parse LLM JSON response: {e}\nRaw output: {raw[:500]}")
        raise ValueError(f"LLM returned invalid JSON: {e}")
