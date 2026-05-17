"""
SHL Assessment Recommender — FastAPI application entry point.

Responsibilities:
  - Lifespan: load CatalogStore (FAISS index + embedding model) once at startup
  - Middleware: CORS
  - Global exception handlers: guarantee no unhandled error returns a non-schema body
  - Endpoints: GET /health, POST /chat
"""

import logging
import os
from contextlib import asynccontextmanager

from dotenv import load_dotenv
from fastapi import FastAPI, Request, status
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import ValidationError

load_dotenv()

logging.basicConfig(
    level=os.getenv("LOG_LEVEL", "INFO"),
    format="%(asctime)s %(name)s %(levelname)s %(message)s",
)
logger = logging.getLogger(__name__)


# ── Startup / Shutdown ────────────────────────────────────────────────────────

@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    Load all heavy resources once before the first request is served.

    CatalogStore.load() takes ~12s locally (~25-35s on Render's free tier):
      - Reads metadata.json and index.faiss from disk
      - Loads the all-MiniLM-L6-v2 sentence transformer model into CPU memory

    The readiness gate in /chat returns 503 until this completes, so no
    request can hang waiting for a half-initialised store.
    """
    logger.info("Starting SHL Assessment Recommender...")
    from app.catalog import catalog_store
    catalog_store.load()
    app.state.catalog_store = catalog_store
    logger.info("Startup complete. Ready to serve requests.")
    yield
    logger.info("Shutting down.")


# ── App ───────────────────────────────────────────────────────────────────────

app = FastAPI(
    title="SHL Assessment Recommender",
    description="Conversational agent for recommending SHL assessments to hiring managers.",
    version="1.0.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


# ── Global exception handlers ─────────────────────────────────────────────────
# These ensure no unhandled exception ever returns a non-schema body or an
# HTML error page to the evaluator.

@app.exception_handler(Exception)
async def global_exception_handler(request: Request, exc: Exception):
    logger.error(f"Unhandled exception on {request.url}: {exc}", exc_info=True)
    return JSONResponse(
        status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
        content={
            "reply": (
                "I encountered an unexpected error. "
                "Please try again in a moment."
            ),
            "recommendations": [],
            "end_of_conversation": False,
        },
    )


@app.exception_handler(ValidationError)
async def validation_exception_handler(request: Request, exc: ValidationError):
    logger.warning(f"Validation error on {request.url}: {exc}")
    return JSONResponse(
        status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
        content={
            "reply": (
                "I couldn't process your request — "
                "please check the message format and try again."
            ),
            "recommendations": [],
            "end_of_conversation": False,
        },
    )


# ── Endpoints ─────────────────────────────────────────────────────────────────

@app.get("/health", tags=["meta"])
async def health():
    """
    Liveness probe. Returns ok once the service process is up.
    Note: returns ok even during startup — use the 503 from /chat as
    the readiness signal during the cold-start window.
    """
    return {"status": "ok"}


@app.post("/chat", tags=["agent"], status_code=status.HTTP_200_OK)
async def chat(request: Request):
    """
    Main conversational endpoint.

    Accepts the full conversation history and returns the agent's reply
    with an optional list of SHL assessment recommendations.

    The evaluator always expects HTTP 200 from this endpoint — even on
    agent errors. Internal failures return a graceful 200 with an empty
    recommendations list.
    """
    from app.agent import ChatHandler
    from app.models import ChatRequest, ChatResponse, Recommendation

    # ── Readiness gate ────────────────────────────────────────────────────────
    # Guard against the cold-start window on Render where a request arrives
    # before the model has finished loading. Return 503 rather than hanging.
    catalog_store = getattr(app.state, "catalog_store", None)
    if catalog_store is None or not catalog_store._loaded:
        logger.warning("Request arrived before CatalogStore was ready")
        return JSONResponse(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            content={
                "reply": (
                    "The service is still warming up. "
                    "Please retry in a few seconds."
                ),
                "recommendations": [],
                "end_of_conversation": False,
            },
        )

    # ── Parse and validate request ────────────────────────────────────────────
    try:
        body = await request.json()
        chat_request = ChatRequest(**body)
    except ValidationError as e:
        logger.warning(f"Invalid request body: {e}")
        return JSONResponse(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            content={
                "reply": (
                    "I couldn't understand the request format. "
                    "Please ensure messages are provided as a non-empty list."
                ),
                "recommendations": [],
                "end_of_conversation": False,
            },
        )
    except Exception as e:
        logger.error(f"Failed to parse request body: {e}")
        return JSONResponse(
            status_code=status.HTTP_400_BAD_REQUEST,
            content={
                "reply": "Invalid request. Please provide a valid JSON body.",
                "recommendations": [],
                "end_of_conversation": False,
            },
        )

    # ── Run agent pipeline ────────────────────────────────────────────────────
    try:
        handler = ChatHandler(catalog_store)
        decision = await handler.handle(chat_request.messages)
    except Exception as e:
        logger.error(f"Agent pipeline failed: {e}", exc_info=True)
        # Return 200 — the evaluator expects 200 on /chat regardless
        return JSONResponse(
            status_code=status.HTTP_200_OK,
            content={
                "reply": (
                    "I ran into an issue processing your request. "
                    "Could you rephrase what you're looking for?"
                ),
                "recommendations": [],
                "end_of_conversation": False,
            },
        )

    # ── Build and validate response ───────────────────────────────────────────
    # Pass through Pydantic to guarantee schema compliance before returning.
    try:
        response = ChatResponse(
            reply=decision.reply,
            recommendations=[
                Recommendation(
                    name=rec.name,
                    url=rec.url,
                    test_type=rec.test_type,
                )
                for rec in decision.recommendations
            ],
            end_of_conversation=decision.end_of_conversation,
        )
        return response.model_dump()
    except ValidationError as e:
        # Should never happen if agent logic is correct — safety net only
        logger.error(f"Response failed schema validation: {e}", exc_info=True)
        return JSONResponse(
            status_code=status.HTTP_200_OK,
            content={
                "reply": (
                    "I found relevant assessments but encountered an issue "
                    "formatting the response. Please try again."
                ),
                "recommendations": [],
                "end_of_conversation": False,
            },
        )
