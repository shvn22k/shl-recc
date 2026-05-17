"""
SHL Assessment Recommender — FastAPI application entry point.

Wires together the lifespan (catalog loading), middleware, and route handlers.
Heavy resources (FAISS index, embedding model) are loaded once at startup via
the lifespan context manager and reused across all requests.
"""

import logging
import os
from contextlib import asynccontextmanager

from dotenv import load_dotenv
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

load_dotenv()

logging.basicConfig(
    level=os.getenv("LOG_LEVEL", "INFO"),
    format="%(asctime)s %(name)s %(levelname)s %(message)s",
)
logger = logging.getLogger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    Load heavy resources once at startup, release cleanly on shutdown.

    CatalogStore initialisation takes ~5–15 seconds (model load + FAISS read).
    Everything after yield runs at shutdown.
    """
    logger.info("Starting SHL Assessment Recommender...")
    from app.catalog import catalog_store
    catalog_store.load()
    logger.info("Startup complete. Ready to serve requests.")
    yield
    logger.info("Shutting down.")


app = FastAPI(
    title="SHL Assessment Recommender",
    description="Conversational agent for recommending SHL assessments to hiring managers.",
    version="0.1.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/health", tags=["meta"])
async def health():
    """Liveness probe — returns ok when the service is up and catalog is loaded."""
    return {"status": "ok"}


@app.post("/chat", tags=["agent"])
async def chat_stub():
    """
    Conversational assessment recommender.
    Stub — will be fully implemented in Phases 4–6.
    """
    return {
        "reply": "Service is up. Full agent coming soon.",
        "recommendations": [],
        "end_of_conversation": False,
    }
