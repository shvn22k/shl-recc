"""
Catalog loader for the SHL Assessment Recommender.

Loads catalog metadata and the FAISS vector index once at application startup,
then exposes a CatalogStore singleton used by the retriever and agent.

Usage:
    from app.catalog import catalog_store

    # Search by query text
    results = catalog_store.search_by_text("cognitive ability for graduates", k=20)

    # Filter results by job level
    filtered = catalog_store.filter_by_job_level(results, ["graduate", "entry"])

    # Validate a URL before including it in a response
    if catalog_store.is_valid_url(url):
        ...
"""

import json
import logging
import numpy as np
import faiss
from pathlib import Path
from sentence_transformers import SentenceTransformer

logger = logging.getLogger(__name__)

CATALOG_PATH = Path("data/catalog.json")
INDEX_PATH = Path("data/vector_index/index.faiss")
METADATA_PATH = Path("data/vector_index/metadata.json")
EMBEDDING_MODEL = "all-MiniLM-L6-v2"

# Job level synonym expansion — maps natural-language seniority terms to the
# exact strings used in the SHL catalog's job_levels field.
LEVEL_SYNONYMS: dict[str, list[str]] = {
    "graduate":           ["graduate", "entry-level"],
    "entry":              ["entry-level", "graduate", "general population"],
    "entry-level":        ["entry-level", "graduate", "general population"],
    "junior":             ["entry-level", "graduate"],
    "mid":                ["mid-professional", "professional individual contributor"],
    "mid-professional":   ["mid-professional", "professional individual contributor"],
    "professional":       ["professional individual contributor", "mid-professional"],
    "senior":             ["professional individual contributor", "mid-professional"],
    "senior ic":          ["professional individual contributor"],
    "ic":                 ["professional individual contributor"],
    "manager":            ["manager", "front line manager", "supervisor"],
    "lead":               ["manager", "front line manager", "supervisor"],
    "front line manager": ["front line manager", "manager", "supervisor"],
    "supervisor":         ["supervisor", "front line manager", "manager"],
    "director":           ["director", "executive"],
    "executive":          ["executive", "director"],
    "cxo":                ["executive", "director"],
    "vp":                 ["executive", "director"],
    "c-suite":            ["executive", "director"],
    "general population": ["general population"],
}


class CatalogStore:
    """
    In-memory store for the SHL assessment catalog.

    Holds the FAISS vector index, assessment metadata, and the embedding model.
    Designed to be loaded once at startup and reused across all requests — the
    model and index are expensive to initialize but cheap to query.
    """

    def __init__(self):
        self.metadata: list[dict] = []
        self.index: faiss.Index | None = None
        self.model: SentenceTransformer | None = None
        self.url_whitelist: set[str] = set()
        self._loaded = False

    def load(self) -> None:
        """
        Load all resources from disk. Safe to call multiple times — subsequent
        calls are no-ops once the store is initialized.
        """
        if self._loaded:
            return

        logger.info("Loading CatalogStore...")

        # --- Metadata ---
        if not METADATA_PATH.exists():
            raise FileNotFoundError(
                f"Metadata not found at {METADATA_PATH}. "
                "Run: python scraper/embed_catalog.py"
            )
        with open(METADATA_PATH, encoding="utf-8") as f:
            self.metadata = json.load(f)
        logger.info(f"Loaded {len(self.metadata)} assessment metadata records")

        # --- FAISS index ---
        if not INDEX_PATH.exists():
            raise FileNotFoundError(
                f"FAISS index not found at {INDEX_PATH}. "
                "Run: python scraper/embed_catalog.py"
            )
        self.index = faiss.read_index(str(INDEX_PATH))
        logger.info(f"Loaded FAISS index: {self.index.ntotal} vectors")

        # Sanity check — metadata and index must always be in sync
        if len(self.metadata) != self.index.ntotal:
            raise ValueError(
                f"Metadata ({len(self.metadata)} records) and FAISS index "
                f"({self.index.ntotal} vectors) are out of sync. "
                "Re-run embed_catalog.py to rebuild both."
            )

        # --- Embedding model ---
        logger.info(f"Loading embedding model: {EMBEDDING_MODEL}")
        self.model = SentenceTransformer(EMBEDDING_MODEL)
        logger.info("Embedding model loaded")

        # --- URL whitelist ---
        # Every URL we ever return in a response must be in this set.
        # Checked by the agent before finalizing recommendations.
        self.url_whitelist = {item["url"] for item in self.metadata}
        logger.info(f"URL whitelist built: {len(self.url_whitelist)} valid URLs")

        self._loaded = True
        logger.info("CatalogStore ready")

    # ── Embedding & Search ────────────────────────────────────────────────────

    def embed_query(self, query: str) -> np.ndarray:
        """
        Embed a query string into a normalized (1, 384) float32 vector.
        Normalize to match how the index vectors were stored, so inner
        product == cosine similarity.
        """
        self._require_loaded()
        return self.model.encode(
            [query],
            normalize_embeddings=True,
        ).astype(np.float32)

    def search(self, query_vector: np.ndarray, k: int = 20) -> list[dict]:
        """
        Search the FAISS index for the top-k nearest assessments.

        Args:
            query_vector: shape (1, 384) normalized float32 array.
            k: number of results to return (capped at catalog size).

        Returns:
            List of metadata dicts ordered by descending similarity score.
            Each dict gains a 'score' key (cosine similarity, 0.0–1.0).
        """
        self._require_loaded()

        k = min(k, self.index.ntotal)
        scores, indices = self.index.search(query_vector, k)

        results = []
        for score, idx in zip(scores[0], indices[0]):
            if idx == -1:  # FAISS sentinel for empty slots
                continue
            record = dict(self.metadata[idx])
            record["score"] = float(score)
            results.append(record)

        return results

    def search_by_text(self, query: str, k: int = 20) -> list[dict]:
        """
        Embed a query string and search in one call.
        This is the primary entry point used by the retriever.
        """
        return self.search(self.embed_query(query), k=k)

    # ── Lookup ────────────────────────────────────────────────────────────────

    def get_by_url(self, url: str) -> dict | None:
        """Retrieve a single assessment by its exact URL."""
        for item in self.metadata:
            if item["url"] == url:
                return item
        return None

    def get_by_name(self, name: str) -> dict | None:
        """
        Retrieve a single assessment by name (case-insensitive).
        Tries exact match first, then falls back to substring match.
        """
        name_lower = name.lower().strip()
        for item in self.metadata:
            if item["name_lower"] == name_lower:
                return item
        for item in self.metadata:
            if name_lower in item["name_lower"] or item["name_lower"] in name_lower:
                return item
        return None

    def is_valid_url(self, url: str) -> bool:
        """
        Check whether a URL exists in the catalog whitelist.
        Called by the agent before including any URL in a response —
        this is the primary guard against hallucinated or stale URLs.
        """
        return url in self.url_whitelist

    # ── Metadata Filters ──────────────────────────────────────────────────────

    def filter_by_job_level(self, assessments: list[dict], levels: list[str]) -> list[dict]:
        """
        Filter assessments to those suitable for the given job levels.

        Uses synonym expansion so callers can pass natural language terms
        like "senior", "junior", "manager" rather than the exact catalog strings.

        Assessments with no job_levels listed are kept (don't penalise missing data).
        Returns all assessments unchanged if levels is empty.
        """
        if not levels:
            return assessments

        # Expand each requested level to the set of catalog strings it maps to
        target_levels: set[str] = set()
        for lvl in (l.lower() for l in levels):
            expansions = LEVEL_SYNONYMS.get(lvl)
            if expansions:
                target_levels.update(expansions)
            else:
                target_levels.add(lvl)  # pass through unknown terms verbatim

        filtered = []
        for assessment in assessments:
            catalog_levels = set(assessment.get("job_levels_lower", []))
            if not catalog_levels or catalog_levels & target_levels:
                filtered.append(assessment)

        return filtered

    def filter_by_test_type(self, assessments: list[dict], types: list[str]) -> list[dict]:
        """
        Filter assessments to those matching any of the given test type codes
        (e.g. ["A", "P"] keeps Ability and Personality tests).

        Returns all assessments unchanged if types is empty.
        """
        if not types:
            return assessments

        types_upper = {t.upper() for t in types}
        return [
            a for a in assessments
            if types_upper & {t.strip() for t in a.get("test_type", "").split(",")}
        ]

    # ── Properties ────────────────────────────────────────────────────────────

    @property
    def all_assessments(self) -> list[dict]:
        """All metadata records, in catalog order."""
        return self.metadata

    @property
    def size(self) -> int:
        """Total number of assessments in the catalog."""
        return len(self.metadata)

    # ── Internal ──────────────────────────────────────────────────────────────

    def _require_loaded(self) -> None:
        if not self._loaded:
            raise RuntimeError("CatalogStore not loaded. Call catalog_store.load() first.")


# Module-level singleton — every other module imports this directly.
# Loaded once at app startup via the FastAPI lifespan handler in main.py.
catalog_store = CatalogStore()
