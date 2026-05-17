"""
Embedding pipeline for the SHL Assessment Recommender.

Reads data/catalog.json, builds a rich text representation of each assessment,
embeds them using sentence-transformers/all-MiniLM-L6-v2, and saves the FAISS
index + metadata to data/vector_index/.

Run once after scraping (and again if catalog.json is ever updated):
    python scraper/embed_catalog.py
"""

import json
import logging
import numpy as np
import faiss
from pathlib import Path
from sentence_transformers import SentenceTransformer

CATALOG_PATH = Path("data/catalog.json")
INDEX_DIR = Path("data/vector_index")
INDEX_PATH = INDEX_DIR / "index.faiss"
METADATA_PATH = INDEX_DIR / "metadata.json"
EMBEDDING_MODEL = "all-MiniLM-L6-v2"

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)


def build_text_representation(assessment: dict) -> str:
    """
    Convert one assessment into the string we embed.

    Field order is intentional — fields listed earlier have more influence
    on the resulting vector. Name and test type anchor the semantic space;
    description provides the bulk of the signal; job levels and duration
    handle constraint-matching queries like 'short test for graduates'.
    """
    parts = []

    name = assessment.get("name", "").strip()
    if name:
        parts.append(name)

    label = assessment.get("test_type_label", "").strip()
    if label:
        parts.append(label)

    description = assessment.get("description", "").strip()
    if description:
        parts.append(description)

    job_levels = assessment.get("job_levels", [])
    if job_levels:
        parts.append("Job levels: " + ", ".join(job_levels))

    duration = assessment.get("duration", "").strip()
    if duration:
        parts.append(f"Duration: {duration}")

    # Cap at 5 languages — prevents long language lists from swamping the embedding
    languages = assessment.get("languages", [])
    if languages:
        parts.append("Languages: " + ", ".join(languages[:5]))

    if assessment.get("remote_testing"):
        parts.append("Supports remote testing.")

    if assessment.get("adaptive_irt"):
        parts.append("Adaptive and IRT-based.")

    return ". ".join(parts)


def build_metadata_record(assessment: dict) -> dict:
    """
    Extract the fields needed at query time into a clean, flat record.

    Stored in metadata.json parallel to the FAISS index — index position i
    in the FAISS index corresponds to record i here. Pre-computed lowercase
    fields avoid redundant string operations on every search call.
    """
    return {
        "name": assessment.get("name", ""),
        "url": assessment.get("url", ""),
        "test_type": assessment.get("test_type", ""),
        "test_type_label": assessment.get("test_type_label", ""),
        "description": assessment.get("description", ""),
        "job_levels": assessment.get("job_levels", []),
        "languages": assessment.get("languages", []),
        "duration": assessment.get("duration", ""),
        "remote_testing": assessment.get("remote_testing", False),
        "adaptive_irt": assessment.get("adaptive_irt", False),
        # Pre-lowercased for fast keyword matching at query time
        "name_lower": assessment.get("name", "").lower(),
        "description_lower": assessment.get("description", "").lower(),
        "job_levels_lower": [jl.lower() for jl in assessment.get("job_levels", [])],
    }


def run_smoke_test(model: SentenceTransformer, index: faiss.Index, metadata: list[dict]) -> None:
    """
    Run a handful of representative queries and log the top-3 results.
    Gives us a quick sanity check that the semantic space is meaningful.
    """
    test_queries = [
        "cognitive ability test for graduate engineers",
        "personality assessment for senior sales manager",
        "Java programming knowledge test",
        "contact center customer service screening",
        "safety and dependability industrial worker",
    ]

    logger.info("Running smoke test queries...")
    for query in test_queries:
        q_vec = model.encode([query], normalize_embeddings=True).astype(np.float32)
        scores, indices = index.search(q_vec, k=3)
        top = [f"{metadata[i]['name']} ({scores[0][j]:.3f})" for j, i in enumerate(indices[0])]
        logger.info(f"  '{query}'")
        logger.info(f"    -> {top}")


def main() -> None:
    # 1. Load catalog
    logger.info(f"Loading catalog from {CATALOG_PATH}")
    if not CATALOG_PATH.exists():
        raise FileNotFoundError(f"{CATALOG_PATH} not found. Run the scraper first.")

    with open(CATALOG_PATH, encoding="utf-8") as f:
        catalog = json.load(f)
    logger.info(f"Loaded {len(catalog)} assessments")

    # 2. Build text representations and metadata records
    logger.info("Building text representations...")
    texts = [build_text_representation(a) for a in catalog]
    metadata = [build_metadata_record(a) for a in catalog]

    # Spot-check the first record so we can visually confirm field coverage
    logger.info(f"Sample text representation (first 300 chars):\n  {texts[0][:300]}...")

    # 3. Load embedding model (downloads once, then cached locally by HuggingFace)
    logger.info(f"Loading embedding model: {EMBEDDING_MODEL}")
    model = SentenceTransformer(EMBEDDING_MODEL)

    # 4. Embed all assessments
    logger.info("Embedding assessments (may take 30–60 seconds on CPU)...")
    embeddings = model.encode(
        texts,
        batch_size=32,
        show_progress_bar=True,
        normalize_embeddings=True,  # L2-normalize so inner product == cosine similarity
    )
    logger.info(f"Embeddings shape: {embeddings.shape}")  # expected (377, 384)

    # 5. Build FAISS flat index
    #    IndexFlatIP on normalized vectors = exact cosine similarity search.
    #    At 377 items, exact search is instantaneous — no need for approximation.
    logger.info("Building FAISS index...")
    dimension = embeddings.shape[1]
    index = faiss.IndexFlatIP(dimension)
    index.add(embeddings.astype(np.float32))
    logger.info(f"FAISS index: {index.ntotal} vectors, {dimension} dimensions")

    # 6. Save both artefacts to disk
    INDEX_DIR.mkdir(parents=True, exist_ok=True)

    faiss.write_index(index, str(INDEX_PATH))
    logger.info(f"FAISS index saved -> {INDEX_PATH}")

    with open(METADATA_PATH, "w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2, ensure_ascii=False)
    logger.info(f"Metadata saved -> {METADATA_PATH} ({len(metadata)} records)")

    # 7. Quick retrieval quality check
    run_smoke_test(model, index, metadata)

    logger.info("Embedding pipeline complete.")


if __name__ == "__main__":
    main()
