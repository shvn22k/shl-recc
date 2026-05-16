"""
SHL Catalog Scraper
====================
Scrapes all Individual Test Solutions from the SHL product catalog.
Outputs: data/catalog.json

Usage:
    python scraper/catalog_scraper.py
"""

import asyncio
import httpx
import json
import logging
import re
import time
from bs4 import BeautifulSoup
from datetime import datetime, timezone
from pathlib import Path

BASE_URL = "https://www.shl.com"
CATALOG_URL = "https://www.shl.com/products/product-catalog/"
OUTPUT_PATH = Path("data/catalog.json")
PAGE_SIZE = 12
REQUEST_DELAY = 1.5  # seconds between requests — be polite to SHL's servers
MAX_RETRIES = 3

TEST_TYPE_LABELS = {
    "A": "Ability & Aptitude",
    "P": "Personality & Behavior",
    "K": "Knowledge & Skills",
    "B": "Biodata & Situational Judgment",
    "S": "Simulations",
    "C": "Competencies",
    "D": "Development & 360",
    "E": "Engagement",  # seen in the wild on SHL pages
}

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)

HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
        "AppleWebKit/537.36 (KHTML, like Gecko) "
        "Chrome/124.0.0.0 Safari/537.36"
    ),
    "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
    "Accept-Language": "en-US,en;q=0.9",
    "Accept-Encoding": "gzip, deflate, br",
    "Connection": "keep-alive",
    "Referer": "https://www.shl.com/",
}


async def fetch_page(client: httpx.AsyncClient, url: str, params: dict = None, retries: int = MAX_RETRIES) -> str:
    """
    Fetch a URL with retry logic and exponential backoff.
    Returns the HTML string or raises RuntimeError after max retries.
    """
    for attempt in range(retries):
        try:
            response = await client.get(url, params=params)
            response.raise_for_status()
            return response.text
        except (httpx.HTTPStatusError, httpx.RequestError) as e:
            wait = 2 ** attempt
            logger.warning(f"Attempt {attempt + 1}/{retries} failed for {url}: {e}. Retrying in {wait}s...")
            if attempt < retries - 1:
                await asyncio.sleep(wait)
            else:
                raise RuntimeError(f"All {retries} retries failed for {url}") from e


def parse_catalog_page(html: str) -> list[dict]:
    """
    Parse one catalog listing page (Individual Test Solutions table only).
    Returns a list of preliminary assessment dicts.

    The page has 2 tables:
      Table 0: Pre-packaged Job Solutions
      Table 1: Individual Test Solutions  ← we want this one
    """
    soup = BeautifulSoup(html, "lxml")
    tables = soup.find_all("table")

    # Find the Individual Test Solutions table specifically
    target_table = None
    for table in tables:
        header_row = table.find("tr")
        if header_row:
            header_text = header_row.get_text()
            if "Individual Test Solutions" in header_text:
                target_table = table
                break

    if not target_table:
        logger.warning("Could not find Individual Test Solutions table on page")
        return []

    results = []
    rows = target_table.find_all("tr")

    for row in rows:
        # Skip header rows (no data-entity-id and no <a> in first td)
        cells = row.find_all("td")
        if len(cells) < 4:
            continue

        # Column 1: name and URL
        link = cells[0].find("a")
        if not link:
            continue

        name = link.get_text(strip=True)
        href = link.get("href", "").strip()
        if not name or not href:
            continue

        # Ensure absolute URL
        if href.startswith("/"):
            url = BASE_URL + href
        elif href.startswith("http"):
            url = href
        else:
            logger.warning(f"Unexpected href format: {href!r}, skipping")
            continue

        # Column 2: remote testing — presence of -yes span
        remote_testing = bool(cells[1].find("span", class_="-yes"))

        # Column 3: adaptive/IRT — presence of -yes span
        adaptive_irt = bool(cells[2].find("span", class_="-yes"))

        # Column 4: test type codes
        type_spans = cells[3].find_all("span", class_="product-catalogue__key")
        test_type = ",".join(s.get_text(strip=True) for s in type_spans if s.get_text(strip=True))

        results.append({
            "name": name,
            "url": url,
            "test_type": test_type,
            "remote_testing": remote_testing,
            "adaptive_irt": adaptive_irt,
        })

    return results


def get_total_count(html: str) -> int:
    """
    Try to extract total result count from catalog page.
    Falls back to a large default so pagination runs until empty pages.
    """
    soup = BeautifulSoup(html, "lxml")

    # Look for text patterns like "1-12 of 267" or "267 results"
    for text in soup.stripped_strings:
        match = re.search(r"of\s+(\d+)\s+results?", text, re.IGNORECASE)
        if match:
            return int(match.group(1))
        match = re.search(r"(\d+)\s+results?", text, re.IGNORECASE)
        if match:
            count = int(match.group(1))
            if count > 20:  # sanity check — avoid matching small numbers
                return count

    logger.info("Could not determine total count — will paginate until empty pages")
    return 9999  # large default


def _get_row_text(rows_by_label: dict, label: str) -> str:
    """Helper: get cleaned text from a detail row by its h4 label."""
    row = rows_by_label.get(label.lower())
    if not row:
        return ""
    p = row.find("p")
    if p:
        return p.get_text(separator=" ", strip=True)
    return row.get_text(separator=" ", strip=True)


def parse_detail_page(html: str, url: str) -> dict:
    """
    Parse a single assessment detail page.
    Returns a dict with: description, job_levels, job_families, languages, duration.

    All field extractions are wrapped in try/except — a single field failure
    must not crash the entire scrape.
    """
    soup = BeautifulSoup(html, "lxml")

    # Build index of rows by h4 label (lowercased)
    rows_by_label: dict[str, BeautifulSoup] = {}
    for row in soup.find_all("div", class_="product-catalogue-training-calendar__row"):
        h4 = row.find("h4")
        if h4:
            label = h4.get_text(strip=True).lower()
            rows_by_label[label] = row

    # --- Description ---
    description = ""
    try:
        row = rows_by_label.get("description")
        if row:
            p = row.find("p")
            description = p.get_text(separator=" ", strip=True) if p else ""
    except Exception as e:
        logger.warning(f"[{url}] Failed to parse description: {e}")

    # --- Job levels ---
    job_levels: list[str] = []
    try:
        text = _get_row_text(rows_by_label, "job levels")
        if text:
            job_levels = [lvl.strip() for lvl in text.split(",") if lvl.strip()]
    except Exception as e:
        logger.warning(f"[{url}] Failed to parse job_levels: {e}")

    # --- Job families ---
    job_families: list[str] = []
    try:
        # Try several possible labels
        for label in ("job families", "job family", "industries"):
            text = _get_row_text(rows_by_label, label)
            if text:
                job_families = [fam.strip() for fam in text.split(",") if fam.strip()]
                break
    except Exception as e:
        logger.warning(f"[{url}] Failed to parse job_families: {e}")

    # --- Languages ---
    languages: list[str] = []
    try:
        for label in ("languages", "language availability", "language"):
            text = _get_row_text(rows_by_label, label)
            if text:
                languages = [lang.strip() for lang in text.split(",") if lang.strip()]
                break
    except Exception as e:
        logger.warning(f"[{url}] Failed to parse languages: {e}")

    # --- Duration ---
    duration = ""
    try:
        for label in ("assessment length", "duration", "timing", "time"):
            row = rows_by_label.get(label)
            if row:
                p = row.find("p")
                raw = p.get_text(strip=True) if p else ""
                # Parse "Approximate Completion Time in minutes = 30"
                m = re.search(r"=\s*(\d+)", raw)
                if m:
                    duration = f"{m.group(1)} minutes"
                elif raw:
                    # Catch "Untimed", "Variable", etc.
                    duration = raw
                break
    except Exception as e:
        logger.warning(f"[{url}] Failed to parse duration: {e}")

    return {
        "description": description,
        "job_levels": job_levels,
        "job_families": job_families,
        "languages": languages,
        "duration": duration,
    }


def expand_test_type_label(test_type_code: str) -> str:
    """
    Expand raw test type code(s) to human-readable label.
    Examples:
        "A"   → "Ability & Aptitude"
        "P,C" → "Personality & Behavior, Competencies"
    """
    if not test_type_code:
        return ""
    codes = [c.strip() for c in test_type_code.split(",")]
    labels = [TEST_TYPE_LABELS.get(c, c) for c in codes if c]
    return ", ".join(labels)


async def scrape_catalog() -> list[dict]:
    """
    Main scrape orchestrator. Paginates through all Individual Test Solutions,
    fetches each detail page, and returns a merged list of assessment dicts.
    Saves a checkpoint every 50 items for resilience.
    """
    assessments = []
    seen_urls: set[str] = set()

    async with httpx.AsyncClient(headers=HEADERS, timeout=30, follow_redirects=True) as client:

        # Step 1: Fetch first page to get total count
        logger.info("Fetching first catalog page...")
        first_page_html = await fetch_page(
            client,
            CATALOG_URL,
            params={
                "action_doFilteringForm": "Search",
                "f": "1",
                "type": "1",
                "start": "0",
            },
        )
        total = get_total_count(first_page_html)
        logger.info(f"Total Individual Test Solutions to scrape: {total}")

        # Step 2: Paginate through all listing pages
        start = 0
        while True:
            logger.info(f"Scraping listing page: start={start}")

            if start > 0:
                html = await fetch_page(
                    client,
                    CATALOG_URL,
                    params={
                        "action_doFilteringForm": "Search",
                        "f": "1",
                        "type": "1",
                        "start": str(start),
                    },
                )
                time.sleep(REQUEST_DELAY)
            else:
                html = first_page_html

            items = parse_catalog_page(html)

            if not items:
                logger.info(f"Empty page at start={start}, stopping pagination")
                break

            # Step 3: For each item, fetch its detail page
            for item in items:
                if item["url"] in seen_urls:
                    logger.warning(f"Duplicate URL skipped: {item['url']}")
                    continue
                seen_urls.add(item["url"])

                logger.info(f"  Fetching detail: {item['name']}")
                try:
                    detail_html = await fetch_page(client, item["url"])
                    detail = parse_detail_page(detail_html, item["url"])
                    time.sleep(REQUEST_DELAY)
                except Exception as e:
                    logger.error(f"  Failed to fetch detail for {item['name']}: {e}")
                    detail = {
                        "description": "",
                        "job_levels": [],
                        "job_families": [],
                        "languages": [],
                        "duration": "",
                    }

                # Merge listing data + detail data
                assessment = {
                    **item,
                    **detail,
                    "test_type_label": expand_test_type_label(item["test_type"]),
                    "scraped_at": datetime.now(timezone.utc).isoformat(),
                }
                assessments.append(assessment)

                # Checkpoint every 50 items
                if len(assessments) % 50 == 0:
                    checkpoint_path = OUTPUT_PATH.with_suffix(".checkpoint.json")
                    checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
                    with open(checkpoint_path, "w", encoding="utf-8") as f:
                        json.dump(assessments, f, indent=2, ensure_ascii=False)
                    logger.info(f"Checkpoint saved: {len(assessments)} items")

            start += PAGE_SIZE

            # Safety: stop if we've exceeded expected total
            if start > total + PAGE_SIZE:
                logger.info("Exceeded expected total, stopping")
                break

    logger.info(f"Scrape complete. Total assessments: {len(assessments)}")
    return assessments


if __name__ == "__main__":
    assessments = asyncio.run(scrape_catalog())

    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(OUTPUT_PATH, "w", encoding="utf-8") as f:
        json.dump(assessments, f, indent=2, ensure_ascii=False)

    logger.info(f"Saved {len(assessments)} assessments to {OUTPUT_PATH}")

    # Clean up checkpoint if full scrape succeeded
    checkpoint = OUTPUT_PATH.with_suffix(".checkpoint.json")
    if checkpoint.exists():
        checkpoint.unlink()
        logger.info("Checkpoint file removed (full scrape successful)")
