"""
SHL Catalog Validator
======================
Validates data/catalog.json and prints a detailed pass/fail report.

Usage:
    python scraper/validate_catalog.py

Exit codes:
    0 — PASS or WARN
    1 — FAIL or file not found/invalid JSON
"""

import json
import sys
from collections import Counter
from pathlib import Path

CATALOG_PATH = Path("data/catalog.json")
VALID_URL_PREFIX = "https://www.shl.com/products/product-catalog/view/"
VALID_TYPE_CODES = {"A", "P", "K", "B", "S", "C", "D", "E"}
REQUIRED_FIELDS = ["name", "url", "test_type"]
MIN_COUNT = 100


def header(text: str):
    print(f"\n{'=' * 60}")
    print(f"  {text}")
    print(f"{'=' * 60}")


def check(label: str, passed: bool, detail: str = ""):
    status = "PASS" if passed else "FAIL"
    sep = " -- " if detail else ""
    print(f"  [{status}] {label}{sep}{detail}")
    return passed


def main():
    all_passed = True
    warnings = []

    header("SHL Catalog Validator")

    # ─── 1. File exists and is valid JSON ─────────────────────────────────────
    header("1. File Check")

    if not CATALOG_PATH.exists():
        print(f"  [✗ FAIL] {CATALOG_PATH} not found. Run the scraper first.")
        sys.exit(1)

    try:
        with open(CATALOG_PATH, encoding="utf-8") as f:
            catalog = json.load(f)
        check("Valid JSON", True, f"{CATALOG_PATH}")
    except json.JSONDecodeError as e:
        check("Valid JSON", False, str(e))
        sys.exit(1)

    if not isinstance(catalog, list):
        print("  [✗ FAIL] catalog.json must be a JSON array")
        sys.exit(1)

    total = len(catalog)
    print(f"  Total assessments loaded: {total}")

    # ─── 2. Minimum count ─────────────────────────────────────────────────────
    header("2. Minimum Count")

    count_ok = check(
        f"At least {MIN_COUNT} assessments",
        total >= MIN_COUNT,
        f"{total} found",
    )
    if not count_ok:
        warnings.append(f"Only {total} assessments — expected {MIN_COUNT}+. Scrape may be incomplete.")
        # Don't fail — just warn
    if total < 200:
        warnings.append(f"Only {total} assessments — real catalog has 250+. Consider re-running scraper.")

    # ─── 3. Required fields ────────────────────────────────────────────────────
    header("3. Required Fields")

    field_errors: list[str] = []
    for i, item in enumerate(catalog):
        for field in REQUIRED_FIELDS:
            val = item.get(field)
            if not val or (isinstance(val, str) and not val.strip()):
                field_errors.append(f"  Item #{i} ({item.get('name', 'UNNAMED')}): missing/empty '{field}'")

    req_ok = check(
        "All required fields present",
        len(field_errors) == 0,
        f"{len(field_errors)} errors" if field_errors else "all good",
    )
    if field_errors:
        for err in field_errors[:10]:
            print(err)
        if len(field_errors) > 10:
            print(f"  ... and {len(field_errors) - 10} more")
        all_passed = False

    # ─── 4. URL format validation ──────────────────────────────────────────────
    header("4. URL Validation")

    invalid_urls: list[str] = []
    duplicate_urls: list[str] = []
    seen_urls: set[str] = set()

    for item in catalog:
        url = item.get("url", "")
        if not url.startswith(VALID_URL_PREFIX):
            invalid_urls.append(url)
        if url in seen_urls:
            duplicate_urls.append(url)
        seen_urls.add(url)

    url_format_ok = check(
        "All URLs start with correct prefix",
        len(invalid_urls) == 0,
        f"{len(invalid_urls)} invalid" if invalid_urls else "all valid",
    )
    if invalid_urls:
        for u in invalid_urls[:5]:
            print(f"    BAD URL: {u}")
        if len(invalid_urls) > 5:
            print(f"    ... and {len(invalid_urls) - 5} more")
        all_passed = False

    dedup_ok = check(
        "No duplicate URLs",
        len(duplicate_urls) == 0,
        f"{len(duplicate_urls)} duplicates" if duplicate_urls else "all unique",
    )
    if duplicate_urls:
        for u in duplicate_urls[:5]:
            print(f"    DUPLICATE: {u}")
        all_passed = False

    # ─── 5. Test type validation ───────────────────────────────────────────────
    header("5. Test Type Validation")

    unknown_codes: list[str] = []
    for item in catalog:
        tt = item.get("test_type", "")
        codes = [c.strip() for c in tt.split(",") if c.strip()]
        for code in codes:
            if code not in VALID_TYPE_CODES:
                unknown_codes.append(f"{item.get('name', '?')}: '{code}'")

    type_ok = check(
        "All test_type codes are known",
        len(unknown_codes) == 0,
        f"{len(unknown_codes)} unknown codes" if unknown_codes else "all valid",
    )
    if unknown_codes:
        for u in unknown_codes[:10]:
            print(f"    UNKNOWN CODE: {u}")
        # Warn but don't fail — SHL may add new codes
        warnings.append(f"{len(unknown_codes)} unknown test type codes found")

    # ─── 6. Data quality stats ─────────────────────────────────────────────────
    header("6. Data Quality Statistics")

    def pct(count, total):
        return f"{count} ({count / total * 100:.0f}%)" if total else "0 (0%)"

    with_desc = sum(1 for x in catalog if x.get("description"))
    with_levels = sum(1 for x in catalog if x.get("job_levels"))
    with_families = sum(1 for x in catalog if x.get("job_families"))
    with_langs = sum(1 for x in catalog if x.get("languages"))
    with_duration = sum(1 for x in catalog if x.get("duration"))
    with_remote = sum(1 for x in catalog if x.get("remote_testing"))
    with_adaptive = sum(1 for x in catalog if x.get("adaptive_irt"))

    print(f"  Total assessments:              {total}")
    print(f"  With description:               {pct(with_desc, total)}")
    print(f"  With job_levels:                {pct(with_levels, total)}")
    print(f"  With job_families:              {pct(with_families, total)}")
    print(f"  With languages:                 {pct(with_langs, total)}")
    print(f"  With duration:                  {pct(with_duration, total)}")
    print(f"  Remote testing enabled:         {pct(with_remote, total)}")
    print(f"  Adaptive/IRT enabled:           {pct(with_adaptive, total)}")

    # Quality warnings for optional fields below 50%
    for field, count, label in [
        ("description", with_desc, "description"),
        ("languages", with_langs, "languages"),
    ]:
        if total > 0 and count / total < 0.5:
            warnings.append(f"Low {label} coverage: {pct(count, total)}")

    # Test type distribution
    print("\n  Test type distribution:")
    type_counter: Counter = Counter()
    multi_count = 0
    for item in catalog:
        tt = item.get("test_type", "")
        codes = [c.strip() for c in tt.split(",") if c.strip()]
        if len(codes) > 1:
            multi_count += 1
        for code in codes:
            type_counter[code] += 1

    _TYPE_LABELS = {
        "A": "Ability & Aptitude",
        "P": "Personality & Behavior",
        "K": "Knowledge & Skills",
        "B": "Biodata & Situational Judgment",
        "S": "Simulations",
        "C": "Competencies",
        "D": "Development & 360",
        "E": "Engagement",
    }
    for code in sorted(type_counter.keys()):
        label = _TYPE_LABELS.get(code, code)
        print(f"    {code} ({label}): {type_counter[code]}")
    print(f"    Multi-type: {multi_count}")

    # ─── 7. Sanity checks ─────────────────────────────────────────────────────
    header("7. Sanity Checks (Known Assessments)")

    names = [item.get("name", "").lower() for item in catalog]
    urls_list = [item.get("url", "") for item in catalog]

    opq_found = any("opq" in n for n in names)
    check("OPQ (Occupational Personality Questionnaire) present", opq_found)
    if not opq_found:
        warnings.append("OPQ assessment not found — may indicate incomplete scrape")

    # The catalog URL uses different slug naming so check by partial URL
    verify_found = any("verify" in u for u in urls_list)
    check("Verify assessment present", verify_found)
    if not verify_found:
        warnings.append("Verify assessment not found — may indicate incomplete scrape")

    # ─── Final verdict ─────────────────────────────────────────────────────────
    header("Final Verdict")

    if warnings:
        print("  Warnings:")
        for w in warnings:
            print(f"    [WARN] {w}")

    if all_passed:
        if warnings:
            print("\n  [WARN] Catalog is usable but has quality issues (see warnings above)")
        else:
            print("\n  [PASS] Catalog looks good!")
        sys.exit(0)
    else:
        print("\n  [FAIL] Catalog has structural errors that must be fixed")
        sys.exit(1)


if __name__ == "__main__":
    main()
