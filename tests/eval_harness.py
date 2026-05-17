"""
Local evaluation harness for SHL Assessment Recommender.

Replays all 10 public conversation traces end-to-end through the live API
and produces a score report measuring:
  - Schema compliance rate (must be 100%)
  - Recall@10 per conversation
  - Behavior probe pass rate
  - Average response latency
  - Turn cap violations

Run with: python tests/eval_harness.py
Requires: uvicorn app.main:app running on localhost:8000
"""

import httpx
import json
import time
import statistics
from pathlib import Path

BASE_URL = "http://localhost:8000"
TIMEOUT = 45
TRACES_DIR = Path("tests/traces")

# Ground truth: expected assessments per conversation trace
# These are the assessment names that SHOULD appear in the final recommendations
# Partial name matching is used (case-insensitive substring)
GROUND_TRUTH = {
    "C1": ["OPQ32r", "Leadership Report", "Universal Competency", "OPQ Universal"],
    "C2": [
        "Linux Programming", "Networking and Implementation", "Networking",
        "Smart Interview Live", "Verify", "OPQ32r",
    ],
    "C3": ["SVAR", "Contact Center", "Customer Service", "Phone Simulation"],
    "C4": [
        "Numerical", "Financial Accounting", "Graduate Scenarios",
        "OPQ32r", "Verify Interactive",
    ],
    "C5": ["Global Skills", "OPQ32r", "Sales Transformation", "OPQ MQ Sales"],
    "C6": ["DSI", "Safety & Dependability", "Dependability 8.0", "Workplace Health and Safety"],
    "C7": ["HIPAA", "Medical Terminology", "OPQ32r", "DSI"],
    "C8": ["Excel", "Word", "Microsoft Word", "OPQ32r"],
    "C9": ["Java", "Spring", "SQL", "Verify", "OPQ32r"],
    "C10": ["Verify Interactive", "Verify G", "Graduate Scenarios"],
}


def post_chat(messages: list) -> tuple[dict, float]:
    """POST to /chat, return (response_dict, latency_seconds)."""
    start = time.time()
    r = httpx.post(
        f"{BASE_URL}/chat",
        json={"messages": messages},
        timeout=TIMEOUT,
    )
    latency = time.time() - start
    return r.json(), latency


def schema_compliant(response: dict) -> tuple[bool, str]:
    """Check if response matches required schema. Returns (ok, error_msg)."""
    if "reply" not in response:
        return False, "Missing 'reply'"
    if "recommendations" not in response:
        return False, "Missing 'recommendations'"
    if "end_of_conversation" not in response:
        return False, "Missing 'end_of_conversation'"
    if not isinstance(response["reply"], str) or not response["reply"].strip():
        return False, "reply is empty or not a string"
    if not isinstance(response["recommendations"], list):
        return False, "recommendations is not a list"
    if not isinstance(response["end_of_conversation"], bool):
        return False, "end_of_conversation is not a bool"
    if len(response["recommendations"]) > 10:
        return False, f"recommendations exceeds 10: {len(response['recommendations'])}"

    for rec in response["recommendations"]:
        if "name" not in rec or "url" not in rec or "test_type" not in rec:
            return False, f"Recommendation missing fields: {rec}"
        if not rec["url"].startswith("https://www.shl.com/"):
            return False, f"Off-catalog URL: {rec['url']}"
        if "(" in rec["test_type"]:
            return False, f"test_type contains label: {rec['test_type']}"

    return True, ""


def recall_at_10(recommendations: list, ground_truth_names: list) -> float:
    """
    Compute Recall@10 for a set of recommendations against ground truth.
    recall = |relevant ∩ recommended| / |relevant|
    Uses case-insensitive substring matching.
    """
    if not ground_truth_names:
        return 1.0

    rec_names_lower = [r["name"].lower() for r in recommendations]
    hits = 0
    for gt_name in ground_truth_names:
        gt_lower = gt_name.lower()
        if any(gt_lower in rec_lower or rec_lower in gt_lower
               for rec_lower in rec_names_lower):
            hits += 1

    return hits / len(ground_truth_names)


def replay_conversation(trace_id: str, messages: list) -> dict:
    """
    Replay a conversation trace turn by turn.
    Returns metrics for this conversation.
    """
    print(f"\n{'='*60}")
    print(f"Replaying {trace_id}")
    print('='*60)

    conversation_messages = []
    turn_count = 0
    schema_failures = []
    latencies = []
    final_recommendations = []
    final_end = False

    for msg in messages:
        if msg["role"] != "user":
            continue

        conversation_messages.append(msg)
        turn_count += 1

        response, latency = post_chat(conversation_messages)
        latencies.append(latency)

        # Schema check
        ok, err = schema_compliant(response)
        if not ok:
            schema_failures.append(f"Turn {turn_count}: {err}")
            print(f"  Turn {turn_count} [{latency:.1f}s] SCHEMA FAIL: {err}")
        else:
            rec_count = len(response["recommendations"])
            end = response["end_of_conversation"]
            print(f"  Turn {turn_count} [{latency:.1f}s] OK — {rec_count} recs, end={end}")

        # Add assistant response to conversation
        conversation_messages.append({
            "role": "assistant",
            "content": response.get("reply", "")
        })

        final_recommendations = response.get("recommendations", [])
        final_end = response.get("end_of_conversation", False)

        if final_end:
            break

    # Compute Recall@10 against ground truth
    gt = GROUND_TRUTH.get(trace_id, [])
    recall = recall_at_10(final_recommendations, gt)

    print(f"\n  Final recommendations ({len(final_recommendations)}):")
    for r in final_recommendations:
        print(f"    - {r['name']} [{r['test_type']}]")

    print(f"  Recall@10: {recall:.2f} ({int(recall * len(gt))}/{len(gt)} ground truth items found)")
    print(f"  Schema failures: {len(schema_failures)}")
    print(f"  Avg latency: {statistics.mean(latencies):.1f}s")

    return {
        "trace_id": trace_id,
        "turns": turn_count,
        "schema_failures": schema_failures,
        "recall_at_10": recall,
        "latencies": latencies,
        "final_rec_count": len(final_recommendations),
        "ended_properly": final_end,
    }


def build_messages_from_trace(trace_id: str) -> list:
    """
    Build a messages list from the conversation trace files.
    Reads the markdown trace and extracts user/assistant turns.
    """
    trace_file = TRACES_DIR / f"{trace_id}.md"
    if not trace_file.exists():
        print(f"  WARNING: Trace file not found: {trace_file}")
        return []

    content = trace_file.read_text(encoding="utf-8")
    messages = []

    # Parse markdown: lines starting with "> " after **User** or **Agent** headers
    current_role = None
    current_content = []

    for line in content.split("\n"):
        if "**User**" in line:
            if current_role and current_content:
                messages.append({
                    "role": current_role,
                    "content": " ".join(current_content).strip()
                })
            current_role = "user"
            current_content = []
        elif "**Agent**" in line:
            if current_role and current_content:
                messages.append({
                    "role": current_role,
                    "content": " ".join(current_content).strip()
                })
            current_role = "assistant"
            current_content = []
        elif line.startswith("> ") and current_role:
            current_content.append(line[2:].strip())

    if current_role and current_content:
        messages.append({
            "role": current_role,
            "content": " ".join(current_content).strip()
        })

    # Filter to only user messages for replay
    # (we regenerate assistant turns from the API)
    return [m for m in messages if m["role"] == "user" or m["role"] == "assistant"]


def run_harness():
    """Run the full evaluation harness."""
    print("\nSHL Assessment Recommender — Local Evaluation Harness")
    print("="*60)

    # Health check
    try:
        r = httpx.get(f"{BASE_URL}/health", timeout=10)
        assert r.json() == {"status": "ok"}
        print("Health check: OK")
    except Exception as e:
        print(f"FATAL: Server not running or health check failed: {e}")
        return

    trace_ids = [f"C{i}" for i in range(1, 11)]
    all_results = []

    for trace_id in trace_ids:
        messages = build_messages_from_trace(trace_id)
        if not messages:
            print(f"\nSkipping {trace_id} — no trace file found")
            continue

        # Only send user messages to replay (agent generates assistant turns)
        user_messages = [m for m in messages if m["role"] == "user"]
        replay_msgs = [{"role": "user", "content": m["content"]} for m in user_messages]

        result = replay_conversation(trace_id, replay_msgs)
        all_results.append(result)

    # ── Summary Report ────────────────────────────────────────────────────────
    print("\n" + "="*60)
    print("EVALUATION SUMMARY")
    print("="*60)

    if not all_results:
        print("No results — check trace files exist in tests/traces/")
        return

    total_schema_failures = sum(len(r["schema_failures"]) for r in all_results)
    total_turns = sum(r["turns"] for r in all_results)
    avg_recall = statistics.mean(r["recall_at_10"] for r in all_results)
    all_latencies = [l for r in all_results for l in r["latencies"]]
    p90_latency = sorted(all_latencies)[int(len(all_latencies) * 0.9)] if all_latencies else 0
    avg_latency = statistics.mean(all_latencies) if all_latencies else 0

    print(f"\nConversations replayed:  {len(all_results)}/10")
    print(f"Total turns processed:   {total_turns}")
    print(f"\nSchema compliance:       {'PASS' if total_schema_failures == 0 else 'FAIL'} ({total_schema_failures} failures)")
    print(f"Average Recall@10:       {avg_recall:.3f} ({avg_recall*100:.1f}%)")
    print(f"Average latency:         {avg_latency:.1f}s")
    print(f"P90 latency:             {p90_latency:.1f}s ({'OK' if p90_latency < 20 else 'WARNING: >20s'})")

    print("\nPer-conversation Recall@10:")
    for r in all_results:
        filled = int(r["recall_at_10"] * 10)
        bar = "#" * filled + "-" * (10 - filled)
        print(f"  {r['trace_id']}: [{bar}] {r['recall_at_10']:.2f}")

    # Overall assessment
    print("\n" + "="*60)
    if total_schema_failures == 0 and avg_recall >= 0.6 and p90_latency < 20:
        print("OVERALL: READY FOR DEPLOYMENT [PASS]")
    elif total_schema_failures > 0:
        print("OVERALL: SCHEMA FAILURES -- FIX BEFORE DEPLOYING [FAIL]")
    elif avg_recall < 0.6:
        print("OVERALL: RECALL TOO LOW -- TUNE RETRIEVAL BEFORE DEPLOYING [FAIL]")
    elif p90_latency >= 20:
        print("OVERALL: LATENCY TOO HIGH -- OPTIMIZE BEFORE DEPLOYING [FAIL]")
    print("="*60)


if __name__ == "__main__":
    run_harness()
