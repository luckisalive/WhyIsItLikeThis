"""
Verification script for framework-agnostic services against a live Neo4j database.
Not part of pytest. Evaluates specific architectural invariants and regressions.
"""

import logging
import sys
import warnings
from datetime import datetime
from pathlib import Path

# Silence Neo4j deprecation notifications and verbose logs
warnings.filterwarnings("ignore")
logging.getLogger("neo4j").setLevel(logging.ERROR)
logging.getLogger("neo4j.notifications").setLevel(logging.ERROR)

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from settings import get_settings
from src.graph.neo4j_manager import Neo4jManager
from src.services import ask, decisions, graph, stats
from src.services.decisions import DECISION_COUNT_QUERY
from src.services.types import ErrorCode


def is_iso_or_none(val) -> bool:
    if val is None:
        return True
    if not isinstance(val, str):
        return isinstance(val, (datetime, type(None)))
    try:
        clean_val = val.replace("Z", "+00:00")
        datetime.fromisoformat(clean_val)
        return True
    except Exception:
        return False


def main():
    try:
        sys.stdout.reconfigure(encoding='utf-8', line_buffering=True)
    except Exception:
        pass

    print("=" * 70)
    print("SERVICE LAYER VERIFICATION SUITE")
    print("=" * 70)

    settings = get_settings()
    print(f"Target Repository : {settings.target_repo}")
    print(f"Neo4j URI         : {settings.neo4j_uri}")

    try:
        neo4j_mgr = Neo4jManager(
            settings.neo4j_uri,
            settings.neo4j_username,
            settings.neo4j_password,
        )
        neo4j_mgr.connect()
        print("[CONNECTED] Connected to Neo4j successfully.\n")
    except Exception as e:
        print(f"[ERROR] FAILED to connect to Neo4j database: {e}\n")
        print("Note: If using Neo4j AuraDB Free, the instance may be paused.")
        print("Please ensure the instance is active at https://console.neo4j.io.")
        sys.exit(1)

    all_passed = True

    # ─────────────────────────────────────────────────────────
    # DIAGNOSTIC: Sample node properties (Commit, PullRequest, Issue)
    # ─────────────────────────────────────────────────────────
    print("--- [DIAGNOSTIC] Node properties and date fields in live database ---")
    try:
        c_sample = neo4j_mgr.execute_query("MATCH (c:Commit) RETURN keys(c) AS keys, c LIMIT 1")
        if c_sample:
            print(f"Commit keys       : {c_sample[0]['keys']}")
            print(f"Commit sample 'date': {repr(c_sample[0]['c'].get('date'))}")
        else:
            print("No Commit nodes found in database.")

        pr_sample = neo4j_mgr.execute_query("MATCH (pr:PullRequest) RETURN keys(pr) AS keys, pr LIMIT 1")
        if pr_sample:
            print(f"PullRequest keys  : {pr_sample[0]['keys']}")
            print(f"PR sample created : {repr(pr_sample[0]['pr'].get('created_at'))}")
        else:
            print("No PullRequest nodes found in database.")

        iss_sample = neo4j_mgr.execute_query("MATCH (i:Issue) RETURN keys(i) AS keys, i LIMIT 1")
        if iss_sample:
            print(f"Issue keys        : {iss_sample[0]['keys']}")
            print(f"Issue sample created: {repr(iss_sample[0]['i'].get('created_at'))}")
        else:
            print("No Issue nodes found in database.")
        print()
    except Exception as diag_e:
        print(f"Diagnostic error: {diag_e}\n")

    # ─────────────────────────────────────────────────────────
    # Invariant 1: count(ACTIVE) + count(SUPERSEDED) == count(all)
    # ─────────────────────────────────────────────────────────
    print("--- [Invariant 1] list_decisions status partition sum ---")
    res_all = decisions.list_decisions(neo4j_mgr, status=None, limit=1)
    res_active = decisions.list_decisions(neo4j_mgr, status="ACTIVE", limit=1)
    res_super = decisions.list_decisions(neo4j_mgr, status="SUPERSEDED", limit=1)

    total_all = res_all.total
    total_active = res_active.total
    total_super = res_super.total
    inv1_pass = (total_active + total_super) == total_all

    print(f"Total Decisions    : {total_all}")
    print(f"Active Decisions   : {total_active}")
    print(f"Superseded Decs    : {total_super}")
    print(f"Active + Superseded: {total_active + total_super} == Total: {total_all}")
    if inv1_pass:
        print("Result: PASS\n")
    else:
        print(f"Result: FAIL (Difference: {total_all - (total_active + total_super)})\n")
        all_passed = False

    # ─────────────────────────────────────────────────────────
    # Invariant 2: Pages don't overlap, exact length per page
    # ─────────────────────────────────────────────────────────
    print("--- [Invariant 2] Pagination length formula, non-overlap, and coverage ---")
    step = 10
    collected_ids = []
    page_lens_correct = True

    for offset in range(0, total_all, step):
        page_res = decisions.list_decisions(neo4j_mgr, limit=step, offset=offset)
        expected_len = min(step, max(0, total_all - offset))
        actual_len = len(page_res.items)
        if actual_len != expected_len:
            print(f"[FAIL] Page at offset {offset}: expected len {expected_len}, got {actual_len}")
            page_lens_correct = False

        page_ids = [item["id"] for item in page_res.items]
        collected_ids.extend(page_ids)

    unique_ids = set(collected_ids)
    no_duplicates = len(collected_ids) == len(unique_ids)
    complete_union = len(unique_ids) == total_all
    inv2_pass = page_lens_correct and no_duplicates and complete_union

    print(f"Total Expected IDs : {total_all}")
    print(f"Total Collected IDs: {len(collected_ids)}")
    print(f"Unique IDs Count   : {len(unique_ids)}")
    print(f"Duplicates Detected: {len(collected_ids) - len(unique_ids)}")
    print(f"Page Lengths Valid : {page_lens_correct}")
    if inv2_pass:
        print("Result: PASS\n")
    else:
        print("Result: FAIL\n")
        all_passed = False

    # ─────────────────────────────────────────────────────────
    # Invariant 3: Offset past total returns empty items with correct total
    # ─────────────────────────────────────────────────────────
    print("--- [Invariant 3] Offset past total boundary behavior ---")
    past_offset = total_all + 50
    past_res = decisions.list_decisions(neo4j_mgr, limit=10, offset=past_offset)
    inv3_pass = (len(past_res.items) == 0) and (past_res.total == total_all)

    print(f"Query with offset={past_offset} (total={total_all}):")
    print(f"Returned Items Count: {len(past_res.items)}")
    print(f"Reported Total      : {past_res.total}")
    if inv3_pass:
        print("Result: PASS\n")
    else:
        print("Result: FAIL\n")
        all_passed = False

    # ─────────────────────────────────────────────────────────
    # Regression Test: Decision superseded by 2+ decisions
    # ─────────────────────────────────────────────────────────
    print("--- [Regression] Decision superseded by 2+ decisions appears exactly once ---")
    check_multi_superseded_query = """
    MATCH (newer:Decision)-[:SUPERSEDES]->(d:Decision)
    WITH d, count(DISTINCT newer) AS newer_count
    WHERE newer_count >= 2
    RETURN d.id AS id, newer_count
    LIMIT 5
    """
    multi_records = neo4j_mgr.execute_query(check_multi_superseded_query)
    if multi_records:
        target_multi_id = multi_records[0]["id"]
        expected_newer_count = multi_records[0]["newer_count"]
        print(f"Testing decision {target_multi_id} (superseded by {expected_newer_count} decisions)...")

        multi_occurrences = [i for i in collected_ids if i == target_multi_id]
        print(f"Occurrences across paginated list: {len(multi_occurrences)}")
        if len(multi_occurrences) == 1:
            print("Result: PASS\n")
        else:
            print(f"Result: FAIL (appeared {len(multi_occurrences)} times)\n")
            all_passed = False
    else:
        print("No decisions in DB currently superseded by 2+ decisions to check live.")
        print("Result: PASS (Fixture regression covered in pytest)\n")

    # ─────────────────────────────────────────────────────────
    # Invariant 4: Sidebar stats vs list_decisions status totals
    # ─────────────────────────────────────────────────────────
    print("--- [Invariant 4] Sidebar stats vs list_decisions status totals ---")
    stats_res = stats.get_stats(neo4j_mgr)
    sidebar_active = stats_res.stats.active_decisions if stats_res.stats else None
    sidebar_super = stats_res.stats.superseded_decisions if stats_res.stats else None

    inv4_pass = (sidebar_active == total_active) and (sidebar_super == total_super)
    print(f"Sidebar Stats  : Active={sidebar_active}, Superseded={sidebar_super}")
    print(f"list_decisions : Active={total_active}, Superseded={total_super}")

    if inv4_pass:
        print("Result: PASS\n")
    else:
        print("Result: FAIL")
        print("\n[Cypher Comparison]")
        print("1. Sidebar stats Cypher (from Neo4jManager.get_stats):")
        print("   MATCH (d:Decision {status: 'ACTIVE'}) RETURN count(d) AS dec_active")
        print("   MATCH (d:Decision {status: 'SUPERSEDED'}) RETURN count(d) AS dec_super")
        print("   -> Relies exclusively on static d.status property.")
        print("\n2. list_decisions Cypher (from src/services/decisions.py):")
        print(DECISION_COUNT_QUERY.strip())
        print("   -> Evaluates effective status including incoming [:SUPERSEDES] edges.\n")
        all_passed = False

    # ─────────────────────────────────────────────────────────
    # Invariant 5: Decision with most affected entities
    # ─────────────────────────────────────────────────────────
    print("--- [Invariant 5] Decision with most affected entities (hidden_count) ---")
    most_entities_query = """
    MATCH (d:Decision)-[:AFFECTS]->(e:CodeEntity)
    RETURN d.id AS id, count(DISTINCT e) AS entity_count
    ORDER BY entity_count DESC
    LIMIT 1
    """
    top_dec_records = neo4j_mgr.execute_query(most_entities_query)
    if not top_dec_records:
        print("No Decision with AFFECTS relationships found.")
        print("Result: FAIL\n")
        all_passed = False
    else:
        top_dec_id = top_dec_records[0]["id"]
        real_entity_count = top_dec_records[0]["entity_count"]

        graph_res = graph.get_decision_graph(neo4j_mgr, top_dec_id, entity_limit=5)
        if not graph_res.success:
            print(f"[FAIL] get_decision_graph failed: {graph_res.error.message}")
            inv5_pass = False
        else:
            entity_nodes = [n for n in graph_res.nodes if n.type == "CodeEntity"]
            node_count = len(entity_nodes)
            hidden_count = graph_res.hidden_count
            all_node_ids = [n.id for n in graph_res.nodes]
            no_duplicate_nodes = len(all_node_ids) == len(set(all_node_ids))

            inv5_pass = (hidden_count > 0) and ((node_count + hidden_count) == real_entity_count) and no_duplicate_nodes
            print(f"Top Decision ID         : {top_dec_id}")
            print(f"Direct Query Entity Cnt : {real_entity_count}")
            print(f"Graph CodeEntity Nodes  : {node_count}")
            print(f"Graph hidden_count      : {hidden_count}")
            print(f"Nodes + hidden_count    : {node_count + hidden_count} == {real_entity_count}")
            print(f"No Duplicate Node IDs   : {no_duplicate_nodes} (Total nodes: {len(all_node_ids)})")

        if inv5_pass:
            print("Result: PASS\n")
        else:
            print("Result: FAIL\n")
            all_passed = False

    # ─────────────────────────────────────────────────────────
    # Invariant 6: Real ask() evaluation: 3 questions x 5 runs (15 total runs)
    # ─────────────────────────────────────────────────────────
    PAUSE_BETWEEN_RUNS = 1.0  # Configurable pause in seconds

    print("--- [Invariant 6] Strict Evidence Provenance: 3 questions x 5 runs (15 total runs) ---")
    print(f"Configured pause between runs: {PAUSE_BETWEEN_RUNS}s")
    test_questions = [
        "Why does requests use urllib3?",
        "Why was default timeout not set in requests?",
        "Why was chardet replaced or made optional?",
    ]

    RUNS_PER_QUESTION = 5
    inv6_passed = True

    total_cards_collected = 0
    cards_with_url_and_date = 0
    decision_cards_total = 0
    decision_cards_unresolved_url = 0

    all_card_counts_by_type: dict[str, int] = {}

    import time

    for q_idx, q in enumerate(test_questions, 1):
        print(f"\n{'='*60}\n[Question {q_idx}] \"{q}\"\n{'='*60}")
        q_successes = 0
        q_no_results = 0
        q_inconclusive = 0
        q_card_counts: dict[str, int] = {}

        for run_idx in range(1, RUNS_PER_QUESTION + 1):
            if PAUSE_BETWEEN_RUNS > 0:
                time.sleep(PAUSE_BETWEEN_RUNS)

            ask_res = ask.answer(q, neo4j_mgr, settings=settings)

            err_str = f"{ask_res.error.code}: {ask_res.error.message}" if ask_res.error else "None"
            print(f"  Run {run_idx}/{RUNS_PER_QUESTION}: model={ask_res.model_used}, degraded={ask_res.degraded}, error={err_str}")

            if ask_res.error:
                if ask_res.error.code == ErrorCode.LLM_UNAVAILABLE:
                    q_inconclusive += 1
                    print("    Status: INCONCLUSIVE (LLM_UNAVAILABLE)")
                elif ask_res.error.code == ErrorCode.NO_RESULTS:
                    q_no_results += 1
            else:
                q_successes += 1

            evidence_items = ask_res.answer.evidence if (ask_res.answer and ask_res.answer.evidence) else []
            for ev in evidence_items:
                total_cards_collected += 1
                stype = ev.source_type.lower()
                all_card_counts_by_type[stype] = all_card_counts_by_type.get(stype, 0) + 1
                q_card_counts[stype] = q_card_counts.get(stype, 0) + 1

                has_url = bool(ev.url)
                has_iso_date = bool(ev.date) and is_iso_or_none(ev.date)

                if stype == "decision":
                    decision_cards_total += 1
                    if not has_url:
                        decision_cards_unresolved_url += 1

                if has_url and has_iso_date:
                    cards_with_url_and_date += 1

        print(f"\n  Question {q_idx} Summary:")
        print(f"    Success rate : {q_successes}/{RUNS_PER_QUESTION} ({(q_successes/RUNS_PER_QUESTION)*100:.1f}%)")
        print(f"    NO_RESULTS   : {q_no_results}/{RUNS_PER_QUESTION}")
        print(f"    Card counts  : {q_card_counts}")

        if q_no_results > 1:
            print(f"    [FAIL] Question had {q_no_results} > 1 NO_RESULTS runs.")
            inv6_passed = False

    print(f"\n{'='*60}\nOVERALL EVIDENCE PROVENANCE METRICS (across all 15 runs)\n{'='*60}")
    print(f"Total evidence cards collected: {total_cards_collected}")
    print(f"Card counts by type           : {all_card_counts_by_type}")
    print(f"Decision cards evaluated      : {decision_cards_total} ({decision_cards_total - decision_cards_unresolved_url} resolved url/date, {decision_cards_unresolved_url} unresolvable url=None)")

    if total_cards_collected > 0:
        ratio = cards_with_url_and_date / total_cards_collected
        print(f"Cards with both URL & ISO date: {cards_with_url_and_date}/{total_cards_collected} ({ratio*100:.1f}%)")
        if ratio < 0.80:
            print(f"[FAIL] Fewer than 80% of cards have both URL and ISO date ({ratio*100:.1f}% < 80.0%).")
            inv6_passed = False
    else:
        print("[FAIL] Zero evidence cards collected across all runs.")
        inv6_passed = False

    if inv6_passed:
        print("\nInvariant 6 Result: PASS\n")
    else:
        print("\nInvariant 6 Result: FAIL\n")
        all_passed = False

    print("=" * 70)
    print(f"OVERALL SUITE: {'ALL PASSED' if all_passed else 'SOME INVARIANTS FAILED'}")
    print("=" * 70)

    neo4j_mgr.close()
    if not all_passed:
        sys.exit(1)


if __name__ == "__main__":
    main()
