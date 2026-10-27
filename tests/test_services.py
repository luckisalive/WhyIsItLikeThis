"""
Pytest suite for framework-agnostic service layer.
Tests error mapping, deterministic ordering/pagination Cypher assertion,
full subgraph construction with hidden_count, and evidence date provenance.
"""

from unittest.mock import MagicMock

import pytest

from src.models import Answer, Confidence, EvidenceItem, GraphStats
from src.services import ask, decisions, graph, stats
from src.services.decisions import DECISION_COUNT_QUERY, DECISION_PAGE_QUERY
from src.services.types import ErrorCode


# ─────────────────────────────────────────────────────────────
# 1. Error Handling Tests
# ─────────────────────────────────────────────────────────────


def test_stats_db_unreachable_when_neo4j_fails():
    """Verify stats returns DB_UNREACHABLE without raising."""
    mock_mgr = MagicMock()
    mock_mgr.get_stats.side_effect = Exception("Neo4j driver connection failed")

    result = stats.get_stats(mock_mgr)

    assert result.success is False
    assert result.error is not None
    assert result.error.code == ErrorCode.DB_UNREACHABLE
    assert "Neo4j driver connection failed" in result.error.message


def test_stats_db_unreachable_when_neo4j_is_none():
    """Verify stats returns DB_UNREACHABLE when manager is None."""
    result = stats.get_stats(None)

    assert result.success is False
    assert result.error is not None
    assert result.error.code == ErrorCode.DB_UNREACHABLE


def test_decisions_db_unreachable_when_query_fails():
    """Verify list_decisions returns DB_UNREACHABLE when query crashes."""
    mock_mgr = MagicMock()
    mock_mgr.execute_query.side_effect = Exception("Query execution error: Session expired")

    result = decisions.list_decisions(mock_mgr)

    assert result.success is False
    assert result.error is not None
    assert result.error.code == ErrorCode.DB_UNREACHABLE


def test_graph_no_results_when_decision_not_found():
    """Verify get_decision_graph returns NO_RESULTS when decision doesn't exist."""
    mock_mgr = MagicMock()
    mock_mgr.get_decision_with_context.return_value = {}

    result = graph.get_decision_graph(mock_mgr, "DEC-NONEXISTENT")

    assert result.success is False
    assert result.error is not None
    assert result.error.code == ErrorCode.NO_RESULTS
    assert "DEC-NONEXISTENT" in result.error.message


def test_graph_driver_exception_does_not_become_no_results():
    """Verify that a driver exception during get_decision_graph does NOT become NO_RESULTS."""
    mock_mgr = MagicMock()
    mock_mgr.get_decision_with_context.side_effect = Exception("Neo4j driver pool defunct or failed query")

    result = graph.get_decision_graph(mock_mgr, "DEC-PR-7497")

    assert result.success is False
    assert result.error is not None
    # Must NOT be NO_RESULTS
    assert result.error.code != ErrorCode.NO_RESULTS
    assert result.error.code in (ErrorCode.DB_UNREACHABLE, ErrorCode.UNKNOWN)


def test_ask_no_results_when_retriever_finds_nothing():
    """Verify ask returns NO_RESULTS when graph has no matching context."""
    mock_mgr = MagicMock()
    mock_engine = MagicMock()
    mock_retriever = MagicMock()
    mock_retriever.retrieve.return_value = []
    mock_engine.retriever = mock_retriever
    mock_engine._build_insufficient_evidence_answer.return_value = Answer(
        question="Why?",
        answer_text="No evidence found.",
        confidence=Confidence.LOW,
    )

    result = ask.answer("Why?", mock_mgr, answer_engine=mock_engine)

    assert result.success is False
    assert result.error is not None
    assert result.error.code == ErrorCode.NO_RESULTS
    assert result.answer is not None
    assert result.answer.answer_text == "No evidence found."


def test_ask_llm_unavailable_when_all_models_exhausted_and_heuristic_bypassed():
    """Verify ask returns LLM_UNAVAILABLE when LLMs exhausted and heuristic fallback disabled."""
    mock_mgr = MagicMock()
    mock_engine = MagicMock()
    mock_retriever = MagicMock()
    mock_retriever.retrieve.return_value = [{"decision": {"id": "DEC-1"}}]
    mock_engine.retriever = mock_retriever
    mock_engine.all_models_exhausted = True

    result = ask.answer(
        "Why is it like this?",
        mock_mgr,
        answer_engine=mock_engine,
        allow_heuristic=False,
    )

    assert result.success is False
    assert result.error is not None
    assert result.error.code == ErrorCode.LLM_UNAVAILABLE


# ─────────────────────────────────────────────────────────────
# 2. Decisions Pagination, Ordering & Cypher Assertions
# ─────────────────────────────────────────────────────────────


def test_list_decisions_exact_cypher_and_parameters():
    """Assert on exact Cypher and parameters for ordering, pagination, and filters."""
    mock_mgr = MagicMock()
    mock_mgr.execute_query.side_effect = [
        [{"total": 42}],
        [],
    ]

    result = decisions.list_decisions(
        mock_mgr,
        status="ACTIVE",
        type="ARCHITECTURE",
        limit=10,
        offset=20,
    )

    assert result.success is True
    assert result.total == 42
    assert mock_mgr.execute_query.call_count == 2

    # Verify Count Query call
    count_call = mock_mgr.execute_query.call_args_list[0]
    count_query, count_params = count_call[0]
    assert count_query.strip() == DECISION_COUNT_QUERY.strip()
    assert count_params == {
        "status": "ACTIVE",
        "type": "ARCHITECTURE",
        "limit": 10,
        "offset": 20,
    }

    # Verify Page Query call with deterministic ORDER BY
    page_call = mock_mgr.execute_query.call_args_list[1]
    page_query, page_params = page_call[0]
    assert page_query.strip() == DECISION_PAGE_QUERY.strip()
    assert "ORDER BY d.id ASC, d.title ASC" in page_query
    assert "SKIP $offset" in page_query
    assert "LIMIT $limit" in page_query
    assert page_params == {
        "status": "ACTIVE",
        "type": "ARCHITECTURE",
        "limit": 10,
        "offset": 20,
    }


def test_list_decisions_shaping_and_status_consistency():
    """Verify decision shaping and unified status source of truth."""
    mock_mgr = MagicMock()
    fixture_records = [
        # Decision 1: status ACTIVE, no newer -> ACTIVE
        {
            "d": {"id": "DEC-01", "title": "Base Model", "status": "ACTIVE", "rationale": "Simplicity"},
            "superseded_by": [],
        },
        # Decision 2: status ACTIVE in property, but newer exists -> SUPERSEDED
        {
            "d": {"id": "DEC-02", "title": "Old Routing", "status": "ACTIVE", "rationale": "Fast"},
            "superseded_by": [{"id": "DEC-03", "title": "New Routing"}],
        },
    ]
    mock_mgr.execute_query.side_effect = [
        [{"total": 2}],
        fixture_records,
    ]

    result = decisions.list_decisions(mock_mgr, limit=5, offset=0)

    assert result.success is True
    assert result.total == 2
    assert len(result.items) == 2

    item1 = result.items[0]
    assert item1["id"] == "DEC-01"
    assert item1["status"] == "ACTIVE"
    assert item1["is_superseded"] is False
    assert item1["newer"] is None

    item2 = result.items[1]
    assert item2["id"] == "DEC-02"
    assert item2["status"] == "SUPERSEDED"  # Determined by newer supersedes edge
    assert item2["is_superseded"] is True
    assert item2["newer"]["id"] == "DEC-03"


def test_list_decisions_superseded_by_multiple_decisions_regression():
    """
    Regression test for BUG A:
    A decision superseded by 2+ decisions must appear exactly once in items
    with superseded_by as a list.
    """
    mock_mgr = MagicMock()
    fixture_records = [
        {
            "d": {"id": "DEC-01", "title": "Original Config", "status": "SUPERSEDED"},
            "superseded_by": [
                {"id": "DEC-02", "title": "First Override"},
                {"id": "DEC-03", "title": "Second Override"},
            ],
        }
    ]
    mock_mgr.execute_query.side_effect = [
        [{"total": 1}],
        fixture_records,
    ]

    result = decisions.list_decisions(mock_mgr, limit=10, offset=0)

    assert result.success is True
    assert result.total == 1
    # Exactly one item, not duplicated
    assert len(result.items) == 1
    item = result.items[0]
    assert item["id"] == "DEC-01"
    assert len(item["superseded_by"]) == 2
    assert item["newer"]["id"] == "DEC-02"
    assert item["is_superseded"] is True


def test_list_decisions_offset_past_total():
    """Verify offset past total returns empty items with total preserved and no error."""
    mock_mgr = MagicMock()
    mock_mgr.execute_query.return_value = [{"total": 15}]

    result = decisions.list_decisions(mock_mgr, limit=10, offset=20)

    assert result.success is True
    assert result.error is None
    assert result.total == 15
    assert len(result.items) == 0


# ─────────────────────────────────────────────────────────────
# 3. Subgraph Construction & Hidden Count
# ─────────────────────────────────────────────────────────────


def test_get_decision_graph_full_neighborhood_and_hidden_count():
    """Verify Issue, Person, both older & newer decisions, and entity hidden_count."""
    mock_mgr = MagicMock()
    fixture_context = {
        "decision": {"id": "DEC-42", "title": "Use Starlette", "status": "ACTIVE"},
        "pr": {"number": 101, "title": "Add Starlette backend"},
        "commit": {"sha": "a1b2c3d4e5f6", "message": "feat: starlette integration"},
        "issue": {"number": 55, "title": "Async ASGI framework selection"},
        "author": {"login": "tiangolo", "name": "Sebastián Ramírez"},
        "entities": [
            {"path": f"src/core/mod_{i}.py", "name": f"mod_{i}"}
            for i in range(1, 9)  # 8 entities
        ],
        "superseded": [{"id": "DEC-10", "title": "Flask routing"}],
        "superseded_by": [{"id": "DEC-90", "title": "Next-gen ASGI"}],
    }
    mock_mgr.get_decision_with_context.return_value = fixture_context

    result = graph.get_decision_graph(mock_mgr, "DEC-42", entity_limit=5)

    assert result.success is True
    assert result.error is None
    assert mock_mgr.get_decision_with_context.call_count == 1  # No N+1

    # Verify node types present
    node_types = {n.type for n in result.nodes}
    assert "Decision" in node_types
    assert "PullRequest" in node_types
    assert "Commit" in node_types
    assert "Issue" in node_types
    assert "Person" in node_types
    assert "CodeEntity" in node_types

    # Verify hidden count
    code_entities = [n for n in result.nodes if n.type == "CodeEntity"]
    assert len(code_entities) == 5
    assert result.hidden_count == 3  # 8 total - 5 limit = 3 hidden

    # Verify real_id preserved in properties
    node_map = {n.id: n for n in result.nodes}
    assert node_map["DEC-42"].properties["real_id"] == "DEC-42"
    assert node_map["PR-101"].properties["real_id"] == "101"
    assert node_map["COMMIT-a1b2c3d4e5f6"].properties["real_id"] == "a1b2c3d4e5f6"
    assert node_map["ISSUE-55"].properties["real_id"] == "55"
    assert node_map["PERSON-tiangolo"].properties["real_id"] == "tiangolo"

    # Verify older & newer superseding decisions
    assert "DEC-10" in node_map
    assert node_map["DEC-10"].label == "Old: Flask routing"
    assert "DEC-90" in node_map
    assert node_map["DEC-90"].label == "New: Next-gen ASGI"

    # Verify edge labels
    edge_labels = {e.label for e in result.edges}
    assert "DISCUSSED_IN" in edge_labels
    assert "IMPLEMENTED_IN" in edge_labels
    assert "MOTIVATED_BY" in edge_labels
    assert "MADE_BY" in edge_labels
    assert "AFFECTS" in edge_labels
    assert "SUPERSEDES" in edge_labels


# ─────────────────────────────────────────────────────────────
# 4. Ask Evidence Provenance, Catalog Hydration, & Marker Rewriting
# ─────────────────────────────────────────────────────────────


def _load_captured_fixture():
    import json
    from pathlib import Path
    fixture_path = Path(__file__).resolve().parent / "fixtures" / "retrieval_results.json"
    with open(fixture_path, encoding="utf-8") as f:
        return json.load(f)


def test_ask_candidate_hydration_from_captured_fixture():
    """Verify candidate catalog hydrates cards strictly from database fields."""
    fixture = _load_captured_fixture()
    catalog, alias_map = ask._build_candidate_catalog(fixture, target_repo="psf/requests")

    # Verify PR-7379
    assert "PR-7379" in catalog
    pr_item = catalog["PR-7379"]
    assert pr_item.source_type == "pull_request"
    assert pr_item.source_id == "7379"
    assert "clarify timeout parameter" in pr_item.title
    assert pr_item.url == "https://github.com/psf/requests/pull/7379"
    assert pr_item.date == "2026-04-20T19:29:26Z"
    assert pr_item.person is None  # Author = None where not stored

    # Verify DEC-PR-7379
    assert "DEC-PR-7379" in catalog
    dec_item = catalog["DEC-PR-7379"]
    assert dec_item.source_type == "decision"
    assert dec_item.source_id == "DEC-PR-7379"
    assert "timeout" in dec_item.title.lower()
    assert dec_item.url == "https://github.com/psf/requests/pull/7379"
    assert dec_item.date == "2026-04-20T19:29:26Z"
    assert dec_item.person == "Bojun-Vvibe"  # Linked person from people list

    # Verify ISSUE-6730
    assert "ISSUE-6730" in catalog
    issue_item = catalog["ISSUE-6730"]
    assert issue_item.source_type == "issue"
    assert issue_item.source_id == "6730"
    assert issue_item.url == "https://github.com/psf/requests/issues/6730"
    assert issue_item.date == "2024-05-31T10:26:46Z"
    assert issue_item.person is None


def test_ask_unknown_evidence_id_dropped():
    """Verify unknown or unverified IDs are discarded from evidence."""
    fixture = _load_captured_fixture()
    mock_mgr = MagicMock()
    mock_engine = MagicMock()
    mock_retriever = MagicMock()
    mock_retriever.retrieve.return_value = fixture
    mock_engine.retriever = mock_retriever
    mock_engine.current_model = "test-model"

    raw_answer = Answer(
        question="Why timeout?",
        answer_text="Explained in [PR-7379] and [UNKNOWN-9999].",
        confidence=Confidence.MEDIUM,
        selected_evidence_ids=["PR-7379", "UNKNOWN-9999"],
    )
    mock_engine.answer.return_value = raw_answer

    result = ask.answer("Why timeout?", mock_mgr, answer_engine=mock_engine)

    assert result.success is True
    # UNKNOWN-9999 must be dropped
    assert len(result.answer.evidence) == 1
    assert result.answer.evidence[0].source_id == "7379"
    assert result.answer.selected_evidence_ids == ["PR-7379"]


def test_ask_duplicate_evidence_id_deduped():
    """Verify duplicate candidate IDs are deduplicated preserving order."""
    fixture = _load_captured_fixture()
    mock_mgr = MagicMock()
    mock_engine = MagicMock()
    mock_retriever = MagicMock()
    mock_retriever.retrieve.return_value = fixture
    mock_engine.retriever = mock_retriever

    raw_answer = Answer(
        question="Why timeout?",
        answer_text="Explained in [PR-7379].",
        confidence=Confidence.MEDIUM,
        selected_evidence_ids=["PR-7379", "PR-7379", "PR-#7379"],
    )
    mock_engine.answer.return_value = raw_answer

    result = ask.answer("Why timeout?", mock_mgr, answer_engine=mock_engine)

    assert result.success is True
    assert len(result.answer.evidence) == 1
    assert result.answer.evidence[0].source_id == "7379"


def test_ask_order_determinism():
    """Verify same retrieval context with different LLM ordering yields same card contents in requested order."""
    fixture = _load_captured_fixture()
    mock_mgr = MagicMock()
    mock_engine = MagicMock()
    mock_retriever = MagicMock()
    mock_retriever.retrieve.return_value = fixture
    mock_engine.retriever = mock_retriever

    # Order A: PR-7379 then ISSUE-6730
    mock_engine.answer.return_value = Answer(
        question="Why timeout?",
        answer_text="Answer A",
        confidence=Confidence.HIGH,
        selected_evidence_ids=["PR-7379", "ISSUE-6730"],
    )
    res_a = ask.answer("Why timeout?", mock_mgr, answer_engine=mock_engine)

    # Order B: ISSUE-6730 then PR-7379
    mock_engine.answer.return_value = Answer(
        question="Why timeout?",
        answer_text="Answer B",
        confidence=Confidence.HIGH,
        selected_evidence_ids=["ISSUE-6730", "PR-7379"],
    )
    res_b = ask.answer("Why timeout?", mock_mgr, answer_engine=mock_engine)

    # Identical contents, reordered matching selection
    assert res_a.answer.evidence[0].source_id == "7379"
    assert res_a.answer.evidence[1].source_id == "6730"

    assert res_b.answer.evidence[0].source_id == "6730"
    assert res_b.answer.evidence[1].source_id == "7379"

    # Card fields match exactly
    assert res_a.answer.evidence[0].title == res_b.answer.evidence[1].title
    assert res_a.answer.evidence[0].url == res_b.answer.evidence[1].url
    assert res_a.answer.evidence[1].title == res_b.answer.evidence[0].title


def test_ask_marker_rewriting_rules():
    """
    Verify marker rewriting:
    - [A, B] -> [1, 2]
    - Adjacent [A][B] -> [1][2]
    - Unverified markers stripped
    - Never add evidence because a marker named it
    """
    fixture = _load_captured_fixture()
    mock_mgr = MagicMock()
    mock_engine = MagicMock()
    mock_retriever = MagicMock()
    mock_retriever.retrieve.return_value = fixture
    mock_engine.retriever = mock_retriever

    text = (
        "Multi-cite [PR-7379, ISSUE-6730]. "
        "Adjacent cite [PR-7379][ISSUE-6730]. "
        "Partial unverified [PR-7379, UNKNOWN-999]. "
        "Ghost marker [PHANTOM-404]. "
        "Unselected DB candidate [DEC-PR-7379]. "
        "Preserved text [Note]."
    )
    mock_engine.answer.return_value = Answer(
        question="Why timeout?",
        answer_text=text,
        confidence=Confidence.HIGH,
        selected_evidence_ids=["PR-7379", "ISSUE-6730"],
    )

    res = ask.answer("Why timeout?", mock_mgr, answer_engine=mock_engine)
    rewritten = res.answer.answer_text

    # [PR-7379, ISSUE-6730] -> [1, 2]
    assert "Multi-cite [1, 2]." in rewritten
    # [PR-7379][ISSUE-6730] -> [1][2]
    assert "Adjacent cite [1][2]." in rewritten
    # [PR-7379, UNKNOWN-999] -> [1]
    assert "Partial unverified [1]." in rewritten
    # Ghost marker [PHANTOM-404] stripped
    assert "[PHANTOM" not in rewritten
    # Unselected DB candidate [DEC-PR-7379] stripped (and NEVER added to evidence!)
    assert "[DEC-PR-7379]" not in rewritten
    assert len(res.answer.evidence) == 2
    assert {e.source_id for e in res.answer.evidence} == {"7379", "6730"}
    # Non-citation text [Note] preserved
    assert "[Note]" in rewritten


def test_ask_confidence_capping():
    """Verify confidence capping rules."""
    fixture = _load_captured_fixture()
    mock_mgr = MagicMock()
    mock_engine = MagicMock()
    mock_retriever = MagicMock()
    mock_retriever.retrieve.return_value = fixture
    mock_engine.retriever = mock_retriever
    mock_engine._build_insufficient_evidence_answer.return_value = Answer(
        question="Why?",
        answer_text="No evidence found.",
        confidence=Confidence.LOW,
    )

    # 1. Zero verified evidence: LOW confidence, NO_RESULTS error
    mock_engine.answer.return_value = Answer(
        question="Why?",
        answer_text="Some text",
        confidence=Confidence.HIGH,
        selected_evidence_ids=["NONEXISTENT-1"],
    )
    res_zero = ask.answer("Why?", mock_mgr, answer_engine=mock_engine)
    assert res_zero.success is False
    assert res_zero.error.code == ErrorCode.NO_RESULTS
    assert res_zero.answer.confidence == Confidence.LOW
    assert "No evidence found" in res_zero.answer.answer_text

    # 2. 1 item (even if PR): capped at MEDIUM
    mock_engine.answer.return_value = Answer(
        question="Why?",
        answer_text="One PR cite [PR-7379].",
        confidence=Confidence.HIGH,
        selected_evidence_ids=["PR-7379"],
    )
    res_one = ask.answer("Why?", mock_mgr, answer_engine=mock_engine)
    assert res_one.success is True
    assert res_one.answer.confidence == Confidence.MEDIUM
    assert "Confidence capped at MEDIUM" in res_one.answer.confidence_explanation

    # 3. 2 items without PR or commit (e.g. 1 issue + 1 decision from issue): capped at MEDIUM
    mock_engine.answer.return_value = Answer(
        question="Why?",
        answer_text="Citing issue and issue-based decision [ISSUE-6730, DEC-ISSUE-6730].",
        confidence=Confidence.HIGH,
        selected_evidence_ids=["ISSUE-6730", "DEC-ISSUE-6730"],
    )
    res_no_pr = ask.answer("Why?", mock_mgr, answer_engine=mock_engine)
    assert res_no_pr.success is True
    assert res_no_pr.answer.confidence == Confidence.MEDIUM

    # 4. >=2 items including at least one PR or commit: HIGH preserved
    mock_engine.answer.return_value = Answer(
        question="Why?",
        answer_text="Citing PR and issue [PR-7379, ISSUE-6730].",
        confidence=Confidence.HIGH,
        selected_evidence_ids=["PR-7379", "ISSUE-6730"],
    )
    res_high = ask.answer("Why?", mock_mgr, answer_engine=mock_engine)
    assert res_high.success is True
    assert res_high.answer.confidence == Confidence.HIGH


def test_ask_url_collapsing_and_decision_source_credit():
    """
    Verify:
    1. Cards that resolve to the same source URL are collapsed (keep the first).
    2. Decision source is credited as PR/commit for HIGH confidence cap.
    3. Commits lacking message or date are excluded.
    """
    fixture = _load_captured_fixture()
    mock_mgr = MagicMock()
    mock_engine = MagicMock()
    mock_retriever = MagicMock()
    mock_retriever.retrieve.return_value = fixture
    mock_engine.retriever = mock_retriever

    # 1. Candidate catalog excludes commit stubs without message/date
    catalog, _ = ask._build_candidate_catalog(fixture, target_repo="psf/requests")
    # Fixture commits only have sha (no message/date), so none should be in catalog
    assert not any(k.startswith("COMMIT-") for k in catalog.keys())

    # 2. Both DEC-PR-7379 and PR-7379 resolve to the same URL (https://github.com/psf/requests/pull/7379)
    # When both are selected, PR-7379 collapses into DEC-PR-7379
    mock_engine.answer.return_value = Answer(
        question="Why timeout?",
        answer_text="Explained in [DEC-PR-7379, PR-7379].",
        confidence=Confidence.HIGH,
        selected_evidence_ids=["DEC-PR-7379", "PR-7379"],
    )
    res = ask.answer("Why timeout?", mock_mgr, answer_engine=mock_engine)
    assert res.success is True
    # PR-7379 collapsed, so only 1 card remains
    assert len(res.answer.evidence) == 1
    assert res.answer.evidence[0].source_id == "DEC-PR-7379"
    # Both markers rewritten to [1]
    assert "[1]" in res.answer.answer_text
    assert "[2]" not in res.answer.answer_text
    # 1 distinct source URL -> confidence capped at MEDIUM
    assert res.answer.confidence == Confidence.MEDIUM

    # 3. Decision card DEC-PR-7379 counts as a verified PR towards HIGH confidence
    # Combined with ISSUE-6730 (2 distinct sources, 1 is PR from decision) -> HIGH preserved
    mock_engine.answer.return_value = Answer(
        question="Why timeout?",
        answer_text="Explained in [DEC-PR-7379, ISSUE-6730].",
        confidence=Confidence.HIGH,
        selected_evidence_ids=["DEC-PR-7379", "ISSUE-6730"],
    )
    res2 = ask.answer("Why timeout?", mock_mgr, answer_engine=mock_engine)
    assert res2.success is True
    assert len(res2.answer.evidence) == 2
    assert res2.answer.confidence == Confidence.HIGH
