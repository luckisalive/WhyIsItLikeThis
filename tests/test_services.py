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
# 4. Ask Evidence Date Join & Related Decisions Filtering
# ─────────────────────────────────────────────────────────────


def test_ask_evidence_dates_and_filtered_related_decisions():
    """Verify related_decisions filtered to context and evidence dates joined exactly."""
    mock_mgr = MagicMock()
    mock_engine = MagicMock()
    mock_retriever = MagicMock()

    retrieval_fixture = [
        {
            "decision": {"id": "DEC-42", "title": "Use Starlette"},
            "pull_requests": [
                {"number": 101, "title": "Starlette PR", "created_at": "2023-01-15T10:00:00Z"}
            ],
            "commits": [
                {"sha": "a1b2c3d4e5f6789012345678901234567890abcd", "date": "2023-01-16T12:00:00Z"}
            ],
            "issues": [
                {"number": 55, "title": "ASGI Issue", "created_at": "2023-01-14T08:00:00Z"}
            ],
        }
    ]
    mock_retriever.retrieve.return_value = retrieval_fixture
    mock_engine.retriever = mock_retriever
    mock_engine.current_model = "groq/openai/gpt-oss-120b"
    mock_engine.active_model_idx = 0
    mock_engine.active_gemini_idx = 0
    mock_engine.all_models_exhausted = False
    mock_engine._is_placeholder = False

    raw_answer = Answer(
        question="Why Starlette?",
        answer_text="Because ASGI support was necessary.",
        confidence=Confidence.HIGH,
        # LLM suggests valid DEC-42 AND hallucinated DEC-999
        related_decisions=["DEC-42", "DEC-999-HALLUCINATED"],
        evidence=[
            EvidenceItem(
                source_type="pull_request",
                source_id="101",
                title="PR 101",
                summary="PR Discussion",
            ),
            # Evidence with short SHA prefix (7 chars)
            EvidenceItem(
                source_type="commit",
                source_id="a1b2c3d",
                title="Commit",
                summary="Implementation",
            ),
            EvidenceItem(
                source_type="issue",
                source_id="55",
                title="Issue 55",
                summary="Initial Issue",
            ),
            EvidenceItem(
                source_type="pull_request",
                source_id="9999",  # Not in context
                title="Unmatched PR",
                summary="Ghost PR",
            ),
        ],
    )
    mock_engine.answer.return_value = raw_answer

    result = ask.answer("Why Starlette?", mock_mgr, answer_engine=mock_engine)

    assert result.success is True
    assert result.error is None
    assert result.model_used == "groq/openai/gpt-oss-120b"
    assert result.degraded is False

    ans = result.answer
    assert ans is not None

    # Verify hallucinated decision was filtered out
    assert ans.related_decisions == ["DEC-42"]

    # Verify exact date join (including normalized prefix match for commit)
    ev_map = {f"{e.source_type}:{e.source_id}": e for e in ans.evidence}
    assert ev_map["pull_request:101"].date == "2023-01-15T10:00:00Z"
    assert ev_map["commit:a1b2c3d"].date == "2023-01-16T12:00:00Z"
    assert ev_map["issue:55"].date == "2023-01-14T08:00:00Z"
    assert ev_map["pull_request:9999"].date is None  # Unmatched -> None


def test_ask_commit_date_ambiguous_prefix_returns_none():
    """Verify ambiguous commit SHA prefix matching 2+ commits returns None."""
    mock_mgr = MagicMock()
    mock_engine = MagicMock()
    mock_retriever = MagicMock()

    retrieval_fixture = [
        {
            "decision": {"id": "DEC-1"},
            "commits": [
                {"sha": "a1b2c3d4_one", "date": "2023-01-01T00:00:00Z"},
                {"sha": "a1b2c3d4_two", "date": "2023-02-02T00:00:00Z"},
            ],
        }
    ]
    mock_retriever.retrieve.return_value = retrieval_fixture
    mock_engine.retriever = mock_retriever
    mock_engine.current_model = "test-model"
    mock_engine.all_models_exhausted = False
    mock_engine._is_placeholder = False

    raw_answer = Answer(
        question="Why?",
        answer_text="Test",
        evidence=[
            EvidenceItem(
                source_type="commit",
                source_id="a1b2c3d4",  # Matches BOTH commits above!
                title="Ambiguous Commit",
            )
        ],
    )
    mock_engine.answer.return_value = raw_answer

    result = ask.answer("Why?", mock_mgr, answer_engine=mock_engine)

    assert result.success is True
    # Ambiguous prefix must evaluate to None
    assert result.answer.evidence[0].date is None
