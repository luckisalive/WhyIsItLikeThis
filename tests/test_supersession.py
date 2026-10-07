"""Unit tests for supersession detection."""

from unittest.mock import MagicMock

from src.supersession.detector import SupersessionDetector


def test_supersession_explicit_detection():
    mock_neo4j = MagicMock()
    # Mock returning an active decision containing explicit supersedes pattern
    mock_neo4j.execute_query.side_effect = [
        # First query: fetch active decisions
        [
            {
                "d": {
                    "id": "DEC-PR-200",
                    "title": "Migrate to Pydantic v2 (supersedes #45)",
                    "rationale": "Better performance and modern syntax",
                    "decision_made": "Replaces v1 models",
                }
            }
        ],
        # Second query: match referenced older decision
        [
            {"old_id": "DEC-PR-45"}
        ],
    ]

    detector = SupersessionDetector(neo4j_mgr=mock_neo4j, llm_client=None)
    count = detector.detect_explicit()

    assert count == 1
    mock_neo4j.mark_superseded.assert_called_once()
    call_args = mock_neo4j.mark_superseded.call_args[1]
    assert call_args["old_id"] == "DEC-PR-45"
    assert call_args["new_id"] == "DEC-PR-200"


def test_supersession_implicit_heuristic_detection():
    mock_neo4j = MagicMock()
    mock_neo4j.find_overlapping_decisions.return_value = [
        (
            {"id": "DEC-1", "title": "Use custom router", "rationale": "Basic routing"},
            {"id": "DEC-2", "title": "Replace custom router with Starlette router", "rationale": "Full ASGI conformance"},
            {"name": "fastapi/routing.py"},
        )
    ]

    detector = SupersessionDetector(neo4j_mgr=mock_neo4j, llm_client=None)
    count = detector.detect_implicit()

    assert count == 1
    mock_neo4j.mark_superseded.assert_called_once()
    call_args = mock_neo4j.mark_superseded.call_args[1]
    assert call_args["old_id"] == "DEC-1"
    assert call_args["new_id"] == "DEC-2"
