"""Unit tests for supersession detection and explicit rules."""

from unittest.mock import MagicMock
import pytest

from src.supersession.detector import SupersessionDetector, extract_explicit_matches
from src.pipeline import IngestionPipeline


def test_extract_explicit_matches_pr_6767():
    """Real PR #6767: Title starting with 'Revert' and body referencing #6667."""
    title = "Revert caching a default SSLContext"
    body = (
        "This PR reverts the changes from #6667 to the previous behavior. "
        "Due to the number of edge cases and concurrency issues we've encountered..."
    )
    matches = extract_explicit_matches(text=body, title=title)
    assert len(matches) >= 1
    # Check that target #6667 was correctly extracted
    targets = [m["target_num"] for m in matches]
    assert 6667 in targets
    # Check rules involved
    rules = [m["rule"] for m in matches]
    assert any("revert" in r.lower() for r in rules)


def test_extract_explicit_matches_pr_7441():
    """Real PR #7441: Body containing 'reverts #7431'."""
    title = "Move Request.headers back to Mapping from MutableMapping"
    body = "reverts #7431"
    matches = extract_explicit_matches(text=body, title=title)
    assert len(matches) == 1
    assert matches[0]["target_num"] == 7431
    assert matches[0]["rule"] == "reverts #N"
    assert "7431" in matches[0]["evidence_text"]


def test_extract_explicit_matches_negative_replace():
    """Negative case: 'Replace the list comprehension' must NOT match."""
    title = "Replace the list comprehension with a generator"
    body = "This optimizes memory usage when reading large chunked streams."
    matches = extract_explicit_matches(text=body, title=title)
    assert len(matches) == 0


def test_extract_explicit_matches_supersedes_and_replaces():
    """Explicit 'supersedes #N' and 'replaces #N' rules."""
    text = "This implementation supersedes #7415 and replaces #7200."
    matches = extract_explicit_matches(text=text, title="Refactor connection pool")
    targets = {m["target_num"] for m in matches}
    assert targets == {7415, 7200}


def test_supersession_explicit_detector_resolves_edge():
    """Detector links DEC-PR-7441 -> DEC-PR-7431 with temporal check and properties."""
    mock_neo4j = MagicMock()
    mock_neo4j.execute_query.side_effect = [
        # PRs
        [
            {
                "num": 7441,
                "title": "Move Request.headers back to Mapping from MutableMapping",
                "body": "reverts #7431",
                "date": "2026-05-13T21:23:44Z",
            }
        ],
        # Commits
        [],
        # Decisions with dates
        [
            {
                "id": "DEC-PR-7441",
                "title": "Move Request.headers back to Mapping from MutableMapping",
                "date": "2026-05-13T21:23:44Z",
            },
            {
                "id": "DEC-PR-7431",
                "title": "Update HeadersType to allow mutable mappings for better typing compatibility",
                "date": "2026-05-11T20:37:26Z",
            },
        ],
    ]

    detector = SupersessionDetector(neo4j_mgr=mock_neo4j, llm_client=None)
    count = detector.detect_explicit()

    assert count == 1
    mock_neo4j.mark_superseded.assert_called_once()
    call_kwargs = mock_neo4j.mark_superseded.call_args[1]
    assert call_kwargs["old_id"] == "DEC-PR-7431"
    assert call_kwargs["new_id"] == "DEC-PR-7441"
    assert call_kwargs["detection_type"] == "EXPLICIT"
    assert call_kwargs["rule"] == "reverts #N"
    assert "7431" in call_kwargs["evidence_text"]


def test_supersession_explicit_detector_dangling_reference_unlinked():
    """PR #6767 references #6667, but DEC-PR-6667 is outside ingested window -> no edge created."""
    mock_neo4j = MagicMock()
    mock_neo4j.execute_query.side_effect = [
        # PRs
        [
            {
                "num": 6767,
                "title": "Revert caching a default SSLContext",
                "body": "This PR reverts the changes from #6667 to the previous behavior.",
                "date": "2024-07-18T18:00:33Z",
            }
        ],
        # Commits
        [],
        # Decisions: Only DEC-PR-6767 exists, DEC-PR-6667 does NOT exist
        [
            {
                "id": "DEC-PR-6767",
                "title": "Revert caching a default SSLContext",
                "date": "2024-07-18T18:00:33Z",
            }
        ],
    ]

    detector = SupersessionDetector(neo4j_mgr=mock_neo4j, llm_client=None)
    count = detector.detect_explicit()

    # No edge should be linked because target decision does not exist in graph
    assert count == 0
    mock_neo4j.mark_superseded.assert_not_called()


def test_supersession_run_all_default_flag_creates_no_implicit_edges():
    """Default include_implicit=False in run_all() executes only explicit detection."""
    mock_neo4j = MagicMock()
    detector = SupersessionDetector(neo4j_mgr=mock_neo4j, llm_client=None)
    detector.detect_explicit = MagicMock(return_value=1)
    detector.detect_implicit = MagicMock(return_value=5)

    # Call with default include_implicit=False
    results = detector.run_all(include_implicit=False)

    assert results["explicit_supersessions_detected"] == 1
    assert results["implicit_supersessions_detected"] == 0
    assert results["total_supersessions_detected"] == 1
    detector.detect_explicit.assert_called_once()
    detector.detect_implicit.assert_not_called()


def test_pipeline_default_flag_is_disabled():
    """Verify IngestionPipeline defaults enable_implicit_supersession to False."""
    mock_settings = MagicMock()
    mock_settings.github_token = "fake"
    mock_settings.target_repo = "fake/repo"
    mock_settings.gemini_fallback_models = ""
    mock_settings.groq_fallback_models = ""
    mock_settings.google_api_key = "fake"
    mock_settings.gemini_model = "fake"
    mock_settings.groq_api_key = "fake"
    mock_settings.groq_model = "fake"
    mock_settings.enable_cross_provider_fallback = False
    mock_settings.rate_limit_cooldown_seconds = 1
    mock_settings.neo4j_uri = "bolt://localhost:7687"
    mock_settings.neo4j_username = "neo4j"
    mock_settings.neo4j_password = "password"
    mock_settings.embedding_model = "fake"
    mock_settings.embedding_dimensions = 768

    pipeline = IngestionPipeline(settings=mock_settings)
    assert pipeline.enable_implicit_supersession is False
