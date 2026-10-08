"""Unit tests for knowledge extraction layer."""

from src.extraction.code_entity_extractor import CodeEntityExtractor
from src.extraction.context_collator import ContextCollator
from src.extraction.decision_extractor import DecisionExtractor
from src.models import CommitData, EntityType, PRData


def test_context_collator_pr():
    collator = ContextCollator(max_chars=1000)
    pr = PRData(
        number=10,
        title="Migrate to Pydantic v2",
        body="Upgraded core models to improve serialization speed by 5x.",
        author="maintainer",
        state="closed",
        url="https://github.com/repo/pull/10",
        created_at="2023-05-01T00:00:00",
    )
    context = collator.collate_pr(pr)
    assert "PR Title: Migrate to Pydantic v2" in context
    assert "serialization speed" in context


def test_code_entity_extractor_python():
    extractor = CodeEntityExtractor()
    code = b"""
class ApplicationRouter:
    \"\"\"Custom router.\"\"\"
    def include_router(self, router):
        pass

def create_app():
    return ApplicationRouter()
"""
    entities = extractor.extract_from_file("fastapi/routing.py", code)
    names = [e.name for e in entities]
    types = [e.entity_type for e in entities]

    assert "ApplicationRouter" in names
    assert "include_router" in names or "create_app" in names
    assert EntityType.CLASS in types
    assert EntityType.FUNCTION in types


def test_code_entity_extractor_yaml():
    extractor = CodeEntityExtractor()
    yaml_content = b"""
server:
  port: 8000
timeout: 30
retries: 3
"""
    entities = extractor.extract_from_file("config.yaml", yaml_content)
    names = [e.name for e in entities]
    assert "server" in names or "timeout" in names
    assert all(e.entity_type == EntityType.CONFIG_KEY for e in entities)


def test_decision_extractor_heuristic():
    extractor = DecisionExtractor(api_key="PLACEHOLDER_KEY")
    text = (
        "PR Title: We decided to switch to Starlette for async performance\n"
        "We chose Starlette over Werkzeug in order to support native ASGI streaming."
    )
    decision = extractor.extract_from_text(text, "pull_request", "1")
    assert decision is not None
    assert "Starlette" in decision.title
    assert decision.confidence_score >= 0.5


def test_decision_extractor_model_cascade_on_quota():
    from unittest.mock import MagicMock
    extractor = DecisionExtractor(
        api_key="TEST_REAL_KEY_123",
        model="gemini-3.1-flash-lite",
        fallback_models=["gemini-2.5-flash-lite", "gemini-2.5-flash"],
    )
    assert extractor.models == ["gemini-3.1-flash-lite", "gemini-2.5-flash-lite", "gemini-2.5-flash"]
    assert extractor.active_model_idx == 0

    # Mock client to simulate 429 quota exhaustion on first model and success on second
    mock_client = MagicMock()
    mock_resp = MagicMock()
    mock_resp.text = '{"title": "Switch to Starlette", "type": "ARCHITECTURE", "problem_context": "c", "decision_made": "d", "rationale": "r", "confidence_score": 0.9}'

    def mock_generate_content(model, contents, config):
        if model == "gemini-3.1-flash-lite":
            raise Exception("429 RESOURCE_EXHAUSTED: Daily quota exceeded for Queries per day")
        return mock_resp

    mock_client.models.generate_content.side_effect = mock_generate_content
    extractor.client = mock_client

    text = "PR Title: We decided to switch to Starlette for async performance\nWe chose Starlette over Werkzeug."
    decision = extractor.extract_from_text(text, "pull_request", "1")

    assert decision is not None
    assert decision.title == "Switch to Starlette"
    # Verify active model advanced sticky to index 1 (gemini-2.5-flash-lite)
    assert extractor.active_model_idx == 1
    assert extractor.current_model == "gemini-2.5-flash-lite"


def test_decision_extractor_groq_cross_provider_fallback():
    from unittest.mock import MagicMock
    extractor = DecisionExtractor(
        api_key="TEST_REAL_KEY_123",
        model="gemini-3.1-flash-lite",
        fallback_models=["gemini-2.5-flash-lite"],
        groq_api_key="TEST_GROQ_KEY",
        groq_fallback_models=["openai/gpt-oss-20b"],
        enable_cross_provider_fallback=True,
    )

    # Mock Gemini client to always fail with quota exhaustion
    mock_client = MagicMock()
    mock_client.models.generate_content.side_effect = Exception("429 Daily quota reached")
    extractor.client = mock_client

    # Mock Groq client to succeed
    mock_groq = MagicMock()
    mock_groq_resp = MagicMock()
    mock_msg = MagicMock()
    mock_msg.content = '{"title": "Adopt Pydantic", "type": "ARCHITECTURE", "problem_context": "c", "decision_made": "d", "rationale": "r", "confidence_score": 0.85}'
    mock_groq_resp.choices = [MagicMock(message=mock_msg)]
    mock_groq.chat.completions.create.return_value = mock_groq_resp
    extractor.groq_client = mock_groq

    text = "PR Title: We decided to adopt Pydantic for validation\nDiscussion on schemas."
    decision = extractor.extract_from_text(text, "pull_request", "2")

    assert decision is not None
    assert decision.title == "Adopt Pydantic"
    assert extractor.active_model_idx >= len(extractor.models)
    mock_groq.chat.completions.create.assert_called_once()

