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
