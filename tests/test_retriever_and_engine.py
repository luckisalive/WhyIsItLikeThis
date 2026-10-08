"""Unit tests for query retriever and answer engine."""

from unittest.mock import MagicMock

from src.graph.embedding_generator import EmbeddingGenerator
from src.models import Confidence
from src.query.answer_engine import AnswerEngine
from src.query.retriever import WhyRetriever


def test_embedding_generator_fallback():
    embedder = EmbeddingGenerator(api_key="GOOGLE_API_KEY")
    vec = embedder.embed_text("Why does FastAPI use Starlette?")
    assert len(vec) == 768
    # Check that vector is normalized or non-zero
    assert any(abs(v) > 0.0 for v in vec)


def test_retriever_context_formatting():
    mock_neo4j = MagicMock()
    embedder = EmbeddingGenerator(api_key="GOOGLE_API_KEY")
    retriever = WhyRetriever(mock_neo4j, embedder)

    mock_results = [
        {
            "decision": {
                "id": "DEC-PR-1",
                "title": "Use Starlette",
                "rationale": "High throughput ASGI foundation",
                "status": "ACTIVE",
            },
            "score": 0.92,
            "pull_requests": [{"number": 1, "url": "https://github.com/repo/pull/1"}],
            "commits": [{"sha": "abc12345"}],
            "people": [{"name": "tiangolo"}],
            "code_entities": [{"name": "FastAPI"}],
            "superseded_by": [],
            "supersedes": [],
        }
    ]

    context = retriever.format_context(mock_results)
    assert "Use Starlette" in context
    assert "High throughput ASGI foundation" in context
    assert "Discussed in PRs" in context


def test_answer_engine_heuristic_synthesis():
    mock_retriever = MagicMock()
    mock_retriever.retrieve.return_value = [
        {
            "decision": {
                "id": "DEC-PR-10",
                "title": "Adopt Pydantic for validation",
                "decision_made": "Use Pydantic v1 schemas",
                "rationale": "Automatic schema generation and type enforcement",
                "status": "ACTIVE",
            },
            "score": 0.88,
            "pull_requests": [{"number": 10}],
            "people": [{"name": "tiangolo"}],
        }
    ]
    mock_retriever.format_context.return_value = "Mock context"

    engine = AnswerEngine(api_key="GROQ_API_KEY", retriever=mock_retriever)
    ans = engine.answer("Why was Pydantic chosen?")

    assert ans.confidence in (Confidence.HIGH, Confidence.MEDIUM)
    assert "Adopt Pydantic for validation" in ans.answer_text
    assert ans.ask_person == "tiangolo"


def test_answer_engine_insufficient_evidence():
    mock_retriever = MagicMock()
    mock_retriever.retrieve.return_value = []

    engine = AnswerEngine(api_key="GROQ_API_KEY", retriever=mock_retriever)
    ans = engine.answer("Why did you build this in COBOL?")

    assert ans.confidence == Confidence.LOW
    assert "don't have enough evidence" in ans.answer_text.lower()
