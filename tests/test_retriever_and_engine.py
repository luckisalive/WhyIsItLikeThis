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


def test_answer_engine_groq_cascade_on_rate_limit():
    mock_retriever = MagicMock()
    mock_retriever.retrieve.return_value = [{"decision": {"id": "DEC-1", "title": "ASGI Support"}}]
    mock_retriever.format_context.return_value = "Mock context"

    engine = AnswerEngine(
        api_key="TEST_GROQ_KEY",
        model="openai/gpt-oss-120b",
        retriever=mock_retriever,
        fallback_models=["openai/gpt-oss-20b"],
    )
    assert engine.models == ["openai/gpt-oss-120b", "openai/gpt-oss-20b"]
    assert engine.active_model_idx == 0

    mock_client = MagicMock()

    def mock_create(**kwargs):
        model = kwargs.get("model")
        if model == "openai/gpt-oss-120b":
            raise Exception("429 Rate limit reached: TPM/RPD exceeded")
        mock_msg = MagicMock()
        mock_msg.content = '{"answer_text": "Answer from gpt-oss-20b", "confidence": "HIGH", "confidence_explanation": "Valid", "evidence": []}'
        return MagicMock(choices=[MagicMock(message=mock_msg)])

    mock_client.chat.completions.create.side_effect = mock_create
    engine.client = mock_client

    ans = engine.answer("Why ASGI?")
    assert "Answer from gpt-oss-20b" in ans.answer_text
    assert engine.active_model_idx == 1
    assert engine.current_model == "openai/gpt-oss-20b"


def test_answer_engine_gemini_cross_provider_fallback():
    mock_retriever = MagicMock()
    mock_retriever.retrieve.return_value = [{"decision": {"id": "DEC-1", "title": "Starlette Base"}}]
    mock_retriever.format_context.return_value = "Mock context"

    engine = AnswerEngine(
        api_key="TEST_GROQ_KEY",
        model="openai/gpt-oss-120b",
        retriever=mock_retriever,
        fallback_models=["openai/gpt-oss-20b"],
        google_api_key="TEST_GOOGLE_KEY",
        gemini_reasoning_models=["gemini-3.7-flash"],
        enable_cross_provider_fallback=True,
    )

    # Groq fails completely
    mock_groq = MagicMock()
    mock_groq.chat.completions.create.side_effect = Exception("429 Quota exhausted")
    engine.client = mock_groq

    # Gemini succeeds
    mock_gemini = MagicMock()
    mock_resp = MagicMock()
    mock_resp.text = '{"answer_text": "Answer from Gemini 3.7 Flash reasoning", "confidence": "HIGH", "confidence_explanation": "Direct", "evidence": []}'
    mock_gemini.models.generate_content.return_value = mock_resp
    engine.gemini_client = mock_gemini

    ans = engine.answer("Why Starlette?")
    assert "Answer from Gemini 3.7 Flash reasoning" in ans.answer_text
    assert ans.confidence == Confidence.HIGH
    mock_gemini.models.generate_content.assert_called_once()


def test_answer_engine_all_models_exhaustion_cooldown():
    mock_retriever = MagicMock()
    mock_retriever.retrieve.return_value = [{"decision": {"id": "DEC-1", "title": "Starlette Base"}}]
    mock_retriever.format_context.return_value = "Mock context"

    engine = AnswerEngine(
        api_key="TEST_GROQ_KEY",
        model="openai/gpt-oss-120b",
        retriever=mock_retriever,
        fallback_models=["openai/gpt-oss-20b"],
        google_api_key="TEST_GOOGLE_KEY",
        gemini_reasoning_models=["gemini-3.5-flash-lite"],
        enable_cross_provider_fallback=True,
        cooldown_seconds=0,
    )

    # Groq fails
    mock_groq = MagicMock()
    mock_groq.chat.completions.create.side_effect = Exception("429 Quota exhausted")
    engine.client = mock_groq

    # Gemini fails too
    mock_gemini = MagicMock()
    mock_gemini.models.generate_content.side_effect = Exception("429 Resource exhausted")
    engine.gemini_client = mock_gemini

    ans = engine.answer("Why Starlette?")
    assert engine.all_models_exhausted is True
    # Graceful answer from graph
    assert "Starlette" in ans.answer_text
    assert "throttled" in ans.confidence_explanation.lower()


