"""
Service for answering 'why' questions using GraphRAG and model cascades.
Framework-agnostic: no Streamlit imports or session state.
"""

from __future__ import annotations

import logging
from typing import Any, Optional

from src.graph.embedding_generator import EmbeddingGenerator
from src.graph.neo4j_manager import Neo4jManager
from src.query.answer_engine import AnswerEngine
from src.query.retriever import WhyRetriever
from src.services.types import AskResult, ErrorCode, ServiceError

logger = logging.getLogger(__name__)


def create_answer_engine(neo4j_mgr: Neo4jManager, settings: Any) -> AnswerEngine:
    """Instantiate and configure AnswerEngine with retriever and model cascades."""
    embedding_gen = EmbeddingGenerator(
        settings.google_api_key,
        settings.embedding_model,
        settings.embedding_dimensions,
    )
    retriever = WhyRetriever(neo4j_mgr, embedding_gen)

    groq_fallbacks = [m.strip() for m in settings.groq_fallback_models.split(",") if m.strip()]
    gemini_reasoning = [m.strip() for m in settings.gemini_fallback_models.split(",") if m.strip()]

    return AnswerEngine(
        api_key=settings.groq_api_key,
        model=settings.groq_model,
        retriever=retriever,
        fallback_models=groq_fallbacks,
        google_api_key=settings.google_api_key,
        gemini_reasoning_models=gemini_reasoning,
        enable_cross_provider_fallback=settings.enable_cross_provider_fallback,
        cooldown_seconds=settings.rate_limit_cooldown_seconds,
    )


def answer(
    question: str,
    neo4j_mgr: Optional[Neo4jManager],
    settings: Any = None,
    answer_engine: Optional[AnswerEngine] = None,
) -> AskResult:
    """
    Answer architectural questions.
    Stage 1: Pure extraction of question answering flow.
    """
    if not neo4j_mgr:
        return AskResult(
            error=ServiceError(
                code=ErrorCode.DB_UNREACHABLE,
                message="Neo4j database connection required to search knowledge graph.",
            )
        )

    clean_question = question.strip()
    if not clean_question:
        return AskResult(
            error=ServiceError(
                code=ErrorCode.UNKNOWN,
                message="Please enter a question.",
            )
        )

    try:
        engine = answer_engine
        if not engine and settings:
            engine = create_answer_engine(neo4j_mgr, settings)

        if not engine:
            return AskResult(
                error=ServiceError(
                    code=ErrorCode.UNKNOWN,
                    message="Answer engine could not be initialized.",
                )
            )

        ans = engine.answer(clean_question)
        model_name = getattr(engine, "current_model", None)
        degraded = getattr(engine, "all_models_exhausted", False)

        return AskResult(
            answer=ans,
            model_used=model_name,
            degraded=degraded,
        )
    except Exception as e:
        logger.error(f"Error answering question: {e}")
        return AskResult(
            error=ServiceError(
                code=ErrorCode.UNKNOWN,
                message=f"Error executing answer query: {e}",
            )
        )
