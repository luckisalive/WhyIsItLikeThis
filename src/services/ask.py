"""
Service for answering 'why' questions using GraphRAG and model cascades.
Framework-agnostic: no Streamlit imports or session state.
"""

from __future__ import annotations

import logging
from typing import Any, Dict, List, Optional, Set

from src.graph.embedding_generator import EmbeddingGenerator
from src.graph.neo4j_manager import Neo4jManager
from src.models import Answer
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


def _match_commit_date(source_id: str, results: List[Dict[str, Any]]) -> Optional[Any]:
    """
    Match commit date from retrieval context.
    - If SHAs differ in length: normalize and match on unique prefix.
    - If ambiguous (multiple prefix matches): return None.
    - No fuzzy matching.
    """
    if not source_id or not results:
        return None

    target = source_id.lower().strip()
    if target.startswith("commit"):
        target = target[6:].strip()
    target = target.lstrip("#").strip()
    if not target:
        return None

    # Collect all unique commit nodes from retrieval results
    commits_by_sha: Dict[str, Dict[str, Any]] = {}
    for res in results:
        for c in res.get("commits", []):
            if isinstance(c, dict) and c.get("sha"):
                sha_str = str(c["sha"]).lower().strip()
                if sha_str not in commits_by_sha:
                    commits_by_sha[sha_str] = c

    if not commits_by_sha:
        return None

    # 1. Exact full SHA match
    if target in commits_by_sha:
        return commits_by_sha[target].get("date")

    # 2. Length difference: target is prefix of full SHA
    prefix_matches = [
        c for sha, c in commits_by_sha.items()
        if sha.startswith(target)
    ]
    if len(prefix_matches) == 1:
        return prefix_matches[0].get("date")
    elif len(prefix_matches) > 1:
        logger.warning(f"Ambiguous commit prefix '{target}' matches multiple commits.")
        return None

    # Target is longer than stored SHA
    reverse_matches = [
        c for sha, c in commits_by_sha.items()
        if target.startswith(sha)
    ]
    if len(reverse_matches) == 1:
        return reverse_matches[0].get("date")
    elif len(reverse_matches) > 1:
        logger.warning(f"Ambiguous reverse commit match for '{target}'.")
        return None

    return None


def _match_pr_date(source_id: str, results: List[Dict[str, Any]]) -> Optional[Any]:
    """Exact PR number match against retrieval context."""
    target = str(source_id).lower().strip().lstrip("#").strip()
    if target.startswith("pr"):
        target = target[2:].strip().lstrip("#").strip()
    if not target:
        return None

    for res in results:
        for pr in res.get("pull_requests", []):
            if isinstance(pr, dict):
                pr_num = str(pr.get("number", "")).strip()
                if pr_num == target:
                    return pr.get("created_at") or pr.get("merged_at")
    return None


def _match_issue_date(source_id: str, results: List[Dict[str, Any]]) -> Optional[Any]:
    """Exact issue number match against retrieval context."""
    target = str(source_id).lower().strip().lstrip("#").strip()
    if target.startswith("issue"):
        target = target[5:].strip().lstrip("#").strip()
    if not target:
        return None

    for res in results:
        for iss in res.get("issues", []):
            if isinstance(iss, dict):
                iss_num = str(iss.get("number", "")).strip()
                if iss_num == target:
                    return iss.get("created_at")
    return None


def answer(
    question: str,
    neo4j_mgr: Optional[Neo4jManager],
    settings: Any = None,
    answer_engine: Optional[AnswerEngine] = None,
    allow_heuristic: bool = True,
) -> AskResult:
    """
    Answer an architectural question using GraphRAG.

    - related_decisions: retains model suggestions filtered to IDs present in retrieval context.
    - evidence.date: normalized exact or unique-prefix join against retrieval context; None if no match.
    - model_used & degraded: tracked on AskResult.
    - Error handling: returns DB_UNREACHABLE, NO_RESULTS, LLM_UNAVAILABLE without raising.
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

        # 1. Retrieve context to join evidence dates and validate related decisions
        retriever = getattr(engine, "retriever", None)
        retrieval_results = []
        if retriever:
            try:
                retrieval_results = retriever.retrieve(clean_question)
            except Exception as re:
                logger.error(f"Retriever failure: {re}")
                return AskResult(
                    error=ServiceError(
                        code=ErrorCode.DB_UNREACHABLE,
                        message=f"Database unreachable during retrieval: {re}",
                    )
                )

        if not retrieval_results:
            ans = engine._build_insufficient_evidence_answer(clean_question)
            return AskResult(
                answer=ans,
                model_used="insufficient_evidence",
                degraded=True,
                error=ServiceError(
                    code=ErrorCode.NO_RESULTS,
                    message="No relevant decisions or evidence found in graph.",
                ),
            )

        # Check if LLM unavailable and heuristic synthesis requested/bypassed
        if getattr(engine, "all_models_exhausted", False) and not allow_heuristic:
            return AskResult(
                error=ServiceError(
                    code=ErrorCode.LLM_UNAVAILABLE,
                    message="All LLM models are unavailable or rate-limited, and heuristic fallback is disabled.",
                ),
                degraded=True,
            )

        # 2. Generate answer
        ans = engine.answer(clean_question)

        # 3. Filter related_decisions to IDs present in retrieval context
        context_decision_ids: Set[str] = {
            r["decision"]["id"]
            for r in retrieval_results
            if isinstance(r.get("decision"), dict) and r["decision"].get("id")
        }
        filtered_related = [d_id for d_id in ans.related_decisions if d_id in context_decision_ids]
        ans.related_decisions = filtered_related

        # 4. Join evidence dates from retrieval context
        for ev in ans.evidence:
            stype = ev.source_type.lower().strip()
            sid = str(ev.source_id).strip()
            if stype in ("commit", "c"):
                ev.date = _match_commit_date(sid, retrieval_results)
            elif stype in ("pull_request", "pr"):
                ev.date = _match_pr_date(sid, retrieval_results)
            elif stype in ("issue", "iss"):
                ev.date = _match_issue_date(sid, retrieval_results)
            else:
                ev.date = None

        # 5. Determine model used and degraded status
        all_exhausted = bool(getattr(engine, "all_models_exhausted", False))
        active_idx = getattr(engine, "active_model_idx", 0)
        gemini_idx = getattr(engine, "active_gemini_idx", 0)
        try:
            is_fallback_active = (int(active_idx) > 0 or int(gemini_idx) > 0)
        except (TypeError, ValueError):
            is_fallback_active = False

        if all_exhausted or getattr(engine, "_is_placeholder", False):
            model_used = "heuristic_graph_synthesis"
            degraded = True
        else:
            model_used = getattr(engine, "current_model", "groq/openai/gpt-oss-120b")
            degraded = is_fallback_active

        return AskResult(
            answer=ans,
            model_used=model_used,
            degraded=degraded,
        )

    except Exception as e:
        logger.error(f"Error answering question: {e}")
        err_msg = str(e).lower()
        if "connect" in err_msg or "unreachable" in err_msg or "serviceunavailable" in err_msg:
            code = ErrorCode.DB_UNREACHABLE
        elif "rate limit" in err_msg or "quota" in err_msg or "429" in err_msg:
            code = ErrorCode.LLM_UNAVAILABLE
        else:
            code = ErrorCode.UNKNOWN

        return AskResult(
            error=ServiceError(
                code=code,
                message=f"Error executing answer query: {e}",
            )
        )
