"""
Service for answering 'why' questions using GraphRAG and model cascades.
Framework-agnostic: no Streamlit imports or session state.
"""

from __future__ import annotations

import logging
from datetime import datetime
from typing import Any, Dict, List, Optional, Set, Tuple

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


def _to_iso_string(val: Any) -> Optional[str]:
    """Convert temporal value (Neo4j DateTime, datetime, or ISO string) to standard ISO string."""
    if val is None:
        return None
    if isinstance(val, str):
        clean_val = val.strip()
        try:
            datetime.fromisoformat(clean_val.replace("Z", "+00:00"))
            return clean_val
        except Exception:
            return clean_val
    if hasattr(val, "isoformat"):
        return val.isoformat()
    return str(val)


def _parse_source_ref(source_type: str, source_id: str) -> Tuple[str, str]:
    """
    Resolves evidence source_id format:
      DEC-COMMIT-<8 chars> -> ('commit', '<sha>')
      DEC-PR-<n>          -> ('pull_request', '<n>')
      DEC-ISSUE-<n>       -> ('issue', '<n>')
      COMMIT-<sha>        -> ('commit', '<sha>')
      PR-<n> / PR #<n>    -> ('pull_request', '<n>')
      ISSUE-<n>           -> ('issue', '<n>')
    """
    sid = str(source_id).strip()
    sid_upper = sid.upper()

    if sid_upper.startswith("DEC-COMMIT-"):
        return "commit", sid[11:].strip()
    if sid_upper.startswith("DEC-PR-"):
        return "pull_request", sid[7:].strip()
    if sid_upper.startswith("DEC-ISSUE-"):
        return "issue", sid[10:].strip()

    if sid_upper.startswith("COMMIT-") or sid_upper.startswith("COMMIT "):
        return "commit", sid[7:].strip()
    if sid_upper.startswith("PR-") or sid_upper.startswith("PR "):
        return "pull_request", sid[3:].strip().lstrip("#").strip()
    if sid_upper.startswith("ISSUE-") or sid_upper.startswith("ISSUE "):
        return "issue", sid[6:].strip().lstrip("#").strip()

    stype = source_type.lower().strip()
    clean_id = sid.lstrip("#").strip()
    if stype in ("c", "commit"):
        return "commit", clean_id
    if stype in ("pr", "pull_request", "pull"):
        return "pull_request", clean_id
    if stype in ("iss", "issue"):
        return "issue", clean_id

    return stype, clean_id


def _resolve_evidence_date(
    source_type: str,
    source_id: str,
    results: List[Dict[str, Any]],
) -> Optional[str]:
    """
    Resolve evidence date through its decision or context items:
      DEC-COMMIT-<sha> -> linked commit whose sha starts with it
      DEC-PR-<n>       -> linked PR with that number
      DEC-ISSUE-<n>    -> linked issue with that number
    Ambiguous or missing -> None. Converts temporal values to ISO string.
    """
    resolved_type, target_ref = _parse_source_ref(source_type, source_id)
    if not target_ref or not results:
        return None

    # Check commits
    if resolved_type == "commit":
        target_sha = target_ref.lower()
        all_commits: List[Dict[str, Any]] = []
        for res in results:
            # If decision ID matches the source_id directly, check its commits first
            dec_id = res.get("decision", {}).get("id", "")
            if dec_id and str(source_id).strip().upper() == dec_id.upper():
                dec_commits = res.get("commits", [])
                if dec_commits and isinstance(dec_commits[0], dict):
                    return _to_iso_string(dec_commits[0].get("date"))

            for c in res.get("commits", []):
                if isinstance(c, dict) and c.get("sha"):
                    all_commits.append(c)

        # Match commit SHA: exact or prefix match
        exact_matches = [c for c in all_commits if str(c.get("sha", "")).lower() == target_sha]
        if len(exact_matches) == 1:
            return _to_iso_string(exact_matches[0].get("date"))

        prefix_matches = [c for c in all_commits if str(c.get("sha", "")).lower().startswith(target_sha)]
        # Deduplicate commits by SHA to avoid false ambiguity
        unique_prefix_shas = {str(c.get("sha", "")).lower(): c for c in prefix_matches}
        if len(unique_prefix_shas) == 1:
            return _to_iso_string(list(unique_prefix_shas.values())[0].get("date"))
        elif len(unique_prefix_shas) > 1:
            logger.warning(f"Ambiguous commit prefix '{target_sha}' matched {len(unique_prefix_shas)} distinct commits.")
            return None

        return None

    # Check Pull Requests
    if resolved_type == "pull_request":
        target_num = target_ref.lstrip("#").strip()
        all_prs: List[Dict[str, Any]] = []
        for res in results:
            dec_id = res.get("decision", {}).get("id", "")
            if dec_id and str(source_id).strip().upper() == dec_id.upper():
                dec_prs = res.get("pull_requests", [])
                if dec_prs and isinstance(dec_prs[0], dict):
                    pr_date = dec_prs[0].get("created_at") or dec_prs[0].get("merged_at")
                    return _to_iso_string(pr_date)

            for pr in res.get("pull_requests", []):
                if isinstance(pr, dict):
                    all_prs.append(pr)

        matched_prs = [pr for pr in all_prs if str(pr.get("number", "")).strip() == target_num]
        if matched_prs:
            pr_val = matched_prs[0]
            pr_date = pr_val.get("created_at") or pr_val.get("merged_at")
            return _to_iso_string(pr_date)

        return None

    # Check Issues
    if resolved_type == "issue":
        target_num = target_ref.lstrip("#").strip()
        all_issues: List[Dict[str, Any]] = []
        for res in results:
            dec_id = res.get("decision", {}).get("id", "")
            if dec_id and str(source_id).strip().upper() == dec_id.upper():
                dec_issues = res.get("issues", [])
                if dec_issues and isinstance(dec_issues[0], dict):
                    return _to_iso_string(dec_issues[0].get("created_at"))

            for iss in res.get("issues", []):
                if isinstance(iss, dict):
                    all_issues.append(iss)

        matched_issues = [iss for iss in all_issues if str(iss.get("number", "")).strip() == target_num]
        if matched_issues:
            return _to_iso_string(matched_issues[0].get("created_at"))

        return None

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
    - evidence.date: resolved through linked decision / context (exact or unique-prefix); None if ambiguous/missing.
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

        # 4. Join evidence dates from retrieval context (resolving through decision / prefix match)
        for ev in ans.evidence:
            ev.date = _resolve_evidence_date(ev.source_type, ev.source_id, retrieval_results)

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
