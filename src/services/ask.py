"""
Service for answering 'why' questions using GraphRAG and model cascades.
Framework-agnostic: no Streamlit imports or session state.
"""

from __future__ import annotations

import logging
import re
from datetime import datetime
from typing import Any, Dict, List, Optional, Set, Tuple

from src.graph.embedding_generator import EmbeddingGenerator
from src.graph.neo4j_manager import Neo4jManager
from src.models import Answer, Confidence, EvidenceItem
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


def _build_candidate_catalog(
    retrieval_results: List[Dict[str, Any]],
    target_repo: str = "psf/requests",
) -> Tuple[Dict[str, EvidenceItem], Dict[str, str]]:
    """
    Builds canonical candidate lookup table and alias map strictly from database nodes.
    - Commits lacking message or date are strictly excluded.
    - Decision cards resolve url and date from their underlying source PR/commit/issue.
    - Author = None where not stored
    - Headline = first line of message
    - Date = created_at (PR/issue) or date (commit); no merged_at
    """
    catalog: Dict[str, EvidenceItem] = {}
    alias_map: Dict[str, str] = {}
    unresolved_decisions_count = 0
    total_decisions_count = 0

    for res in retrieval_results:
        # 1. Decision
        dec = res.get("decision", {})
        if isinstance(dec, dict) and dec.get("id"):
            total_decisions_count += 1
            dec_id = dec["id"]
            author = None
            if res.get("people"):
                p0 = res["people"][0]
                if isinstance(p0, dict):
                    author = p0.get("name") or p0.get("login")
            summary = dec.get("rationale") or dec.get("decision_made") or ""

            # Resolve url and date from the decision's source record
            src_type = str(dec.get("source_type") or "").lower()
            src_id = str(dec.get("source_id") or "").strip()

            dec_url = None
            dec_date = None
            resolved_source = None

            if src_type in ("pull_request", "pr"):
                for r in retrieval_results:
                    for pr in r.get("pull_requests", []):
                        if isinstance(pr, dict) and str(pr.get("number")) == src_id:
                            dec_url = pr.get("url") or f"https://github.com/{target_repo}/pull/{src_id}"
                            dec_date = _to_iso_string(pr.get("created_at"))
                            resolved_source = ("pull_request", src_id)
                            break
                    if resolved_source:
                        break
            elif src_type == "commit":
                for r in retrieval_results:
                    for c in r.get("commits", []):
                        if isinstance(c, dict) and c.get("sha"):
                            sha = str(c["sha"]).strip()
                            if sha.startswith(src_id) or src_id.startswith(sha):
                                dec_url = c.get("url") or f"https://github.com/{target_repo}/commit/{sha}"
                                dec_date = _to_iso_string(c.get("date"))
                                resolved_source = ("commit", sha[:8])
                                break
                    if resolved_source:
                        break
            elif src_type == "issue":
                for r in retrieval_results:
                    for iss in r.get("issues", []):
                        if isinstance(iss, dict) and str(iss.get("number")) == src_id:
                            dec_url = iss.get("url") or f"https://github.com/{target_repo}/issues/{src_id}"
                            dec_date = _to_iso_string(iss.get("created_at"))
                            resolved_source = ("issue", src_id)
                            break
                    if resolved_source:
                        break

            if not dec_url:
                unresolved_decisions_count += 1

            catalog[dec_id] = EvidenceItem(
                source_type="decision",
                source_id=dec_id,
                title=dec.get("title", dec_id),
                summary=summary,
                url=dec_url,
                date=dec_date,
                person=author,
                cited=True,
                underlying_source=resolved_source or ("decision", dec_id),
            )
            alias_map[dec_id.upper()] = dec_id

        # 2. Pull Requests
        for pr in res.get("pull_requests", []):
            if isinstance(pr, dict) and pr.get("number"):
                num_str = str(pr["number"])
                cid = f"PR-{num_str}"
                title = pr.get("title") or f"PR #{num_str}"
                summary = (pr.get("body") or "").strip()[:300]
                url = pr.get("url") or f"https://github.com/{target_repo}/pull/{num_str}"
                date = _to_iso_string(pr.get("created_at"))
                catalog[cid] = EvidenceItem(
                    source_type="pull_request",
                    source_id=num_str,
                    title=title,
                    summary=summary,
                    url=url,
                    date=date,
                    person=None,
                    cited=True,
                    underlying_source=("pull_request", num_str),
                )
                alias_map[cid.upper()] = cid
                alias_map[f"PR-#{num_str}".upper()] = cid
                alias_map[f"PR #{num_str}".upper()] = cid
                alias_map[num_str] = cid

        # 3. Commits (strictly require both message and date)
        for c in res.get("commits", []):
            if isinstance(c, dict) and c.get("sha") and c.get("message") and c.get("date"):
                full_sha = str(c["sha"]).strip()
                short_sha = full_sha[:8]
                cid = f"COMMIT-{short_sha}"
                msg = c.get("message", "")
                lines = msg.splitlines() if msg else []
                headline = lines[0] if lines else f"Commit {short_sha}"
                body_lines = lines[1:] if len(lines) > 1 else []
                summary = "\n".join(body_lines).strip()[:300] if body_lines else headline
                url = c.get("url") or f"https://github.com/{target_repo}/commit/{full_sha}"
                date = _to_iso_string(c.get("date"))
                catalog[cid] = EvidenceItem(
                    source_type="commit",
                    source_id=short_sha,
                    title=headline,
                    summary=summary,
                    url=url,
                    date=date,
                    person=None,
                    cited=True,
                    underlying_source=("commit", short_sha),
                )
                alias_map[cid.upper()] = cid
                alias_map[f"COMMIT-{full_sha}".upper()] = cid
                alias_map[full_sha.upper()] = cid
                alias_map[short_sha.upper()] = cid

        # 4. Issues
        for iss in res.get("issues", []):
            if isinstance(iss, dict) and iss.get("number"):
                num_str = str(iss["number"])
                cid = f"ISSUE-{num_str}"
                title = iss.get("title") or f"Issue #{num_str}"
                summary = (iss.get("body") or "").strip()[:300]
                url = iss.get("url") or f"https://github.com/{target_repo}/issues/{num_str}"
                date = _to_iso_string(iss.get("created_at"))
                catalog[cid] = EvidenceItem(
                    source_type="issue",
                    source_id=num_str,
                    title=title,
                    summary=summary,
                    url=url,
                    date=date,
                    person=None,
                    cited=True,
                    underlying_source=("issue", num_str),
                )
                alias_map[cid.upper()] = cid
                alias_map[f"ISSUE-#{num_str}".upper()] = cid
                alias_map[f"ISSUE #{num_str}".upper()] = cid
                alias_map[num_str] = cid

    if total_decisions_count > 0:
        logger.debug(
            f"Decision card hydration: {total_decisions_count - unresolved_decisions_count}/{total_decisions_count} "
            f"resolved url/date; {unresolved_decisions_count} kept with url=None."
        )

    return catalog, alias_map


def _rewrite_citation_markers(
    answer_text: str,
    verified_evidence: List[EvidenceItem],
    candidate_catalog: Dict[str, EvidenceItem],
    alias_map: Dict[str, str],
    collapsed_map: Optional[Dict[str, int]] = None,
) -> str:
    """
    Rewrites [CANONICAL-ID] markers to [1], [2] in final verified evidence order.
    - Handles [A, B] and adjacent [A][B].
    - Strips markers for candidates not in the final evidence list.
    - Preserves mappings for cards collapsed by identical source URL.
    - Never adds evidence because a marker named it.
    """
    id_to_num: Dict[str, int] = {}
    if collapsed_map:
        id_to_num.update(collapsed_map)

    for idx, ev in enumerate(verified_evidence, 1):
        for cid, item in candidate_catalog.items():
            if item.source_type == ev.source_type and item.source_id == ev.source_id:
                id_to_num[cid.upper()] = idx
                if ev.source_type == "commit":
                    id_to_num[f"COMMIT-{ev.source_id}".upper()] = idx
                elif ev.source_type == "pull_request":
                    id_to_num[f"PR-{ev.source_id}".upper()] = idx
                elif ev.source_type == "issue":
                    id_to_num[f"ISSUE-{ev.source_id}".upper()] = idx

    known_candidates_upper = set(alias_map.keys())

    def replace_bracket(match: re.Match) -> str:
        content = match.group(1).strip()
        tokens = [t.strip() for t in content.split(",") if t.strip()]
        resolved_nums = []
        is_citation_marker = False

        for token in tokens:
            token_clean = token.lstrip("#").strip()
            token_upper = token_clean.upper()

            resolved_cid = alias_map.get(token_upper) or token_upper
            num = id_to_num.get(resolved_cid) or id_to_num.get(token_upper)

            if num is not None:
                is_citation_marker = True
                if num not in resolved_nums:
                    resolved_nums.append(num)
            else:
                if (
                    token_upper in known_candidates_upper
                    or re.match(r"^[A-Z]+-[A-Za-z0-9_\-]+$", token_upper)
                    or re.match(r"^[0-9a-f]{7,40}$", token_clean, re.IGNORECASE)
                ):
                    is_citation_marker = True

        if is_citation_marker:
            if resolved_nums:
                return f"[{', '.join(str(n) for n in resolved_nums)}]"
            else:
                return ""
        else:
            return match.group(0)

    rewritten = re.sub(r"\[([^\]]+)\]", replace_bracket, answer_text)
    rewritten = re.sub(r"\[\s*\]", "", rewritten)
    rewritten = re.sub(r" +\[", " [", rewritten)
    rewritten = re.sub(r" +([,\.])", r"\1", rewritten)
    return rewritten


def answer(
    question: str,
    neo4j_mgr: Optional[Neo4jManager],
    settings: Any = None,
    answer_engine: Optional[AnswerEngine] = None,
    allow_heuristic: bool = True,
) -> AskResult:
    """
    Answer an architectural question using GraphRAG with strict evidence provenance.

    - Candidates: Canonical IDs (PR-..., COMMIT-..., ISSUE-..., DEC-...).
    - Evidence items: Hydrated strictly from database records; never authored by LLM.
    - Zero verified evidence: Insufficient-evidence text, LOW confidence, NO_RESULTS error.
    - Confidence cap: HIGH requires >=2 distinct underlying sources with at least one PR or commit; else MEDIUM.
    - Citation markers: Rewritten to [1], [2]; unresolved markers stripped.
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

        # 1. Retrieve context to establish candidate evidence items and decision context
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

        # 2. Build candidate catalog strictly from database records
        target_repo = getattr(settings, "target_repo", "psf/requests") if settings else "psf/requests"
        candidate_catalog, alias_map = _build_candidate_catalog(retrieval_results, target_repo)

        # 3. Generate answer
        ans = engine.answer(clean_question)

        # 4. Filter related_decisions to IDs present in retrieval context
        context_decision_ids: Set[str] = {
            r["decision"]["id"]
            for r in retrieval_results
            if isinstance(r.get("decision"), dict) and r["decision"].get("id")
        }
        ans.related_decisions = [d_id for d_id in ans.related_decisions if d_id in context_decision_ids]

        # 5. Hydrate verified evidence items strictly from candidate catalog (collapsing same URL)
        verified_evidence: List[EvidenceItem] = []
        seen_urls: Set[str] = set()
        seen_cids: Set[str] = set()
        collapsed_map: Dict[str, int] = {}

        raw_selected = getattr(ans, "selected_evidence_ids", []) or []
        for raw_id in raw_selected:
            raw_clean = str(raw_id).strip()
            canonical_cid = alias_map.get(raw_clean.upper()) or raw_clean
            if canonical_cid in candidate_catalog and canonical_cid not in seen_cids:
                item = candidate_catalog[canonical_cid]

                # Collapse evidence cards that resolve to the same source URL (keep the first)
                if item.url and item.url in seen_urls:
                    for idx, ev in enumerate(verified_evidence, 1):
                        if ev.url == item.url:
                            collapsed_map[canonical_cid.upper()] = idx
                            break
                    seen_cids.add(canonical_cid)
                    continue

                seen_cids.add(canonical_cid)
                if item.url:
                    seen_urls.add(item.url)
                verified_evidence.append(item)
                collapsed_map[canonical_cid.upper()] = len(verified_evidence)

        # 6. Determine model used, fallback status, and degraded state
        is_heuristic_fallback = (
            getattr(ans, "is_heuristic", False) is True
            or getattr(engine, "all_models_exhausted", False) is True
            or getattr(engine, "_is_placeholder", False) is True
        )
        active_idx = getattr(engine, "active_model_idx", 0)
        gemini_idx = getattr(engine, "active_gemini_idx", 0)
        try:
            is_fallback_active = int(active_idx) > 0 or int(gemini_idx) > 0
        except (TypeError, ValueError):
            is_fallback_active = False

        if is_heuristic_fallback:
            model_used = "heuristic_graph_synthesis"
            degraded = True
            # Flag cards as not cited in the answer
            for ev in verified_evidence:
                ev.cited = False
        else:
            raw_model = getattr(engine, "current_model", None)
            model_used = str(raw_model) if isinstance(raw_model, str) else "groq/openai/gpt-oss-120b"
            degraded = is_fallback_active

        # 7. Zero verified evidence rule: replace text with insufficient-evidence text, LOW confidence, NO_RESULTS
        if not verified_evidence:
            insufficient_ans = engine._build_insufficient_evidence_answer(clean_question)
            ans.answer_text = insufficient_ans.answer_text
            ans.confidence = Confidence.LOW
            ans.confidence_explanation = "Refused to invent an answer because no verified evidence supported the question."
            ans.evidence = []
            ans.selected_evidence_ids = []
            return AskResult(
                answer=ans,
                model_used=model_used,
                degraded=degraded,
                error=ServiceError(
                    code=ErrorCode.NO_RESULTS,
                    message="No verified evidence found to support the answer.",
                ),
            )

        # 8. Confidence cap: HIGH requires >=2 distinct underlying sources with at least one PR or commit
        distinct_sources: Set[Tuple[str, str]] = set()
        for ev in verified_evidence:
            if ev.underlying_source:
                distinct_sources.add(ev.underlying_source)
            else:
                distinct_sources.add((ev.source_type, ev.source_id))

        has_pr_or_commit = any(src[0] in ("pull_request", "commit") for src in distinct_sources)

        if is_heuristic_fallback:
            if ans.confidence == Confidence.HIGH:
                ans.confidence = Confidence.MEDIUM
            ans.confidence_explanation = (
                "Direct graph retrieval fallback (confidence capped at MEDIUM; cards not cited in synthesized text)."
            )
        elif ans.confidence == Confidence.HIGH:
            if len(distinct_sources) < 2 or not has_pr_or_commit:
                ans.confidence = Confidence.MEDIUM
                ans.confidence_explanation += (
                    " (Confidence capped at MEDIUM: requires >=2 distinct underlying sources "
                    "including at least one PR or commit)."
                )

        # 9. Rewrite inline citation markers and strip unresolvable markers
        ans.answer_text = _rewrite_citation_markers(
            ans.answer_text,
            verified_evidence,
            candidate_catalog,
            alias_map,
            collapsed_map,
        )

        ans.evidence = verified_evidence
        ans.selected_evidence_ids = [cid for cid in raw_selected if (alias_map.get(str(cid).strip().upper()) or str(cid).strip()) in seen_cids]

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
