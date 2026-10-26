"""
Service for querying and listing architectural decisions.
Framework-agnostic: no Streamlit imports or session state.
"""

from __future__ import annotations

import logging
from typing import Any, Dict, List, Optional

from src.graph.neo4j_manager import Neo4jManager
from src.services.types import DecisionListResult, ErrorCode, ServiceError

logger = logging.getLogger(__name__)

# Note on Deterministic Ordering:
# In the Neo4j schema and ingestion pipeline (see src/models.py and Neo4jManager.upsert_decision),
# Decision nodes do NOT contain a 'date', 'created_at', or timestamp property.
# To establish deterministic pagination without inventing a non-existent date field,
# decisions are ordered deterministically by `d.id ASC, d.title ASC`.

DECISION_COUNT_QUERY = """
MATCH (d:Decision)
WHERE ($type IS NULL OR d.decision_type = $type OR d.type = $type)
  AND (
    $status IS NULL
    OR ($status = 'SUPERSEDED' AND (d.status = 'SUPERSEDED' OR EXISTS { MATCH (:Decision)-[:SUPERSEDES]->(d) }))
    OR ($status = 'ACTIVE' AND (d.status = 'ACTIVE' OR d.status IS NULL) AND NOT EXISTS { MATCH (:Decision)-[:SUPERSEDES]->(d) })
  )
RETURN count(d) AS total
"""

DECISION_PAGE_QUERY = """
MATCH (d:Decision)
WHERE ($type IS NULL OR d.decision_type = $type OR d.type = $type)
  AND (
    $status IS NULL
    OR ($status = 'SUPERSEDED' AND (d.status = 'SUPERSEDED' OR EXISTS { MATCH (:Decision)-[:SUPERSEDES]->(d) }))
    OR ($status = 'ACTIVE' AND (d.status = 'ACTIVE' OR d.status IS NULL) AND NOT EXISTS { MATCH (:Decision)-[:SUPERSEDES]->(d) })
  )
OPTIONAL MATCH (newer:Decision)-[:SUPERSEDES]->(d)
RETURN d, newer
ORDER BY d.id ASC, d.title ASC
SKIP $offset
LIMIT $limit
"""


def list_decisions(
    neo4j_mgr: Optional[Neo4jManager],
    status: Optional[str] = None,
    type: Optional[str] = None,
    limit: int = 50,
    offset: int = 0,
) -> DecisionListResult:
    """
    List decisions with deterministic ordering and pagination.

    Status source of truth:
      - 'SUPERSEDED': d.status == 'SUPERSEDED' OR incoming SUPERSEDES edge exists.
      - 'ACTIVE': d.status == 'ACTIVE' (or null) AND no incoming SUPERSEDES edge.
    """
    if not neo4j_mgr:
        return DecisionListResult(
            error=ServiceError(
                code=ErrorCode.DB_UNREACHABLE,
                message="Neo4j connection not available.",
            )
        )

    norm_status = status.upper().strip() if status else None
    if norm_status and norm_status not in ("ACTIVE", "SUPERSEDED"):
        return DecisionListResult(
            error=ServiceError(
                code=ErrorCode.UNKNOWN,
                message=f"Invalid status filter: {status}. Expected ACTIVE or SUPERSEDED.",
            )
        )

    params: Dict[str, Any] = {
        "status": norm_status,
        "type": type.strip() if type else None,
        "offset": max(0, offset),
        "limit": max(1, limit),
    }

    try:
        count_records = neo4j_mgr.execute_query(DECISION_COUNT_QUERY, params)
        total = count_records[0]["total"] if count_records else 0

        if total == 0:
            return DecisionListResult(
                items=[],
                total=0,
                error=ServiceError(
                    code=ErrorCode.NO_RESULTS,
                    message="No decisions found matching the specified criteria.",
                ),
            )

        page_records = neo4j_mgr.execute_query(DECISION_PAGE_QUERY, params)
        items: List[Dict[str, Any]] = []

        for rec in page_records:
            d = rec.get("d") or {}
            newer = rec.get("newer")
            stored_status = d.get("status", "ACTIVE")
            is_superseded = stored_status == "SUPERSEDED" or bool(newer)
            effective_status = "SUPERSEDED" if is_superseded else "ACTIVE"

            items.append({
                "id": d.get("id", ""),
                "title": d.get("title", "Untitled Decision"),
                "decision_type": d.get("decision_type", d.get("type", "ARCHITECTURE")),
                "status": effective_status,
                "is_superseded": is_superseded,
                "rationale": d.get("rationale", "No rationale documented."),
                "problem_context": d.get("problem_context", ""),
                "newer": newer,
                "d": d,
            })

        return DecisionListResult(items=items, total=total)

    except Exception as e:
        logger.error(f"Error querying decisions: {e}")
        err_msg = str(e).lower()
        if "connect" in err_msg or "unreachable" in err_msg or "serviceunavailable" in err_msg or "defunct" in err_msg:
            code = ErrorCode.DB_UNREACHABLE
        else:
            code = ErrorCode.DB_UNREACHABLE  # query execution against DB
        return DecisionListResult(
            error=ServiceError(
                code=code,
                message=f"Error executing decisions query: {e}",
            )
        )
