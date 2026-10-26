"""
Service for querying and listing decisions.
Framework-agnostic: no Streamlit imports or session state.
"""

from __future__ import annotations

import logging
from typing import Optional

from src.graph.neo4j_manager import Neo4jManager
from src.services.types import DecisionListResult, ErrorCode, ServiceError

logger = logging.getLogger(__name__)


def list_decisions(
    neo4j_mgr: Optional[Neo4jManager],
    status: Optional[str] = None,
    type: Optional[str] = None,
    limit: int = 50,
    offset: int = 0,
) -> DecisionListResult:
    """
    List decisions.
    Stage 1: Pure extraction of baseline timeline query.
    """
    if not neo4j_mgr:
        return DecisionListResult(
            error=ServiceError(
                code=ErrorCode.DB_UNREACHABLE,
                message="Neo4j connection not available.",
            )
        )

    timeline_query = """
    MATCH (d:Decision)
    OPTIONAL MATCH (newer:Decision)-[:SUPERSEDES]->(d)
    RETURN d, newer
    ORDER BY d.id DESC
    LIMIT 50
    """

    try:
        records = neo4j_mgr.execute_query(timeline_query)
        items = []
        for rec in records:
            d = rec.get("d") or {}
            newer = rec.get("newer")
            items.append({
                "id": d.get("id", ""),
                "title": d.get("title", "Untitled Decision"),
                "d": d,
                "newer": newer,
            })
        return DecisionListResult(items=items, total=len(items))
    except Exception as e:
        logger.error(f"Error querying decisions: {e}")
        return DecisionListResult(
            error=ServiceError(
                code=ErrorCode.DB_UNREACHABLE,
                message=f"Error executing decisions query: {e}",
            )
        )
