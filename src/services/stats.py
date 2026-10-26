"""
Service for knowledge graph statistics.
Framework-agnostic: no Streamlit imports or session state.
"""

from __future__ import annotations

import logging
from typing import Optional

from src.graph.neo4j_manager import Neo4jManager
from src.services.types import ErrorCode, ServiceError, StatsResult

logger = logging.getLogger(__name__)


def get_stats(neo4j_mgr: Optional[Neo4jManager]) -> StatsResult:
    """Fetch knowledge graph statistics."""
    if not neo4j_mgr:
        return StatsResult(
            error=ServiceError(
                code=ErrorCode.DB_UNREACHABLE,
                message="Neo4j connection not available.",
            )
        )

    try:
        stats = neo4j_mgr.get_stats()
        return StatsResult(stats=stats)
    except Exception as e:
        logger.error(f"Error fetching stats: {e}")
        return StatsResult(
            error=ServiceError(
                code=ErrorCode.DB_UNREACHABLE,
                message=f"Failed to fetch stats: {e}",
            )
        )
