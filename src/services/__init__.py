"""
Services package: framework-agnostic service layer for 'Why The Code Is Like This'.
"""

from src.services import ask, decisions, graph, stats
from src.services.types import (
    AskResult,
    BaseServiceResult,
    DecisionListResult,
    ErrorCode,
    GraphEdge,
    GraphNode,
    GraphResult,
    ServiceError,
    StatsResult,
)

__all__ = [
    "ask",
    "decisions",
    "graph",
    "stats",
    "ErrorCode",
    "ServiceError",
    "BaseServiceResult",
    "StatsResult",
    "DecisionListResult",
    "GraphNode",
    "GraphEdge",
    "GraphResult",
    "AskResult",
]
