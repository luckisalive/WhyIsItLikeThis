"""
Typed result models and error codes for the framework-agnostic service layer.
All results use Pydantic attribute access only (no dict-subscripting).
"""

from __future__ import annotations

from enum import Enum
from typing import Any, Dict, List, Optional
from pydantic import BaseModel, computed_field

from src.models import Answer, GraphStats


class ErrorCode(str, Enum):
    DB_UNREACHABLE = "DB_UNREACHABLE"
    LLM_UNAVAILABLE = "LLM_UNAVAILABLE"
    NO_RESULTS = "NO_RESULTS"
    UNKNOWN = "UNKNOWN"


class ServiceError(BaseModel):
    code: ErrorCode
    message: str
    details: Optional[Dict[str, Any]] = None


class BaseServiceResult(BaseModel):
    """Base result with computed success derived strictly from error is None."""

    error: Optional[ServiceError] = None

    @computed_field
    @property
    def success(self) -> bool:
        return self.error is None


class StatsResult(BaseServiceResult):
    stats: Optional[GraphStats] = None


class DecisionListResult(BaseServiceResult):
    items: List[Dict[str, Any]] = []
    total: int = 0


class GraphNode(BaseModel):
    id: str
    type: str
    label: str
    properties: Dict[str, Any] = {}


class GraphEdge(BaseModel):
    source: str
    target: str
    label: str
    properties: Dict[str, Any] = {}


class GraphResult(BaseServiceResult):
    nodes: List[GraphNode] = []
    edges: List[GraphEdge] = []
    hidden_count: int = 0
    decision: Optional[Dict[str, Any]] = None
    author: Optional[Dict[str, Any]] = None


class AskResult(BaseServiceResult):
    answer: Optional[Answer] = None
    model_used: Optional[str] = None
    degraded: bool = False
