"""
Shared Pydantic models used across all components.
Defines the core data structures for decisions, code entities, and evidence chains.
"""

from __future__ import annotations

from datetime import datetime
from enum import Enum
from typing import Optional, Any

from pydantic import BaseModel, Field


# ──────────────────────────────────────────────
# Enums
# ──────────────────────────────────────────────


class DecisionType(str, Enum):
    ARCHITECTURE = "ARCHITECTURE"
    API_DESIGN = "API_DESIGN"
    PERFORMANCE = "PERFORMANCE"
    SECURITY = "SECURITY"
    DEPRECATION = "DEPRECATION"
    CONFIGURATION = "CONFIGURATION"
    DEPENDENCY = "DEPENDENCY"
    FEATURE = "FEATURE"
    BUG_FIX = "BUG_FIX"
    REFACTOR = "REFACTOR"
    INFRASTRUCTURE = "INFRASTRUCTURE"


class DecisionStatus(str, Enum):
    ACTIVE = "ACTIVE"
    SUPERSEDED = "SUPERSEDED"


class Confidence(str, Enum):
    HIGH = "HIGH"
    MEDIUM = "MEDIUM"
    LOW = "LOW"


class EntityType(str, Enum):
    FILE = "file"
    FUNCTION = "function"
    CLASS = "class"
    METHOD = "method"
    CONFIG_KEY = "config_key"
    MODULE = "module"


# ──────────────────────────────────────────────
# Extraction Models (LLM output structures)
# ──────────────────────────────────────────────


class Alternative(BaseModel):
    """An alternative that was considered but rejected."""

    alternative: str = ""
    rejection_reason: str = ""


class ExtractedDecision(BaseModel):
    """Structured decision extracted by the LLM from a PR/commit/issue."""

    title: str = Field(default="", description="Concise imperative title of the decision")
    decision_type: DecisionType = Field(
        default=DecisionType.ARCHITECTURE, alias="type", description="Category of the decision"
    )
    problem_context: str = Field(default="", description="The problem or bottleneck being addressed")
    decision_made: str = Field(default="", description="What was specifically decided and implemented")
    rationale: str = Field(default="", description="The 'why' — justification for this decision")
    status: DecisionStatus = Field(default=DecisionStatus.ACTIVE)
    alternatives_considered: list[Alternative] = Field(default_factory=list)
    supersedes_references: list[str] = Field(
        default_factory=list,
        description="PR numbers, commit SHAs, or decision IDs that this overrides",
    )
    trade_offs: list[str] = Field(default_factory=list)
    impacted_entities: list[str] = Field(
        default_factory=list,
        description="File paths, function names, config keys affected",
    )
    confidence_score: float = Field(
        default=0.0, ge=0.0, le=1.0, description="How clearly this text states a decision"
    )
    people_involved: list[str] = Field(default_factory=list)

    model_config = {"populate_by_name": True}


# ──────────────────────────────────────────────
# Code Entity Model
# ──────────────────────────────────────────────


class CodeEntity(BaseModel):
    """A code entity extracted via tree-sitter or file system analysis."""

    path: str = Field(description="File path relative to repo root")
    name: str = Field(description="Entity name (function, class, file, config key)")
    entity_type: EntityType = Field(default=EntityType.FILE, alias="type")
    language: str = Field(default="python")
    start_line: Optional[int] = None
    end_line: Optional[int] = None
    signature: Optional[str] = None
    docstring: Optional[str] = None

    model_config = {"populate_by_name": True}


# ──────────────────────────────────────────────
# Ingestion Data Models
# ──────────────────────────────────────────────


class FileChange(BaseModel):
    """A file changed in a commit."""

    filename: str
    status: str = ""  # added, removed, modified, renamed
    additions: int = 0
    deletions: int = 0
    patch: Optional[str] = None


class CommitData(BaseModel):
    """Processed commit data from GitHub."""

    sha: str
    message: str = ""
    author_login: str = Field(default="", alias="author")
    author_name: str = ""
    date: Any = ""  # Can be str or datetime
    additions: int = 0
    deletions: int = 0
    files_changed: list[FileChange] = Field(default_factory=list, alias="files")

    model_config = {"populate_by_name": True}

    @property
    def id(self) -> str:
        return self.sha


class CommentData(BaseModel):
    """A general comment on a PR or issue."""

    id: Any = ""  # Can be int or str
    author_login: str = Field(default="", alias="author")
    body: str = ""
    created_at: Any = ""  # Can be str or datetime

    model_config = {"populate_by_name": True}


class ReviewCommentData(BaseModel):
    """A code review comment on a specific line."""

    id: Any = ""  # Can be int or str
    author_login: str = Field(default="", alias="author")
    body: str = ""
    path: str = ""
    created_at: Any = ""  # Can be str or datetime
    commit_id: str = ""
    line: Optional[int] = None

    model_config = {"populate_by_name": True}


class PRData(BaseModel):
    """Processed pull request data from GitHub."""

    number: int
    title: str = ""
    body: Optional[str] = None
    author_login: str = Field(default="", alias="author")
    state: str = ""
    url: str = ""
    created_at: Any = ""  # Can be str or datetime
    closed_at: Optional[Any] = None
    merged_at: Optional[Any] = None
    commits: list[str] = Field(default_factory=list)
    files_changed: list[Any] = Field(default_factory=list)
    comments: list[CommentData] = Field(default_factory=list)
    review_comments: list[ReviewCommentData] = Field(default_factory=list)
    labels: list[str] = Field(default_factory=list)

    model_config = {"populate_by_name": True}

    @property
    def id(self) -> str:
        return str(self.number)


class IssueData(BaseModel):
    """Processed issue data from GitHub."""

    number: int
    title: str = ""
    body: Optional[str] = None
    author_login: str = Field(default="", alias="author")
    state: str = ""
    url: str = ""
    labels: list[str] = Field(default_factory=list)
    created_at: Any = ""  # Can be str or datetime
    closed_at: Optional[Any] = None
    comments: list[CommentData] = Field(default_factory=list)

    model_config = {"populate_by_name": True}

    @property
    def id(self) -> str:
        return str(self.number)


# ──────────────────────────────────────────────
# Query / Answer Models
# ──────────────────────────────────────────────


class EvidenceItem(BaseModel):
    """A single piece of evidence supporting an answer."""

    source_type: str = ""  # "pull_request", "commit", "issue", "decision"
    source_id: str = ""  # PR number, commit SHA, issue number, decision ID
    title: str = ""
    summary: str = ""
    url: Optional[str] = None
    date: Optional[Any] = None
    person: Optional[str] = None


class SupersessionInfo(BaseModel):
    """Information about a superseded decision."""

    superseded_decision: str = ""
    superseded_by: str = ""
    reason: str = ""
    superseded_date: Optional[Any] = None


class Answer(BaseModel):
    """A complete answer to a 'why' question."""

    question: str = ""
    answer_text: str = ""
    confidence: Confidence = Confidence.LOW
    confidence_explanation: str = ""
    evidence: list[EvidenceItem] = Field(default_factory=list)
    supersession: Optional[SupersessionInfo] = None
    ask_person: Optional[str] = Field(
        default=None,
        description="The person most likely to know more about this",
    )
    related_decisions: list[str] = Field(
        default_factory=list, description="IDs of related decisions"
    )


# ──────────────────────────────────────────────
# Graph Statistics
# ──────────────────────────────────────────────


class GraphStats(BaseModel):
    """Statistics about the knowledge graph."""

    total_decisions: int = 0
    active_decisions: int = 0
    superseded_decisions: int = 0
    total_commits: int = 0
    total_prs: int = 0
    total_issues: int = 0
    total_people: int = 0
    total_code_entities: int = 0
    total_relationships: int = 0
    # Aliases used by some components
    nodes: int = 0
    edges: int = 0
    components: int = 0
    density: float = 0.0
