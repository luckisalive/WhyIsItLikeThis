"""Unit tests for shared Pydantic models."""

from src.models import (
    Answer,
    CodeEntity,
    CommitData,
    Confidence,
    DecisionStatus,
    DecisionType,
    EntityType,
    EvidenceItem,
    ExtractedDecision,
    FileChange,
    GraphStats,
    IssueData,
    PRData,
    SupersessionInfo,
)


def test_extracted_decision_model():
    dec = ExtractedDecision(
        title="Use Starlette for ASGI foundation",
        type="ARCHITECTURE",
        problem_context="Needed high performance async request handling",
        decision_made="Built FastAPI directly on top of Starlette",
        rationale="Starlette provides robust ASGI routing and middleware",
        confidence_score=0.95,
        impacted_entities=["fastapi/applications.py"],
    )
    assert dec.title == "Use Starlette for ASGI foundation"
    assert dec.decision_type == DecisionType.ARCHITECTURE
    assert dec.confidence_score == 0.95
    assert dec.status == DecisionStatus.ACTIVE
    assert "fastapi/applications.py" in dec.impacted_entities


def test_code_entity_model():
    entity = CodeEntity(
        path="fastapi/routing.py",
        name="APIRoute",
        type=EntityType.CLASS,
        language="python",
        start_line=10,
        end_line=100,
    )
    assert entity.path == "fastapi/routing.py"
    assert entity.name == "APIRoute"
    assert entity.entity_type == EntityType.CLASS


def test_commit_and_pr_models():
    commit = CommitData(
        sha="abcdef123456",
        message="feat: add async support",
        author="tiangolo",
        date="2020-01-01T00:00:00",
        files=[FileChange(filename="main.py", status="modified", additions=5, deletions=2)],
    )
    assert commit.sha == "abcdef123456"
    assert commit.author_login == "tiangolo"
    assert commit.id == "abcdef123456"
    assert len(commit.files_changed) == 1

    pr = PRData(
        number=101,
        title="Support background tasks",
        author="tiangolo",
        state="closed",
        url="https://github.com/tiangolo/fastapi/pull/101",
        created_at="2020-01-02T00:00:00",
    )
    assert pr.number == 101
    assert pr.id == "101"
    assert pr.author_login == "tiangolo"


def test_answer_model():
    ans = Answer(
        question="Why Starlette?",
        answer_text="Because it is fast and ASGI standard.",
        confidence=Confidence.HIGH,
        confidence_explanation="Confirmed by early commits",
        evidence=[
            EvidenceItem(
                source_type="pull_request",
                source_id="1",
                title="Initial commit",
                summary="Base structure",
            )
        ],
        supersession=SupersessionInfo(
            superseded_decision="Old approach",
            superseded_by="New approach",
            reason="Better throughput",
        ),
        ask_person="tiangolo",
    )
    assert ans.confidence == Confidence.HIGH
    assert ans.supersession.superseded_by == "New approach"
    assert len(ans.evidence) == 1
