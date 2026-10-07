"""Unit tests for SQLite persistent cache."""

import os
import tempfile
import pytest

from src.ingestion.cache import SQLiteCache


@pytest.fixture
def temp_cache():
    temp_dir = tempfile.mkdtemp()
    db_path = os.path.join(temp_dir, "test_cache.db")
    cache = SQLiteCache(db_path=db_path)
    yield cache
    try:
        if os.path.exists(db_path):
            os.remove(db_path)
        os.rmdir(temp_dir)
    except Exception:
        pass


def test_commit_cache(temp_cache):
    commit_data = {"sha": "123abc456", "message": "Initial commit", "author": "dev"}
    temp_cache.store_commit("123abc456", commit_data)

    retrieved = temp_cache.get_commit("123abc456")
    assert retrieved is not None
    assert retrieved["message"] == "Initial commit"
    assert retrieved["author"] == "dev"

    all_commits = temp_cache.get_all_commits()
    assert len(all_commits) == 1


def test_pr_cache(temp_cache):
    pr_data = {"number": 42, "title": "Refactor router", "state": "closed"}
    temp_cache.store_pr(42, pr_data)

    retrieved = temp_cache.get_pr(42)
    assert retrieved is not None
    assert retrieved["title"] == "Refactor router"


def test_issue_cache(temp_cache):
    issue_data = {"number": 99, "title": "Bug in query parsing"}
    temp_cache.store_issue(99, issue_data)

    retrieved = temp_cache.get_issue(99)
    assert retrieved is not None
    assert retrieved["title"] == "Bug in query parsing"


def test_sync_state(temp_cache):
    temp_cache.set_last_fetch_timestamp("commits", "2023-01-01T00:00:00")
    ts = temp_cache.get_last_fetch_timestamp("commits")
    assert ts == "2023-01-01T00:00:00"
