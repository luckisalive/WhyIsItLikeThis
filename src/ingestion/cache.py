import sqlite3
import json
import os
import logging
from typing import Optional, List, Dict, Any

logger = logging.getLogger(__name__)

class SQLiteCache:
    """SQLite-based local cache for GitHub API responses."""

    def __init__(self, db_path: str = "data/github_cache.db"):
        self.db_path = db_path
        os.makedirs(os.path.dirname(self.db_path), exist_ok=True)
        self.conn = sqlite3.connect(self.db_path, check_same_thread=False)
        self.conn.row_factory = sqlite3.Row
        self._init_db()

    def _init_db(self):
        """Auto-create tables for the cache."""
        with self.conn:
            self.conn.executescript('''
                CREATE TABLE IF NOT EXISTS commits (
                    id TEXT PRIMARY KEY,
                    data TEXT NOT NULL,
                    fetched_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                );
                CREATE TABLE IF NOT EXISTS pull_requests (
                    id TEXT PRIMARY KEY,
                    data TEXT NOT NULL,
                    fetched_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                );
                CREATE TABLE IF NOT EXISTS issues (
                    id TEXT PRIMARY KEY,
                    data TEXT NOT NULL,
                    fetched_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                );
                CREATE TABLE IF NOT EXISTS design_docs (
                    id TEXT PRIMARY KEY,
                    data TEXT NOT NULL,
                    fetched_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                );
                CREATE TABLE IF NOT EXISTS sync_state (
                    entity_type TEXT PRIMARY KEY,
                    last_fetch_timestamp TIMESTAMP NOT NULL
                );
            ''')
        logger.info(f"Initialized SQLite cache at {self.db_path}")

    def _get(self, table: str, entity_id: str) -> Optional[Dict[str, Any]]:
        cursor = self.conn.execute(f"SELECT data FROM {table} WHERE id = ?", (str(entity_id),))
        row = cursor.fetchone()
        if row:
            return json.loads(row['data'])
        return None

    def _store(self, table: str, entity_id: str, data: Dict[str, Any]):
        with self.conn:
            self.conn.execute(
                f"INSERT OR REPLACE INTO {table} (id, data, fetched_at) VALUES (?, ?, CURRENT_TIMESTAMP)",
                (str(entity_id), json.dumps(data))
            )

    def _get_all(self, table: str) -> List[Dict[str, Any]]:
        cursor = self.conn.execute(f"SELECT data FROM {table}")
        return [json.loads(row['data']) for row in cursor.fetchall()]

    def get_commit(self, sha: str) -> Optional[Dict[str, Any]]:
        """Get a cached commit by SHA."""
        return self._get('commits', sha)

    def store_commit(self, sha: str, data: Dict[str, Any]):
        """Store a commit in the cache."""
        self._store('commits', sha, data)

    def get_pr(self, number: int) -> Optional[Dict[str, Any]]:
        """Get a cached pull request by number."""
        return self._get('pull_requests', str(number))

    def store_pr(self, number: int, data: Dict[str, Any]):
        """Store a pull request in the cache."""
        self._store('pull_requests', str(number), data)

    def get_issue(self, number: int) -> Optional[Dict[str, Any]]:
        """Get a cached issue by number."""
        return self._get('issues', str(number))

    def store_issue(self, number: int, data: Dict[str, Any]):
        """Store an issue in the cache."""
        self._store('issues', str(number), data)

    def get_last_fetch_timestamp(self, entity_type: str) -> Optional[str]:
        """Get the last fetch timestamp for an entity type."""
        cursor = self.conn.execute("SELECT last_fetch_timestamp FROM sync_state WHERE entity_type = ?", (entity_type,))
        row = cursor.fetchone()
        return row['last_fetch_timestamp'] if row else None

    def set_last_fetch_timestamp(self, entity_type: str, ts: str):
        """Set the last fetch timestamp for an entity type."""
        with self.conn:
            self.conn.execute(
                "INSERT OR REPLACE INTO sync_state (entity_type, last_fetch_timestamp) VALUES (?, ?)",
                (entity_type, ts)
            )

    def get_all_commits(self) -> List[Dict[str, Any]]:
        """Get all cached commits."""
        return self._get_all('commits')

    def get_all_prs(self) -> List[Dict[str, Any]]:
        """Get all cached pull requests."""
        return self._get_all('pull_requests')

    def get_all_issues(self) -> List[Dict[str, Any]]:
        """Get all cached issues."""
        return self._get_all('issues')
