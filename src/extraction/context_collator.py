import logging
from typing import List
from src.models import PRData, CommitData, IssueData

logger = logging.getLogger(__name__)

class ContextCollator:
    """Collates context from various sources for LLM extraction."""
    
    def __init__(self, max_chars: int = 4000):
        """Initialize the context collator with a max character limit."""
        self.max_chars = max_chars

    def collate_pr(self, pr: PRData) -> str:
        """Merges PR title, body, top review comments, and files changed into a context string."""
        parts = []
        parts.append(f"PR Title: {pr.title}")
        parts.append(f"PR Body: {pr.body or ''}")
        
        if pr.files_changed:
            files_str = ", ".join([f.filename for f in pr.files_changed[:10]])
            parts.append(f"Files Changed: {files_str}")

        if pr.review_comments:
            comments_str = "\n".join([f"Comment: {c.body}" for c in pr.review_comments[:5]])
            parts.append(f"Review Comments:\n{comments_str}")

        text = "\n\n".join(parts)
        if len(text) > self.max_chars:
            logger.warning(f"Truncating PR {pr.id} context from {len(text)} to {self.max_chars} chars.")
            text = text[:self.max_chars]
            
        return text

    def collate_commit(self, commit: CommitData) -> str:
        """Formats commit message and files changed."""
        parts = []
        parts.append(f"Commit Message: {commit.message}")
        
        if commit.files_changed:
            files_str = ", ".join([f.filename for f in commit.files_changed[:10]])
            parts.append(f"Files Changed: {files_str}")
        
        text = "\n\n".join(parts)
        if len(text) > self.max_chars:
            logger.warning(f"Truncating commit {commit.id} context.")
            text = text[:self.max_chars]
        return text

    def collate_issue(self, issue: IssueData) -> str:
        """Merges issue title, body, and comments."""
        parts = []
        parts.append(f"Issue Title: {issue.title}")
        parts.append(f"Issue Body: {issue.body or ''}")
        
        if issue.comments:
            comments_str = "\n".join([f"Comment: {c.body}" for c in issue.comments[:5]])
            parts.append(f"Comments:\n{comments_str}")
            
        text = "\n\n".join(parts)
        if len(text) > self.max_chars:
            logger.warning(f"Truncating issue {issue.id} context.")
            text = text[:self.max_chars]
        return text

    def collate_batch(self, commits: List[CommitData]) -> str:
        """Batches multiple commits into one context block."""
        parts = []
        for i, commit in enumerate(commits):
            parts.append(f"--- Commit {i+1} ---")
            parts.append(self.collate_commit(commit))
        
        text = "\n\n".join(parts)
        if len(text) > self.max_chars:
            logger.warning("Truncating batched commits context.")
            text = text[:self.max_chars]
        return text
