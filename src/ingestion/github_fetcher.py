import time
import logging
from typing import List, Dict, Any, Optional
from github import Github, RateLimitExceededException
from github.Repository import Repository
from github.Commit import Commit
from github.PullRequest import PullRequest
from github.Issue import Issue

from src.models import (
    CommitData, FileChange, PRData, CommentData, ReviewCommentData, IssueData
)
from src.ingestion.cache import SQLiteCache

logger = logging.getLogger(__name__)

class GitHubFetcher:
    """Fetches data from GitHub using PyGithub and caches it."""

    def __init__(self, token: str, repo_name: str, cache: SQLiteCache):
        self.gh = Github(token)
        self.repo_name = repo_name
        self.cache = cache
        self._repo: Optional[Repository] = None

    @property
    def repo(self) -> Repository:
        if self._repo is None:
            self._repo = self.gh.get_repo(self.repo_name)
        return self._repo

    def _check_rate_limit(self):
        """Check remaining quota and auto-sleep if approaching limit."""
        try:
            core_rate = self.gh.get_rate_limit().core
            if core_rate.remaining < 50:
                sleep_time = (core_rate.reset.timestamp() - time.time()) + 5
                if sleep_time > 0:
                    logger.warning(f"Rate limit approaching. Sleeping for {sleep_time:.2f} seconds...")
                    time.sleep(sleep_time)
        except Exception as e:
            logger.error(f"Error checking rate limit: {e}")

    def _is_significant(self, commit: Commit) -> bool:
        """Filter out noise commits."""
        author_login = commit.author.login.lower() if commit.author and commit.author.login else ""
        if "bot" in author_login or "dependabot" in author_login or "renovate" in author_login:
            return False

        message = commit.commit.message or ""
        
        # Check files
        if commit.files:
            all_trivial = True
            for f in commit.files:
                if not (f.filename.endswith('package-lock.json') or f.filename.endswith('yarn.lock') or f.filename.endswith('poetry.lock')):
                    all_trivial = False
                    break
            if all_trivial:
                return False
                
        # Merge commits with empty body (often trivial)
        if len(commit.parents) > 1 and len(message.strip().split('\n')) <= 1:
            return False

        return True

    def fetch_commits(self, limit: int = 500) -> List[CommitData]:
        """Fetches commits, caches them, returns parsed models."""
        commits_data = []
        try:
            self._check_rate_limit()
            paginated_commits = self.repo.get_commits()
            count = 0
            for commit in paginated_commits:
                if count >= limit:
                    break
                    
                sha = commit.sha
                cached = self.cache.get_commit(sha)
                
                if cached:
                    commits_data.append(CommitData(**cached))
                    count += 1
                    continue

                if not self._is_significant(commit):
                    continue

                author_name = commit.commit.author.name if commit.commit.author else "Unknown"
                
                files_changed = []
                for f in (commit.files or []):
                    files_changed.append(FileChange(
                        filename=f.filename,
                        status=f.status,
                        additions=f.additions,
                        deletions=f.deletions
                    ))

                c_data = CommitData(
                    sha=sha,
                    message=commit.commit.message or "",
                    author=author_name,
                    date=commit.commit.author.date.isoformat() if commit.commit.author else "",
                    files_changed=files_changed
                )
                
                self.cache.store_commit(sha, c_data.model_dump())
                commits_data.append(c_data)
                count += 1
                
                if count % 50 == 0:
                    logger.info(f"Fetched {count} commits...")
                    self._check_rate_limit()

        except Exception as e:
            logger.error(f"Error fetching commits: {e}")
            
        return commits_data

    def fetch_pull_requests(self, limit: int = 300) -> List[PRData]:
        """Fetches closed/merged PRs, parses and caches them."""
        prs_data = []
        try:
            self._check_rate_limit()
            paginated_prs = self.repo.get_pulls(state='closed', sort='updated', direction='desc')
            count = 0
            
            for pr in paginated_prs:
                if count >= limit:
                    break

                cached = self.cache.get_pr(pr.number)
                if cached:
                    prs_data.append(PRData(**cached))
                    count += 1
                    continue

                author_login = pr.user.login if pr.user else "Unknown"
                labels = [label.name for label in pr.labels] if pr.labels else []
                
                commits = [c.sha for c in pr.get_commits()]
                files_changed = []
                for f in pr.get_files():
                    files_changed.append(FileChange(
                        filename=f.filename,
                        status=f.status,
                        additions=f.additions,
                        deletions=f.deletions
                    ))
                
                comments = []
                for c in pr.get_issue_comments():
                    comments.append(CommentData(
                        id=str(c.id),
                        author=c.user.login if c.user else "Unknown",
                        body=c.body or "",
                        created_at=c.created_at.isoformat() if c.created_at else ""
                    ))
                
                review_comments = []
                for rc in pr.get_review_comments():
                    review_comments.append(ReviewCommentData(
                        id=str(rc.id),
                        author=rc.user.login if rc.user else "Unknown",
                        body=rc.body or "",
                        created_at=rc.created_at.isoformat() if rc.created_at else "",
                        path=rc.path,
                        commit_id=rc.commit_id
                    ))

                pr_model = PRData(
                    number=pr.number,
                    title=pr.title or "",
                    body=pr.body or "",
                    author=author_login,
                    state=pr.state,
                    url=pr.html_url,
                    created_at=pr.created_at.isoformat() if pr.created_at else "",
                    closed_at=pr.closed_at.isoformat() if pr.closed_at else None,
                    merged_at=pr.merged_at.isoformat() if pr.merged_at else None,
                    labels=labels,
                    commits=commits,
                    files_changed=files_changed,
                    comments=comments,
                    review_comments=review_comments
                )
                
                self.cache.store_pr(pr.number, pr_model.model_dump())
                prs_data.append(pr_model)
                count += 1
                
                if count % 20 == 0:
                    logger.info(f"Fetched {count} PRs...")
                    self._check_rate_limit()

        except Exception as e:
            logger.error(f"Error fetching PRs: {e}")
            
        return prs_data

    def fetch_issues(self, limit: int = 200) -> List[IssueData]:
        """Fetches closed issues, filtering out PRs."""
        issues_data = []
        try:
            self._check_rate_limit()
            paginated_issues = self.repo.get_issues(state='closed', sort='updated', direction='desc')
            count = 0
            
            for issue in paginated_issues:
                if count >= limit:
                    break

                if issue.pull_request:
                    continue

                cached = self.cache.get_issue(issue.number)
                if cached:
                    issues_data.append(IssueData(**cached))
                    count += 1
                    continue

                author_login = issue.user.login if issue.user else "Unknown"
                labels = [label.name for label in issue.labels] if issue.labels else []
                
                comments = []
                for c in issue.get_comments():
                    comments.append(CommentData(
                        id=str(c.id),
                        author=c.user.login if c.user else "Unknown",
                        body=c.body or "",
                        created_at=c.created_at.isoformat() if c.created_at else ""
                    ))

                issue_model = IssueData(
                    number=issue.number,
                    title=issue.title or "",
                    body=issue.body or "",
                    author=author_login,
                    state=issue.state,
                    url=issue.html_url,
                    created_at=issue.created_at.isoformat() if issue.created_at else "",
                    closed_at=issue.closed_at.isoformat() if issue.closed_at else None,
                    labels=labels,
                    comments=comments
                )
                
                self.cache.store_issue(issue.number, issue_model.model_dump())
                issues_data.append(issue_model)
                count += 1
                
                if count % 20 == 0:
                    logger.info(f"Fetched {count} issues...")
                    self._check_rate_limit()

        except Exception as e:
            logger.error(f"Error fetching issues: {e}")
            
        return issues_data

    def fetch_design_docs(self) -> List[Dict[str, Any]]:
        """Looks for doc files in the repo."""
        docs = []
        try:
            self._check_rate_limit()
            target_dirs = ["docs", "adr", "decisions"]
            target_files = ["CHANGES.md", "HISTORY.md", "CHANGELOG.md"]
            
            contents = self.repo.get_contents("")
            while contents:
                file_content = contents.pop(0)
                if file_content.type == "dir":
                    if file_content.path.split('/')[0] in target_dirs:
                        contents.extend(self.repo.get_contents(file_content.path))
                elif file_content.type == "file":
                    is_target = False
                    path_parts = file_content.path.split('/')
                    if path_parts[0] in target_dirs and file_content.name.endswith('.md'):
                        is_target = True
                    elif file_content.name in target_files:
                        is_target = True
                        
                    if is_target:
                        try:
                            # GitHub API limits size for direct content fetch, fallback if needed
                            decoded_content = file_content.decoded_content.decode('utf-8')
                            docs.append({
                                "path": file_content.path,
                                "content": decoded_content,
                                "sha": file_content.sha
                            })
                        except Exception as e:
                            logger.warning(f"Could not decode content for {file_content.path}: {e}")
                            
        except Exception as e:
            logger.error(f"Error fetching design docs: {e}")
            
        return docs
