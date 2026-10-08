"""
End-to-end ingestion pipeline.
Orchestrates data fetching, decision extraction, graph construction, and supersession detection.
"""

import logging
import time
from typing import Optional, Callable

from settings import Settings
from src.ingestion.github_fetcher import GitHubFetcher
from src.ingestion.cache import SQLiteCache
from src.extraction.context_collator import ContextCollator
from src.extraction.decision_extractor import DecisionExtractor
from src.extraction.code_entity_extractor import CodeEntityExtractor
from src.graph.neo4j_manager import Neo4jManager
from src.graph.embedding_generator import EmbeddingGenerator
from src.supersession.detector import SupersessionDetector

logger = logging.getLogger(__name__)


class IngestionPipeline:
    """End-to-end ingestion pipeline from GitHub to knowledge graph."""

    def __init__(self, settings: Settings):
        self.settings = settings
        self.cache = SQLiteCache()
        self.fetcher = GitHubFetcher(settings.github_token, settings.target_repo, self.cache)
        self.collator = ContextCollator()
        # Decision extraction powered by Google Gemini with multi-model cascade and Groq fallback
        gemini_fallbacks = [
            m.strip() for m in settings.gemini_fallback_models.split(",") if m.strip()
        ]
        groq_fallbacks = [
            m.strip() for m in settings.groq_fallback_models.split(",") if m.strip()
        ]
        self.decision_extractor = DecisionExtractor(
            api_key=settings.google_api_key,
            model=settings.gemini_model,
            fallback_models=gemini_fallbacks,
            groq_api_key=settings.groq_api_key,
            groq_fallback_models=groq_fallbacks,
            enable_cross_provider_fallback=settings.enable_cross_provider_fallback,
        )

        self.entity_extractor = CodeEntityExtractor()
        self.neo4j_mgr = Neo4jManager(settings.neo4j_uri, settings.neo4j_username, settings.neo4j_password)
        self.embedding_gen = EmbeddingGenerator(
            settings.google_api_key, settings.embedding_model, settings.embedding_dimensions
        )

        # Groq client for supersession detection and reasoning (openai/gpt-oss-120b)
        from groq import Groq
        self.groq_client = Groq(api_key=settings.groq_api_key)
        self.supersession_detector = SupersessionDetector(self.neo4j_mgr, self.groq_client, settings.groq_model)

    async def run(self, progress_callback: Optional[Callable] = None):
        """Runs the complete end-to-end pipeline."""

        def notify(msg: str, pct: float = 0.0):
            logger.info(msg)
            if progress_callback:
                progress_callback(msg, pct)

        try:
            # Step 1: Setup Neo4j schema
            notify("Setting up Neo4j schema...", 0.05)
            self.neo4j_mgr.setup_schema()

            # Step 2: Fetch commits
            notify("Fetching commits from GitHub...", 0.10)
            commits = self.fetcher.fetch_commits(limit=self.settings.max_commits)
            logger.info(f"Fetched {len(commits)} commits")

            # Step 3: Fetch PRs
            notify("Fetching pull requests from GitHub...", 0.20)
            prs = self.fetcher.fetch_pull_requests(limit=self.settings.max_prs)
            logger.info(f"Fetched {len(prs)} pull requests")

            # Step 4: Fetch issues
            notify("Fetching issues from GitHub...", 0.30)
            issues = self.fetcher.fetch_issues(limit=self.settings.max_issues)
            logger.info(f"Fetched {len(issues)} issues")

            # Step 5: Upsert raw data to Neo4j (commits, PRs, issues, people)
            notify("Writing raw data to Neo4j...", 0.35)
            for commit in commits:
                try:
                    self.neo4j_mgr.upsert_commit(commit)
                except Exception as e:
                    logger.error(f"Error upserting commit {commit.sha}: {e}")

            for pr in prs:
                try:
                    self.neo4j_mgr.upsert_pr(pr)
                except Exception as e:
                    logger.error(f"Error upserting PR {pr.number}: {e}")

            for issue in issues:
                try:
                    self.neo4j_mgr.upsert_issue(issue)
                except Exception as e:
                    logger.error(f"Error upserting issue {issue.number}: {e}")

            # Step 6: Extract decisions from PRs
            notify("Extracting decisions from pull requests...", 0.45)
            all_decisions = []
            for i, pr in enumerate(prs):
                try:
                    decision = self.decision_extractor.extract_from_pr(pr, self.collator)
                    if decision:
                        decision_id = f"DEC-PR-{pr.number}"
                        embedding = self.embedding_gen.embed_decision(decision)
                        self.neo4j_mgr.upsert_decision(
                            decision_id, decision, "pull_request", str(pr.number), embedding
                        )
                        # Link decision to PR, author, and affected entities
                        self.neo4j_mgr.link_decision(
                            decision_id,
                            pr_number=pr.number,
                            commit_sha=pr.commits[0] if pr.commits else None,
                            issue_number=None,
                            author=pr.author_login,
                            affected_entities=decision.impacted_entities,
                        )
                        all_decisions.append(decision)
                except Exception as e:
                    logger.error(f"Error extracting from PR {pr.number}: {e}")
                if (i + 1) % 10 == 0:
                    notify(f"Processed {i+1}/{len(prs)} PRs...", 0.45 + 0.15 * (i / len(prs)))

            # Step 7: Extract decisions from commits (batched)
            notify("Extracting decisions from commits...", 0.60)
            batch_size = self.settings.batch_size
            for i in range(0, len(commits), batch_size):
                batch = commits[i : i + batch_size]
                try:
                    batch_decisions = self.decision_extractor.extract_from_commits_batch(
                        batch, self.collator
                    )
                    for decision in batch_decisions:
                        decision_id = f"DEC-COMMIT-{batch[0].sha[:8]}"
                        embedding = self.embedding_gen.embed_decision(decision)
                        self.neo4j_mgr.upsert_decision(
                            decision_id, decision, "commit", batch[0].sha, embedding
                        )
                        self.neo4j_mgr.link_decision(
                            decision_id,
                            pr_number=None,
                            commit_sha=batch[0].sha,
                            issue_number=None,
                            author=batch[0].author_login,
                            affected_entities=decision.impacted_entities,
                        )
                        all_decisions.append(decision)
                except Exception as e:
                    logger.error(f"Error extracting from commit batch starting at {batch[0].sha}: {e}")

            # Step 8: Extract decisions from issues
            notify("Extracting decisions from issues...", 0.70)
            for issue in issues:
                try:
                    decision = self.decision_extractor.extract_from_issue(issue, self.collator)
                    if decision:
                        decision_id = f"DEC-ISSUE-{issue.number}"
                        embedding = self.embedding_gen.embed_decision(decision)
                        self.neo4j_mgr.upsert_decision(
                            decision_id, decision, "issue", str(issue.number), embedding
                        )
                        self.neo4j_mgr.link_decision(
                            decision_id,
                            pr_number=None,
                            commit_sha=None,
                            issue_number=issue.number,
                            author=issue.author_login,
                            affected_entities=decision.impacted_entities,
                        )
                        all_decisions.append(decision)
                except Exception as e:
                    logger.error(f"Error extracting from issue {issue.number}: {e}")

            # Step 9: Extract code entities from repo files
            notify("Extracting code entities...", 0.80)
            try:
                design_docs = self.fetcher.fetch_design_docs()
                for doc in design_docs:
                    content = doc.get("content", "")
                    path = doc.get("path", "")
                    if content and path:
                        entities = self.entity_extractor.extract_from_file(
                            path, content.encode("utf-8")
                        )
                        for entity in entities:
                            self.neo4j_mgr.upsert_code_entity(entity)
            except Exception as e:
                logger.error(f"Error extracting code entities: {e}")

            # Step 10: Run supersession detection
            notify("Running supersession detection...", 0.90)
            try:
                results = self.supersession_detector.run_all()
                logger.info(f"Supersession detection results: {results}")
            except Exception as e:
                logger.error(f"Error in supersession detection: {e}")

            notify("✅ Pipeline complete!", 1.0)
            logger.info(f"Total decisions extracted: {len(all_decisions)}")

        except Exception as e:
            logger.error(f"Pipeline failed: {e}", exc_info=True)
            raise
        finally:
            self.neo4j_mgr.close()

    async def run_quick(
        self,
        max_commits: int = 50,
        max_prs: int = 30,
        max_issues: int = 20,
        progress_callback: Optional[Callable] = None,
    ):
        """Runs the pipeline with reduced limits for quick testing."""
        old_c = self.settings.max_commits
        old_p = self.settings.max_prs
        old_i = self.settings.max_issues

        self.settings.max_commits = max_commits
        self.settings.max_prs = max_prs
        self.settings.max_issues = max_issues

        try:
            await self.run(progress_callback)
        finally:
            self.settings.max_commits = old_c
            self.settings.max_prs = old_p
            self.settings.max_issues = old_i
