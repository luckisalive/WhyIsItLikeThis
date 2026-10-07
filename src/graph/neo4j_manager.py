"""
Neo4j knowledge graph manager.
Handles AuraDB connection, schema initialization, entity upserts, vector search, and context traversals.
"""

import logging
from typing import Any, Dict, List, Optional, Tuple

from neo4j import GraphDatabase

from src.models import (
    CodeEntity,
    CommitData,
    DecisionStatus,
    ExtractedDecision,
    GraphStats,
    IssueData,
    PRData,
)

logger = logging.getLogger(__name__)


class Neo4jManager:
    """Manages all Neo4j AuraDB operations."""

    def __init__(self, uri: str, username: str, password: str):
        self.uri = uri
        self.username = username
        self.password = password
        self.driver = GraphDatabase.driver(uri, auth=(username, password))

    def connect(self) -> bool:
        """Verify driver connectivity to Neo4j AuraDB."""
        try:
            self.driver.verify_connectivity()
            logger.info("Successfully connected and verified connectivity to Neo4j.")
            return True
        except Exception as e:
            logger.error(f"Failed to connect to Neo4j: {e}")
            raise e

    def execute_query(self, query: str, parameters: Optional[Dict[str, Any]] = None) -> List[Dict[str, Any]]:
        """Convenience query runner returning results as a list of Python dictionaries."""
        params = parameters or {}
        try:
            records, _, _ = self.driver.execute_query(query, params)
            results = []
            for record in records:
                results.append(record.data())
            return results
        except Exception as e:
            logger.error(f"Error executing query: {e}")
            raise e

    def setup_schema(self) -> None:
        """Creates all constraints and indexes required for the knowledge graph."""
        queries = [
            "CREATE CONSTRAINT person_login IF NOT EXISTS FOR (p:Person) REQUIRE p.login IS UNIQUE",
            "CREATE CONSTRAINT commit_sha IF NOT EXISTS FOR (c:Commit) REQUIRE c.sha IS UNIQUE",
            "CREATE CONSTRAINT pr_number IF NOT EXISTS FOR (pr:PullRequest) REQUIRE pr.number IS UNIQUE",
            "CREATE CONSTRAINT issue_number IF NOT EXISTS FOR (i:Issue) REQUIRE i.number IS UNIQUE",
            "CREATE CONSTRAINT decision_id IF NOT EXISTS FOR (d:Decision) REQUIRE d.id IS UNIQUE",
            "CREATE CONSTRAINT code_entity_path IF NOT EXISTS FOR (e:CodeEntity) REQUIRE e.path IS UNIQUE",
            """
            CREATE VECTOR INDEX decision_embeddings IF NOT EXISTS
            FOR (d:Decision) ON (d.embedding)
            OPTIONS {indexConfig: {
                `vector.dimensions`: 768,
                `vector.similarity_function`: 'cosine'
            }}
            """,
            """
            CREATE FULLTEXT INDEX decision_fulltext IF NOT EXISTS
            FOR (n:Decision) ON EACH [n.title, n.rationale, n.decision_made]
            """,
            """
            CREATE FULLTEXT INDEX code_entity_fulltext IF NOT EXISTS
            FOR (n:CodeEntity) ON EACH [n.path, n.name]
            """,
        ]
        for query in queries:
            try:
                self.driver.execute_query(query)
                logger.info(f"Executed schema statement: {query.strip().splitlines()[0]}...")
            except Exception as e:
                logger.warning(f"Note on schema query: {e}")

    def upsert_decision(
        self,
        decision_id: str,
        decision: ExtractedDecision,
        source_type: str,
        source_id: str,
        embedding: List[float],
    ) -> None:
        """MERGE Decision node with all properties + vector embedding."""
        query = """
        MERGE (d:Decision {id: $id})
        SET d.title = $title,
            d.rationale = $rationale,
            d.problem_context = $problem_context,
            d.decision_made = $decision_made,
            d.status = $status,
            d.decision_type = $decision_type,
            d.type = $decision_type,
            d.confidence_score = $confidence_score,
            d.source_type = $source_type,
            d.source_id = $source_id,
            d.embedding = $embedding
        """
        status_val = (
            decision.status.value
            if hasattr(decision.status, "value")
            else str(getattr(decision, "status", "ACTIVE"))
        )
        type_val = (
            decision.decision_type.value
            if hasattr(decision.decision_type, "value")
            else str(getattr(decision, "type", "ARCHITECTURE"))
        )

        params = {
            "id": decision_id,
            "title": decision.title,
            "rationale": decision.rationale,
            "problem_context": decision.problem_context,
            "decision_made": decision.decision_made,
            "status": status_val,
            "decision_type": type_val,
            "confidence_score": decision.confidence_score,
            "source_type": source_type,
            "source_id": str(source_id),
            "embedding": embedding,
        }
        try:
            self.driver.execute_query(query, params)
            logger.info(f"Upserted Decision {decision_id}")
        except Exception as e:
            logger.error(f"Failed to upsert Decision {decision_id}: {e}")

    def upsert_commit(self, commit: CommitData) -> None:
        """MERGE Commit node, author Person node, and MODIFIES relationships for changed files."""
        query = """
        MERGE (c:Commit {sha: $sha})
        SET c.message = $message, c.date = $date
        MERGE (p:Person {login: $author_login})
        MERGE (c)-[:AUTHORED_BY]->(p)
        """
        author_login = commit.author_login or getattr(commit, "author", "Unknown") or "Unknown"
        date_str = str(commit.date) if commit.date else ""

        params = {
            "sha": commit.sha,
            "message": commit.message,
            "date": date_str,
            "author_login": author_login,
        }
        try:
            self.driver.execute_query(query, params)
            files = commit.files_changed or getattr(commit, "files", []) or []
            for file_change in files:
                file_query = """
                MATCH (c:Commit {sha: $sha})
                MERGE (e:CodeEntity {path: $path})
                MERGE (c)-[m:MODIFIES]->(e)
                SET m.status = $status, m.additions = $additions, m.deletions = $deletions
                """
                file_params = {
                    "sha": commit.sha,
                    "path": file_change.filename,
                    "status": file_change.status,
                    "additions": file_change.additions,
                    "deletions": file_change.deletions,
                }
                self.driver.execute_query(file_query, file_params)
            logger.info(f"Upserted Commit {commit.sha[:8]}")
        except Exception as e:
            logger.error(f"Failed to upsert Commit {commit.sha}: {e}")

    def upsert_pr(self, pr: PRData) -> None:
        """MERGE PullRequest node, author Person, CONTAINS commits, and HAS_REVIEW comments."""
        query = """
        MERGE (pr:PullRequest {number: $number})
        SET pr.title = $title, pr.body = $body, pr.state = $state,
            pr.created_at = $created_at, pr.url = $url
        MERGE (p:Person {login: $author_login})
        MERGE (pr)-[:AUTHORED_BY]->(p)
        """
        author_login = pr.author_login or getattr(pr, "author", "Unknown") or "Unknown"
        params = {
            "number": pr.number,
            "title": pr.title,
            "body": pr.body or "",
            "state": pr.state,
            "created_at": str(pr.created_at) if pr.created_at else "",
            "url": pr.url,
            "author_login": author_login,
        }
        try:
            self.driver.execute_query(query, params)
            for commit_sha in pr.commits:
                commit_query = """
                MATCH (pr:PullRequest {number: $number})
                MERGE (c:Commit {sha: $sha})
                MERGE (pr)-[:CONTAINS]->(c)
                """
                self.driver.execute_query(commit_query, {"number": pr.number, "sha": commit_sha})

            for comment in pr.review_comments:
                comment_query = """
                MATCH (pr:PullRequest {number: $number})
                MERGE (rc:ReviewComment {id: $comment_id})
                SET rc.body = $body, rc.created_at = $created_at, rc.path = $path
                MERGE (p:Person {login: $author_login})
                MERGE (rc)-[:AUTHORED_BY]->(p)
                MERGE (pr)-[:HAS_REVIEW]->(rc)
                """
                comment_params = {
                    "number": pr.number,
                    "comment_id": str(comment.id),
                    "body": comment.body,
                    "created_at": str(comment.created_at) if comment.created_at else "",
                    "path": comment.path,
                    "author_login": comment.author_login or getattr(comment, "author", "Unknown") or "Unknown",
                }
                self.driver.execute_query(comment_query, comment_params)
            logger.info(f"Upserted PR #{pr.number}")
        except Exception as e:
            logger.error(f"Failed to upsert PR #{pr.number}: {e}")

    def upsert_issue(self, issue: IssueData) -> None:
        """MERGE Issue node, author Person, and HAS_COMMENT for discussion comments."""
        query = """
        MERGE (i:Issue {number: $number})
        SET i.title = $title, i.body = $body, i.state = $state,
            i.created_at = $created_at, i.url = $url
        MERGE (p:Person {login: $author_login})
        MERGE (i)-[:AUTHORED_BY]->(p)
        """
        author_login = issue.author_login or getattr(issue, "author", "Unknown") or "Unknown"
        params = {
            "number": issue.number,
            "title": issue.title,
            "body": issue.body or "",
            "state": issue.state,
            "created_at": str(issue.created_at) if issue.created_at else "",
            "url": issue.url,
            "author_login": author_login,
        }
        try:
            self.driver.execute_query(query, params)
            for comment in issue.comments:
                comment_query = """
                MATCH (i:Issue {number: $number})
                MERGE (c:IssueComment {id: $comment_id})
                SET c.body = $body, c.created_at = $created_at
                MERGE (p:Person {login: $author_login})
                MERGE (c)-[:AUTHORED_BY]->(p)
                MERGE (i)-[:HAS_COMMENT]->(c)
                """
                comment_params = {
                    "number": issue.number,
                    "comment_id": str(comment.id),
                    "body": comment.body,
                    "created_at": str(comment.created_at) if comment.created_at else "",
                    "author_login": comment.author_login or getattr(comment, "author", "Unknown") or "Unknown",
                }
                self.driver.execute_query(comment_query, comment_params)
            logger.info(f"Upserted Issue #{issue.number}")
        except Exception as e:
            logger.error(f"Failed to upsert Issue #{issue.number}: {e}")

    def upsert_code_entity(self, entity: CodeEntity) -> None:
        """MERGE CodeEntity node (file, class, function, or config key)."""
        query = """
        MERGE (e:CodeEntity {path: $path})
        SET e.name = $name,
            e.entity_type = $entity_type,
            e.type = $entity_type,
            e.language = $language
        """
        entity_type_val = (
            entity.entity_type.value
            if hasattr(entity.entity_type, "value")
            else str(getattr(entity, "type", "file"))
        )
        params = {
            "path": entity.path,
            "name": entity.name,
            "entity_type": entity_type_val,
            "language": entity.language,
        }
        try:
            self.driver.execute_query(query, params)
            logger.info(f"Upserted CodeEntity {entity.path} ({entity.name})")
        except Exception as e:
            logger.error(f"Failed to upsert CodeEntity {entity.path}: {e}")

    def link_decision(
        self,
        decision_id: str,
        pr_number: Optional[int],
        commit_sha: Optional[str],
        issue_number: Optional[int],
        author: str,
        affected_entities: List[str],
    ) -> None:
        """Creates DISCUSSED_IN, IMPLEMENTED_IN, MOTIVATED_BY, MADE_BY, AFFECTS relationships."""
        try:
            if pr_number is not None:
                query = """
                MATCH (d:Decision {id: $decision_id}), (pr:PullRequest {number: $pr_number})
                MERGE (d)-[:DISCUSSED_IN]->(pr)
                """
                self.driver.execute_query(query, {"decision_id": decision_id, "pr_number": pr_number})

            if commit_sha:
                query = """
                MATCH (d:Decision {id: $decision_id}), (c:Commit {sha: $commit_sha})
                MERGE (d)-[:IMPLEMENTED_IN]->(c)
                """
                self.driver.execute_query(query, {"decision_id": decision_id, "commit_sha": commit_sha})

            if issue_number is not None:
                query = """
                MATCH (d:Decision {id: $decision_id}), (i:Issue {number: $issue_number})
                MERGE (d)-[:MOTIVATED_BY]->(i)
                """
                self.driver.execute_query(query, {"decision_id": decision_id, "issue_number": issue_number})

            if author:
                query_author = """
                MATCH (d:Decision {id: $decision_id})
                MERGE (p:Person {login: $author})
                MERGE (d)-[:MADE_BY]->(p)
                """
                self.driver.execute_query(query_author, {"decision_id": decision_id, "author": author})

            for entity in affected_entities:
                if entity:
                    query_entity = """
                    MATCH (d:Decision {id: $decision_id})
                    MERGE (e:CodeEntity {path: $entity})
                    MERGE (d)-[:AFFECTS]->(e)
                    """
                    self.driver.execute_query(query_entity, {"decision_id": decision_id, "entity": entity})

            logger.info(f"Linked decision {decision_id}")
        except Exception as e:
            logger.error(f"Failed to link decision {decision_id}: {e}")

    def mark_superseded(self, old_id: str, new_id: str, reason: str) -> None:
        """Creates SUPERSEDES edge and sets old decision status to SUPERSEDED."""
        query = """
        MATCH (new_d:Decision {id: $new_id}), (old_d:Decision {id: $old_id})
        MERGE (new_d)-[s:SUPERSEDES]->(old_d)
        SET s.reason = $reason,
            old_d.status = $status
        """
        params = {
            "new_id": new_id,
            "old_id": old_id,
            "reason": reason,
            "status": DecisionStatus.SUPERSEDED.value,
        }
        try:
            self.driver.execute_query(query, params)
            logger.info(f"Decision {new_id} SUPERSEDES {old_id} (Reason: {reason})")
        except Exception as e:
            logger.error(f"Failed to mark superseded {old_id} by {new_id}: {e}")

    def get_stats(self) -> GraphStats:
        """Counts all node types and relationships in the knowledge graph."""
        query = """
        CALL { MATCH (d:Decision) RETURN count(d) AS dec_total }
        CALL { MATCH (d:Decision {status: 'ACTIVE'}) RETURN count(d) AS dec_active }
        CALL { MATCH (d:Decision {status: 'SUPERSEDED'}) RETURN count(d) AS dec_super }
        CALL { MATCH (c:Commit) RETURN count(c) AS com_total }
        CALL { MATCH (pr:PullRequest) RETURN count(pr) AS pr_total }
        CALL { MATCH (i:Issue) RETURN count(i) AS iss_total }
        CALL { MATCH (p:Person) RETURN count(p) AS per_total }
        CALL { MATCH (e:CodeEntity) RETURN count(e) AS ent_total }
        CALL { MATCH ()-[r]->() RETURN count(r) AS rel_total }
        RETURN dec_total, dec_active, dec_super, com_total, pr_total, iss_total, per_total, ent_total, rel_total
        """
        try:
            records, _, _ = self.driver.execute_query(query)
            if records:
                row = records[0]
                total_nodes = (
                    row["dec_total"]
                    + row["com_total"]
                    + row["pr_total"]
                    + row["iss_total"]
                    + row["per_total"]
                    + row["ent_total"]
                )
                return GraphStats(
                    total_decisions=row["dec_total"],
                    active_decisions=row["dec_active"],
                    superseded_decisions=row["dec_super"],
                    total_commits=row["com_total"],
                    total_prs=row["pr_total"],
                    total_issues=row["iss_total"],
                    total_people=row["per_total"],
                    total_code_entities=row["ent_total"],
                    total_relationships=row["rel_total"],
                    nodes=total_nodes,
                    edges=row["rel_total"],
                )
        except Exception as e:
            logger.warning(f"Complex stats query failed: {e}. Falling back to simple node/edge count.")

        # Fallback simple count
        stats = GraphStats()
        try:
            n_res, _, _ = self.driver.execute_query("MATCH (n) RETURN count(n) AS count")
            e_res, _, _ = self.driver.execute_query("MATCH ()-[r]->() RETURN count(r) AS count")
            stats.nodes = n_res[0]["count"]
            stats.edges = e_res[0]["count"]
            stats.total_relationships = stats.edges
        except Exception as err:
            logger.error(f"Fallback stats error: {err}")
        return stats

    def vector_search(self, embedding: List[float], top_k: int = 5) -> List[Dict[str, Any]]:
        """Raw vector search on decision_embeddings index."""
        query = """
        CALL db.index.vector.queryNodes('decision_embeddings', $top_k, $embedding)
        YIELD node, score
        RETURN node, score
        """
        params = {"top_k": top_k, "embedding": embedding}
        results = []
        try:
            records, _, _ = self.driver.execute_query(query, params)
            for record in records:
                results.append({"node": dict(record["node"]), "score": record["score"]})
        except Exception as e:
            logger.error(f"Vector search failed: {e}")
        return results

    def get_decision_with_context(self, decision_id: str) -> Dict[str, Any]:
        """Fetches a decision with its full neighborhood context (PR, commits, issues, people, entities)."""
        query = """
        MATCH (d:Decision {id: $decision_id})
        OPTIONAL MATCH (d)-[:DISCUSSED_IN]->(pr:PullRequest)
        OPTIONAL MATCH (d)-[:IMPLEMENTED_IN]->(c:Commit)
        OPTIONAL MATCH (d)-[:MOTIVATED_BY]->(i:Issue)
        OPTIONAL MATCH (d)-[:MADE_BY]->(p:Person)
        OPTIONAL MATCH (d)-[:AFFECTS]->(e:CodeEntity)
        OPTIONAL MATCH (d)-[:SUPERSEDES*]->(old_d:Decision)
        OPTIONAL MATCH (new_d:Decision)-[:SUPERSEDES*]->(d)
        RETURN d, pr, c, i, p,
               collect(DISTINCT e) AS entities,
               collect(DISTINCT old_d) AS superseded,
               collect(DISTINCT new_d) AS superseded_by
        """
        try:
            records, _, _ = self.driver.execute_query(query, {"decision_id": decision_id})
            if not records:
                return {}
            record = records[0]
            return {
                "decision": dict(record["d"]) if record["d"] else None,
                "pr": dict(record["pr"]) if record["pr"] else None,
                "commit": dict(record["c"]) if record["c"] else None,
                "issue": dict(record["i"]) if record["i"] else None,
                "author": dict(record["p"]) if record["p"] else None,
                "entities": [dict(e) for e in record["entities"] if e],
                "superseded": [dict(d) for d in record["superseded"] if d],
                "superseded_by": [dict(d) for d in record["superseded_by"] if d],
            }
        except Exception as e:
            logger.error(f"Failed to get decision context for {decision_id}: {e}")
            return {}

    def find_overlapping_decisions(self) -> List[Tuple[Dict[str, Any], Dict[str, Any], Dict[str, Any]]]:
        """Finds pairs of ACTIVE decisions that AFFECT the same CodeEntity."""
        query = """
        MATCH (d1:Decision)-[:AFFECTS]->(e:CodeEntity)<-[:AFFECTS]-(d2:Decision)
        WHERE d1.id < d2.id
          AND d1.status = 'ACTIVE'
          AND d2.status = 'ACTIVE'
          AND NOT (d2)-[:SUPERSEDES]->(d1)
          AND NOT (d1)-[:SUPERSEDES]->(d2)
        RETURN d1, d2, e
        """
        results = []
        try:
            records, _, _ = self.driver.execute_query(query)
            for record in records:
                results.append((dict(record["d1"]), dict(record["d2"]), dict(record["e"])))
        except Exception as e:
            logger.error(f"Failed to find overlapping decisions: {e}")
        return results

    def close(self) -> None:
        """Close the Neo4j driver connection."""
        self.driver.close()
