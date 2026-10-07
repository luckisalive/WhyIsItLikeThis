"""
Hybrid GraphRAG retriever.
Combines vector similarity search over Decision embeddings with Cypher graph traversal
to collect full chains of evidence (PRs, commits, issues, people, superseded lineage).
"""

import logging
from typing import Any, Dict, List

from src.graph.embedding_generator import EmbeddingGenerator
from src.graph.neo4j_manager import Neo4jManager

logger = logging.getLogger(__name__)


class WhyRetriever:
    """Hybrid retriever combining vector similarity search and deep graph traversal."""

    def __init__(self, neo4j_mgr: Neo4jManager, embedding_gen: EmbeddingGenerator):
        self.neo4j_mgr = neo4j_mgr
        self.embedding_gen = embedding_gen

    def retrieve(self, question: str, top_k: int = 5) -> List[Dict[str, Any]]:
        """Retrieve relevant decisions and connected evidence for a natural language question."""
        embedding = self.embedding_gen.embed_text(question)
        if not embedding:
            logger.warning("Failed to generate embedding for question.")
            return []

        # Primary query: Vector Index + Graph Traversal
        vector_query = """
        CALL db.index.vector.queryNodes('decision_embeddings', $top_k, $embedding)
        YIELD node AS decision, score
        OPTIONAL MATCH (decision)-[:DISCUSSED_IN]->(pr:PullRequest)
        OPTIONAL MATCH (decision)-[:IMPLEMENTED_IN]->(commit:Commit)
        OPTIONAL MATCH (decision)-[:MOTIVATED_BY]->(issue:Issue)
        OPTIONAL MATCH (decision)-[:MADE_BY]->(person:Person)
        OPTIONAL MATCH (decision)-[:AFFECTS]->(entity:CodeEntity)
        OPTIONAL MATCH (newer:Decision)-[:SUPERSEDES]->(decision)
        OPTIONAL MATCH (decision)-[:SUPERSEDES]->(older:Decision)
        RETURN decision, score,
               collect(DISTINCT pr) AS pull_requests,
               collect(DISTINCT commit) AS commits,
               collect(DISTINCT issue) AS issues,
               collect(DISTINCT person) AS people,
               collect(DISTINCT entity) AS code_entities,
               collect(DISTINCT newer) AS superseded_by,
               collect(DISTINCT older) AS supersedes
        ORDER BY score DESC
        """

        try:
            results = self.neo4j_mgr.execute_query(vector_query, {"embedding": embedding, "top_k": top_k})
            if results:
                return results
        except Exception as e:
            logger.warning(f"Vector search retrieval failed: {e}. Falling back to fulltext/keyword traversal.")

        # Fallback query: Fulltext or keyword matching on Decision nodes
        fallback_query = """
        MATCH (decision:Decision)
        WHERE any(word IN split(toLower($question), ' ') WHERE
              toLower(decision.title) CONTAINS word OR
              toLower(decision.rationale) CONTAINS word OR
              toLower(decision.decision_made) CONTAINS word)
        OPTIONAL MATCH (decision)-[:DISCUSSED_IN]->(pr:PullRequest)
        OPTIONAL MATCH (decision)-[:IMPLEMENTED_IN]->(commit:Commit)
        OPTIONAL MATCH (decision)-[:MOTIVATED_BY]->(issue:Issue)
        OPTIONAL MATCH (decision)-[:MADE_BY]->(person:Person)
        OPTIONAL MATCH (decision)-[:AFFECTS]->(entity:CodeEntity)
        OPTIONAL MATCH (newer:Decision)-[:SUPERSEDES]->(decision)
        OPTIONAL MATCH (decision)-[:SUPERSEDES]->(older:Decision)
        RETURN decision, 0.75 AS score,
               collect(DISTINCT pr) AS pull_requests,
               collect(DISTINCT commit) AS commits,
               collect(DISTINCT issue) AS issues,
               collect(DISTINCT person) AS people,
               collect(DISTINCT entity) AS code_entities,
               collect(DISTINCT newer) AS superseded_by,
               collect(DISTINCT older) AS supersedes
        LIMIT $top_k
        """
        try:
            results = self.neo4j_mgr.execute_query(fallback_query, {"question": question, "top_k": top_k})
            return results
        except Exception as e:
            logger.error(f"Fallback retrieval query failed: {e}")
            return []

    def format_context(self, results: List[Dict[str, Any]]) -> str:
        """Formats graph retrieval results into an explainable evidence context for the LLM."""
        context_parts = []
        for i, res in enumerate(results):
            decision = res.get("decision", {})
            if isinstance(decision, dict):
                d_dict = decision
            else:
                d_dict = dict(decision) if hasattr(decision, "items") else {}

            score = res.get("score", 0.0)
            title = d_dict.get("title", "Untitled Decision")
            rationale = d_dict.get("rationale", "No rationale documented.")
            decision_type = d_dict.get("decision_type", d_dict.get("type", "UNKNOWN"))
            status = d_dict.get("status", "ACTIVE")
            decision_made = d_dict.get("decision_made", "")
            problem_context = d_dict.get("problem_context", "")

            superseded_by = res.get("superseded_by", [])
            supersedes = res.get("supersedes", [])

            pr_urls = [
                pr.get("url") or f"PR #{pr.get('number')}"
                for pr in res.get("pull_requests", [])
                if isinstance(pr, dict) and (pr.get("url") or pr.get("number"))
            ]
            commit_shas = [
                c.get("sha")[:8]
                for c in res.get("commits", [])
                if isinstance(c, dict) and c.get("sha")
            ]
            issue_urls = [
                i.get("url") or f"Issue #{i.get('number')}"
                for i in res.get("issues", [])
                if isinstance(i, dict) and (i.get("url") or i.get("number"))
            ]
            people = [
                p.get("name") or p.get("login")
                for p in res.get("people", [])
                if isinstance(p, dict) and (p.get("name") or p.get("login"))
            ]
            entities = [
                e.get("path") or e.get("name")
                for e in res.get("code_entities", [])
                if isinstance(e, dict) and (e.get("path") or e.get("name"))
            ]

            part = f"--- Decision {i+1} (Relevance Score: {score:.3f}) ---\n"
            part += f"ID: {d_dict.get('id', 'N/A')}\n"
            part += f"Title: {title}\n"
            part += f"Type: {decision_type} | Status: {status}\n"
            if problem_context:
                part += f"Problem Context: {problem_context}\n"
            if decision_made:
                part += f"Decision Made: {decision_made}\n"
            part += f"Rationale: {rationale}\n"

            if superseded_by:
                titles = [
                    d.get("title") or d.get("id")
                    for d in superseded_by
                    if isinstance(d, dict)
                ]
                part += f"⚠️ WARNING: SUPERSEDED by newer decisions: {titles}\n"

            if supersedes:
                titles = [
                    d.get("title") or d.get("id")
                    for d in supersedes
                    if isinstance(d, dict)
                ]
                part += f"ℹ️ NOTE: This decision SUPERSEDES older decisions: {titles}\n"

            if pr_urls:
                part += f"Discussed in PRs: {', '.join(pr_urls)}\n"
            if commit_shas:
                part += f"Implemented in Commits: {', '.join(commit_shas)}\n"
            if issue_urls:
                part += f"Motivated by Issues: {', '.join(issue_urls)}\n"
            if people:
                part += f"Made by: {', '.join(people)}\n"
            if entities:
                part += f"Affects Code Entities: {', '.join(entities)}\n"

            context_parts.append(part)

        return "\n\n".join(context_parts)
