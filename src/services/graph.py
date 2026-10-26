"""
Service for generating decision subgraphs.
Framework-agnostic: no Streamlit imports or session state.
"""

from __future__ import annotations

import logging
from typing import Any, Dict, List, Optional

from src.graph.neo4j_manager import Neo4jManager
from src.services.types import ErrorCode, GraphEdge, GraphNode, GraphResult, ServiceError

logger = logging.getLogger(__name__)


def get_decision_graph(
    neo4j_mgr: Optional[Neo4jManager],
    decision_id: str,
    entity_limit: int = 5,
) -> GraphResult:
    """
    Build graph nodes and edges for a decision's neighborhood in one query.

    Includes:
      - Decision (root)
      - PullRequest (DISCUSSED_IN)
      - Commit (IMPLEMENTED_IN)
      - Issue (MOTIVATED_BY)
      - Person (MADE_BY)
      - Older superseded decisions (SUPERSEDES)
      - Newer superseding decisions (incoming SUPERSEDES)
      - Code entities up to entity_limit (AFFECTS) with hidden_count tracking.

    Every node carries untruncated full label, real ID in properties, and type.
    """
    if not neo4j_mgr:
        return GraphResult(
            error=ServiceError(
                code=ErrorCode.DB_UNREACHABLE,
                message="Neo4j connection not available.",
            )
        )

    clean_id = decision_id.strip()
    if not clean_id:
        return GraphResult(
            error=ServiceError(
                code=ErrorCode.NO_RESULTS,
                message="Decision ID must not be empty.",
            )
        )

    try:
        context_data = neo4j_mgr.get_decision_with_context(clean_id)
        if not context_data or not context_data.get("decision"):
            return GraphResult(
                error=ServiceError(
                    code=ErrorCode.NO_RESULTS,
                    message=f"No decision found with ID: {clean_id}",
                )
            )

        dec_node = context_data.get("decision") or {}
        author = context_data.get("author")

        nodes: List[GraphNode] = []
        edges: List[GraphEdge] = []
        added_ids = set()

        # 1. Root Decision Node
        root_id = f"DEC-{clean_id}"
        full_dec_label = dec_node.get("title") or clean_id
        nodes.append(
            GraphNode(
                id=root_id,
                type="Decision",
                label=full_dec_label,
                properties={**dec_node, "real_id": clean_id},
            )
        )
        added_ids.add(root_id)

        # 2. Connected Pull Request
        if context_data.get("pr"):
            pr = context_data["pr"]
            pr_num = str(pr.get("number", ""))
            pr_id = f"PR-{pr_num}"
            pr_title = pr.get("title", "")
            pr_label = f"PR #{pr_num}: {pr_title}" if pr_title else f"PR #{pr_num}"
            if pr_id not in added_ids:
                nodes.append(
                    GraphNode(
                        id=pr_id,
                        type="PullRequest",
                        label=pr_label,
                        properties={**pr, "real_id": pr_num},
                    )
                )
                edges.append(GraphEdge(source=root_id, target=pr_id, label="DISCUSSED_IN"))
                added_ids.add(pr_id)

        # 3. Connected Commit
        if context_data.get("commit"):
            commit = context_data["commit"]
            raw_sha = commit.get("sha", "")
            c_id = f"COMMIT-{raw_sha}"
            msg = commit.get("message", "").splitlines()[0] if commit.get("message") else ""
            c_label = f"Commit {raw_sha[:7]}: {msg}" if msg else f"Commit {raw_sha[:7]}"
            if c_id not in added_ids:
                nodes.append(
                    GraphNode(
                        id=c_id,
                        type="Commit",
                        label=c_label,
                        properties={**commit, "real_id": raw_sha},
                    )
                )
                edges.append(GraphEdge(source=root_id, target=c_id, label="IMPLEMENTED_IN"))
                added_ids.add(c_id)

        # 4. Connected Issue
        if context_data.get("issue"):
            issue = context_data["issue"]
            issue_num = str(issue.get("number", ""))
            issue_id = f"ISSUE-{issue_num}"
            iss_title = issue.get("title", "")
            issue_label = f"Issue #{issue_num}: {iss_title}" if iss_title else f"Issue #{issue_num}"
            if issue_id not in added_ids:
                nodes.append(
                    GraphNode(
                        id=issue_id,
                        type="Issue",
                        label=issue_label,
                        properties={**issue, "real_id": issue_num},
                    )
                )
                edges.append(GraphEdge(source=root_id, target=issue_id, label="MOTIVATED_BY"))
                added_ids.add(issue_id)

        # 5. Connected Person (Author)
        if author:
            login = author.get("login", "")
            p_id = f"PERSON-{login}"
            p_name = author.get("name")
            p_label = f"@{login} ({p_name})" if p_name else f"@{login}"
            if p_id not in added_ids:
                nodes.append(
                    GraphNode(
                        id=p_id,
                        type="Person",
                        label=p_label,
                        properties={**author, "real_id": login},
                    )
                )
                edges.append(GraphEdge(source=root_id, target=p_id, label="MADE_BY"))
                added_ids.add(p_id)

        # 6. Connected Code Entities (Truncate to entity_limit, track hidden_count)
        all_entities = context_data.get("entities", [])
        for ent in all_entities[:entity_limit]:
            ent_name = ent.get("name") or ent.get("path", "file")
            ent_path = ent.get("path") or ent_name
            ent_id = f"ENTITY-{ent_path}"
            if ent_id not in added_ids:
                nodes.append(
                    GraphNode(
                        id=ent_id,
                        type="CodeEntity",
                        label=ent_name,
                        properties={**ent, "real_id": ent_path},
                    )
                )
                edges.append(GraphEdge(source=root_id, target=ent_id, label="AFFECTS"))
                added_ids.add(ent_id)

        hidden_count = max(0, len(all_entities) - entity_limit)

        # 7. Older Superseded Decisions
        for sup in context_data.get("superseded", []):
            sup_id = sup.get("id", "")
            s_id = f"DEC-{sup_id}"
            if s_id not in added_ids:
                nodes.append(
                    GraphNode(
                        id=s_id,
                        type="Decision",
                        label=f"Old: {sup.get('title', sup_id)}",
                        properties={**sup, "real_id": sup_id},
                    )
                )
                edges.append(GraphEdge(source=root_id, target=s_id, label="SUPERSEDES"))
                added_ids.add(s_id)

        # 8. Newer Superseding Decisions
        for sup_by in context_data.get("superseded_by", []):
            sup_by_id = sup_by.get("id", "")
            s_by_id = f"DEC-{sup_by_id}"
            if s_by_id not in added_ids:
                nodes.append(
                    GraphNode(
                        id=s_by_id,
                        type="Decision",
                        label=f"New: {sup_by.get('title', sup_by_id)}",
                        properties={**sup_by, "real_id": sup_by_id},
                    )
                )
                edges.append(GraphEdge(source=s_by_id, target=root_id, label="SUPERSEDES"))
                added_ids.add(s_by_id)

        return GraphResult(
            nodes=nodes,
            edges=edges,
            hidden_count=hidden_count,
            decision=dec_node,
            author=author,
        )
    except Exception as e:
        logger.error(f"Error building decision graph for {clean_id}: {e}")
        return GraphResult(
            error=ServiceError(
                code=ErrorCode.DB_UNREACHABLE,
                message=f"Database error building decision graph: {e}",
            )
        )
