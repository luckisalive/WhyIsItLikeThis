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
) -> GraphResult:
    """
    Build graph nodes and edges for a decision's neighborhood.
    Stage 1: Pure extraction of baseline neighborhood.
    """
    if not neo4j_mgr:
        return GraphResult(
            error=ServiceError(
                code=ErrorCode.DB_UNREACHABLE,
                message="Neo4j connection not available.",
            )
        )

    try:
        context_data = neo4j_mgr.get_decision_with_context(decision_id)
        if not context_data or not context_data.get("decision"):
            return GraphResult(
                error=ServiceError(
                    code=ErrorCode.NO_RESULTS,
                    message=f"No decision found with ID: {decision_id}",
                )
            )

        dec_node = context_data.get("decision") or {}
        author = context_data.get("author")

        nodes: List[GraphNode] = []
        edges: List[GraphEdge] = []
        added_ids = set()

        # Root Decision Node
        root_id = str(decision_id)
        full_dec_label = dec_node.get("title") or root_id
        nodes.append(
            GraphNode(
                id=root_id,
                type="Decision",
                label=full_dec_label,
                properties={**dec_node, "real_id": root_id},
            )
        )
        added_ids.add(root_id)

        # Connected PR
        if context_data.get("pr"):
            pr = context_data["pr"]
            pr_num = pr.get("number")
            pr_id = f"PR-{pr_num}"
            nodes.append(
                GraphNode(
                    id=pr_id,
                    type="PullRequest",
                    label=f"PR #{pr_num}",
                    properties={**pr, "real_id": str(pr_num)},
                )
            )
            edges.append(GraphEdge(source=root_id, target=pr_id, label="DISCUSSED_IN"))
            added_ids.add(pr_id)

        # Connected Commit
        if context_data.get("commit"):
            commit = context_data["commit"]
            raw_sha = commit.get("sha", "")
            c_id = f"C-{raw_sha[:7]}"
            nodes.append(
                GraphNode(
                    id=c_id,
                    type="Commit",
                    label=c_id,
                    properties={**commit, "real_id": raw_sha},
                )
            )
            edges.append(GraphEdge(source=root_id, target=c_id, label="IMPLEMENTED_IN"))
            added_ids.add(c_id)

        # Connected Entities (Stage 1 uses baseline [:5])
        all_entities = context_data.get("entities", [])
        for ent in all_entities[:5]:
            ent_name = ent.get("name") or ent.get("path", "file")
            ent_id = f"ENT-{ent_name}"
            if ent_id not in added_ids:
                nodes.append(
                    GraphNode(
                        id=ent_id,
                        type="CodeEntity",
                        label=ent_name,
                        properties={**ent, "real_id": ent.get("path") or ent_name},
                    )
                )
                edges.append(GraphEdge(source=root_id, target=ent_id, label="AFFECTS"))
                added_ids.add(ent_id)

        hidden_count = max(0, len(all_entities) - 5)

        # Superseded Lineage (Older decisions)
        for sup in context_data.get("superseded", []):
            sup_id = sup.get("id", "")
            s_id = f"SUP-{sup_id}"
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

        return GraphResult(
            nodes=nodes,
            edges=edges,
            hidden_count=hidden_count,
            decision=dec_node,
            author=author,
        )
    except Exception as e:
        logger.error(f"Error building decision graph for {decision_id}: {e}")
        return GraphResult(
            error=ServiceError(
                code=ErrorCode.UNKNOWN,
                message=f"Error building decision graph: {e}",
            )
        )
