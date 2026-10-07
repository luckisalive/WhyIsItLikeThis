"""
Supersession detector for architectural decisions.
Identifies when newer decisions explicitly or implicitly override older ones,
creating SUPERSEDES relationships and marking superseded decisions in Neo4j.
"""

import logging
import re
from typing import Any, Dict, List

from src.graph.neo4j_manager import Neo4jManager

logger = logging.getLogger(__name__)


class SupersessionDetector:
    """Detects when newer decisions override older ones through syntactic markers and semantic analysis."""

    def __init__(self, neo4j_mgr: Neo4jManager, llm_client: Any, model: str = "llama-3.1-70b-versatile"):
        self.neo4j_mgr = neo4j_mgr
        self.llm = llm_client
        self.model = model

    def detect_explicit(self) -> int:
        """Scan all decision titles, rationales, and PR bodies for explicit supersession keywords."""
        query = "MATCH (d:Decision) WHERE d.status = 'ACTIVE' RETURN d"
        try:
            records = self.neo4j_mgr.execute_query(query)
        except Exception as e:
            logger.error(f"Error fetching decisions for explicit supersession detection: {e}")
            return 0

        superseded_count = 0
        patterns = [
            r"supersedes\s+#?(\d+)",
            r"replaces\s+#?(\d+)",
            r"reverts\s+commit\s+([0-9a-fA-F]+)",
            r"reverts\s+#?(\d+)",
            r"deprecat(?:es|ing)\s+(.*)",
            r"replaces\s+(.*)",
        ]

        for record in records:
            d = record.get("d") or {}
            d_id = d.get("id")
            text = f"{d.get('title', '')} {d.get('rationale', '')} {d.get('decision_made', '')}".lower()

            for pattern in patterns:
                match = re.search(pattern, text)
                if match:
                    ref_target = match.group(1).strip()
                    try:
                        # Attempt to find an older decision related to the matched reference
                        ref_query = """
                        MATCH (old_d:Decision)
                        WHERE old_d.id <> $current_id
                          AND old_d.status = 'ACTIVE'
                          AND (old_d.source_id CONTAINS $ref_target
                               OR toLower(old_d.title) CONTAINS $ref_target)
                        RETURN old_d.id AS old_id
                        LIMIT 1
                        """
                        ref_records = self.neo4j_mgr.execute_query(ref_query, {"ref_target": ref_target, "current_id": d_id})
                        if ref_records:
                            old_id = ref_records[0]["old_id"]
                            self.neo4j_mgr.mark_superseded(
                                old_id=old_id,
                                new_id=d_id,
                                reason=f"Explicitly matches keyword pattern '{match.group(0)}'",
                            )
                            superseded_count += 1
                            break
                    except Exception as e:
                        logger.error(f"Error linking explicit supersession for {d_id}: {e}")

        return superseded_count

    def detect_implicit(self) -> int:
        """Find decisions that target the same CodeEntity and determine if the newer overrides the older."""
        candidates = self.neo4j_mgr.find_overlapping_decisions()
        superseded_count = 0

        for d1, d2, e in candidates:
            entity_name = e.get("name") or e.get("path", "unknown")
            d1_id = d1.get("id")
            d2_id = d2.get("id")

            # Try LLM verification if client is configured
            is_override = False
            if self.llm and hasattr(self.llm, "chat"):
                prompt = (
                    f"Evaluate these two software architecture decisions affecting code entity '{entity_name}':\n\n"
                    f"Earlier Decision ({d1_id}):\n"
                    f"Title: {d1.get('title', '')}\n"
                    f"Rationale: {d1.get('rationale', '')}\n\n"
                    f"Later Decision ({d2_id}):\n"
                    f"Title: {d2.get('title', '')}\n"
                    f"Rationale: {d2.get('rationale', '')}\n\n"
                    "Does the later decision REPLACE, OVERRIDE, or COEXIST with the earlier decision?\n"
                    "Answer strictly with one word: REPLACE, OVERRIDE, or COEXIST."
                )
                try:
                    response = self.llm.chat.completions.create(
                        model=self.model,
                        messages=[{"role": "user", "content": prompt}],
                        temperature=0.0,
                    )
                    content = response.choices[0].message.content.strip().upper()
                    if "REPLACE" in content or "OVERRIDE" in content:
                        is_override = True
                except Exception as err:
                    logger.debug(f"LLM implicit check skipped/failed: {err}")

            # Heuristic fallback if LLM is unavailable:
            if not is_override:
                d2_text = f"{d2.get('title', '')} {d2.get('rationale', '')}".lower()
                override_keywords = ["replace", "deprecate", "rewrite", "migrate", "remove", "supersede"]
                if any(kw in d2_text for kw in override_keywords):
                    is_override = True

            if is_override:
                try:
                    self.neo4j_mgr.mark_superseded(
                        old_id=d1_id,
                        new_id=d2_id,
                        reason=f"Detected architectural override regarding entity '{entity_name}'",
                    )
                    superseded_count += 1
                except Exception as err:
                    logger.error(f"Failed to record implicit supersession between {d1_id} and {d2_id}: {err}")

        return superseded_count

    def run_all(self) -> Dict[str, int]:
        """Runs both explicit and implicit supersession sweeps and returns statistics."""
        explicit_count = self.detect_explicit()
        implicit_count = self.detect_implicit()
        total = explicit_count + implicit_count
        logger.info(
            f"Supersession scan complete. Explicit: {explicit_count}, Implicit: {implicit_count}, Total: {total}"
        )
        return {
            "explicit_supersessions_detected": explicit_count,
            "implicit_supersessions_detected": implicit_count,
            "total_supersessions_detected": total,
        }

    def get_supersession_chain(self, decision_id: str) -> List[Dict[str, Any]]:
        """Traverse the full SUPERSEDES chain for a given decision ID."""
        query = """
        MATCH path = (new_d:Decision)-[:SUPERSEDES*]->(old_d:Decision {id: $decision_id})
        RETURN nodes(path) AS chain
        """
        try:
            records = self.neo4j_mgr.execute_query(query, {"decision_id": decision_id})
            chain_list = []
            if records:
                for record in records:
                    for node in record.get("chain", []):
                        chain_list.append(dict(node))
            return chain_list
        except Exception as e:
            logger.error(f"Error fetching supersession chain for {decision_id}: {e}")
            return []
