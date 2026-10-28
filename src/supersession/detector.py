"""
Supersession detector for architectural decisions.
Identifies when newer decisions explicitly or implicitly override older ones,
creating SUPERSEDES relationships and marking superseded decisions in Neo4j.
"""

import logging
import re
from typing import Any, Dict, List, Optional

from src.graph.neo4j_manager import Neo4jManager

logger = logging.getLogger(__name__)

REVERT_TITLE_PAT = re.compile(r"^revert\b", re.IGNORECASE)
REVERTS_PAT = re.compile(r"\breverts?\b[^\n\.\;]*?(?:[\w\-\.]+/[\w\-\.]+)?#(\d+)", re.IGNORECASE)
SUPERSEDES_PAT = re.compile(r"\bsupersedes\b[^\n\.\;]*?(?:[\w\-\.]+/[\w\-\.]+)?#(\d+)", re.IGNORECASE)
REPLACES_PAT = re.compile(r"\breplaces\b[^\n\.\;]*?(?:[\w\-\.]+/[\w\-\.]+)?#(\d+)", re.IGNORECASE)


def extract_explicit_matches(text: str, title: str = "") -> List[Dict[str, Any]]:
    """
    Extract explicit supersession targets from text and title.
    Rules:
      1. Title starting with "Revert" (targets parsed if present).
      2. "reverts #N" or "Reverts <repo>#N" with text allowed in between.
      3. "supersedes #N" where N is PR/issue number.
      4. "replaces #N" where N is PR/issue number.
    Does NOT match general prose like "Replace the list comprehension".
    """
    matches: List[Dict[str, Any]] = []
    seen = set()

    # Rule 1: Title starting with "Revert"
    clean_title = (title or "").strip()
    if REVERT_TITLE_PAT.match(clean_title):
        # Look for target number in title or text
        m_target = REVERTS_PAT.search(clean_title) or REVERTS_PAT.search(text or "")
        target_num = int(m_target.group(1)) if m_target else None
        evidence_text = m_target.group(0) if m_target else clean_title
        key = ("Title starting with 'Revert'", target_num, evidence_text)
        if key not in seen:
            seen.add(key)
            matches.append({
                "rule": "Title starting with 'Revert'",
                "target_num": target_num,
                "evidence_text": evidence_text,
            })

    # Rule 2, 3, 4: Scan title and text for reverts #N, supersedes #N, replaces #N
    combined_texts = [clean_title, text or ""]
    patterns = [
        ("reverts #N", REVERTS_PAT),
        ("supersedes #N", SUPERSEDES_PAT),
        ("replaces #N", REPLACES_PAT),
    ]

    for block in combined_texts:
        if not block:
            continue
        for rule_name, pat in patterns:
            for m in pat.finditer(block):
                target_num = int(m.group(1))
                evidence_text = m.group(0)
                key = (rule_name, target_num, evidence_text)
                if key not in seen:
                    seen.add(key)
                    matches.append({
                        "rule": rule_name,
                        "target_num": target_num,
                        "evidence_text": evidence_text,
                    })

    return matches


class SupersessionDetector:
    """Detects when newer decisions override older ones through syntactic markers and semantic analysis."""

    def __init__(self, neo4j_mgr: Neo4jManager, llm_client: Any, model: str = "llama-3.1-70b-versatile"):
        self.neo4j_mgr = neo4j_mgr
        self.llm = llm_client
        self.model = model

    def detect_explicit(self) -> int:
        """Scan PR titles/bodies and commit messages for explicit supersession keywords and create edges."""
        superseded_count = 0
        try:
            # Load PRs, Commits, Decisions with dates
            prs_q = "MATCH (pr:PullRequest) RETURN pr.number AS num, pr.title AS title, pr.body AS body, pr.created_at AS date"
            commits_q = "MATCH (c:Commit) RETURN c.sha AS sha, c.message AS msg, c.date AS date"
            decs_q = """
            MATCH (d:Decision)
            OPTIONAL MATCH (d)-[:DISCUSSED_IN]->(pr:PullRequest)
            OPTIONAL MATCH (d)-[:MOTIVATED_BY]->(iss:Issue)
            OPTIONAL MATCH (d)-[:IMPLEMENTED_IN]->(c:Commit)
            RETURN d.id AS id, d.title AS title,
                   coalesce(pr.created_at, iss.created_at, c.date) AS date
            """

            prs = self.neo4j_mgr.execute_query(prs_q) or []
            commits = self.neo4j_mgr.execute_query(commits_q) or []
            decs = {r["id"]: r for r in (self.neo4j_mgr.execute_query(decs_q) or []) if r.get("id")}

            raw_matches: List[Dict[str, Any]] = []

            # 1. Scan PRs
            for pr in prs:
                num = pr.get("num")
                if not num:
                    continue
                pr_matches = extract_explicit_matches(text=pr.get("body") or "", title=pr.get("title") or "")
                for m in pr_matches:
                    raw_matches.append({
                        "source_dec_id": f"DEC-PR-{num}",
                        "rule": m["rule"],
                        "target_num": m["target_num"],
                        "evidence_text": m["evidence_text"],
                    })

            # 2. Scan Commits
            for c in commits:
                sha = c.get("sha")
                if not sha:
                    continue
                c_msg = c.get("msg") or ""
                first_line = c_msg.split("\n")[0] if c_msg else ""
                c_matches = extract_explicit_matches(text=c_msg, title=first_line)
                for m in c_matches:
                    raw_matches.append({
                        "source_dec_id": f"DEC-COMMIT-{sha[:8]}",
                        "rule": m["rule"],
                        "target_num": m["target_num"],
                        "evidence_text": m["evidence_text"],
                    })

            # Resolve edges
            for m in raw_matches:
                t_num = m.get("target_num")
                if not t_num:
                    continue

                source_dec_id = m["source_dec_id"]
                target_dec_id = f"DEC-PR-{t_num}"
                target_dec = decs.get(target_dec_id)
                if not target_dec:
                    target_dec_id = f"DEC-ISSUE-{t_num}"
                    target_dec = decs.get(target_dec_id)

                src_dec = decs.get(source_dec_id)
                if not src_dec or not target_dec:
                    continue

                src_date = str(src_dec.get("date") or "")
                tgt_date = str(target_dec.get("date") or "")

                # Temporal constraint: newer source date strictly later than target date
                if src_date and tgt_date and src_date > tgt_date:
                    self.neo4j_mgr.mark_superseded(
                        old_id=target_dec_id,
                        new_id=source_dec_id,
                        reason=f"Explicit {m['rule']}: {m['evidence_text']}".strip(),
                        detection_type="EXPLICIT",
                        rule=m["rule"],
                        evidence_text=m["evidence_text"],
                    )
                    superseded_count += 1

        except Exception as e:
            logger.error(f"Error during explicit supersession detection: {e}", exc_info=True)

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

    def run_all(self, include_implicit: bool = False) -> Dict[str, int]:
        """Runs supersession sweeps and returns statistics. By default, only explicit runs."""
        explicit_count = self.detect_explicit()
        implicit_count = self.detect_implicit() if include_implicit else 0
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
