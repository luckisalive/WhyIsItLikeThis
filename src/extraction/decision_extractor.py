"""
Architectural decision extractor powered by Groq (openai/gpt-oss-120b) with rule-based fallback.
Parses PR discussions, commit messages, and issues to extract structured ArchitecturalDecisionRecord models.
"""

import json
import logging
import re
import time
from typing import List, Optional

from groq import Groq

from src.extraction.context_collator import ContextCollator
from src.models import (
    Alternative,
    CommitData,
    DecisionType,
    ExtractedDecision,
    IssueData,
    PRData,
)

logger = logging.getLogger(__name__)


class DecisionExtractor:
    """Uses Groq openai/gpt-oss-120b to extract structured architectural decisions from development history."""

    SYSTEM_PROMPT = """
You are an expert Software Architect analyzing Git development history.
Extract structured Architectural Decision Records (ADRs) from the provided pull request or commit discussion.

Rules:
1. Focus strictly on *why* technical trade-offs were made, not just *what* code changed.
2. Ignore trivial changes (formatting, typo fixes, lockfiles, dependency patch bumps).
3. Capture rejected alternatives and design trade-offs discussed by contributors.
4. Detect if this decision overrides, reverts, or replaces an earlier decision.
5. If the text does NOT contain a meaningful technical decision, return: {"confidence_score": 0.0}
6. Return ONLY a valid JSON object matching this schema:
{
  "title": "Imperative summary of what was decided (e.g. Use Starlette as underlying ASGI framework)",
  "type": "ARCHITECTURE" | "API_DESIGN" | "PERFORMANCE" | "SECURITY" | "DEPRECATION" | "CONFIGURATION" | "DEPENDENCY" | "FEATURE" | "BUG_FIX" | "REFACTOR",
  "problem_context": "The limitation, bottleneck, or requirement that triggered this",
  "decision_made": "Specific implementation choice that was adopted",
  "rationale": "The reasoning, theoretical or empirical justification for choosing this approach",
  "alternatives_considered": [
    {"alternative": "Alternative option discussed", "rejection_reason": "Why it was not chosen"}
  ],
  "supersedes_references": ["PR #", "Commit SHA", "ADR ID"],
  "trade_offs": ["Trade-off or downside accepted"],
  "impacted_entities": ["file path", "class name", "function name"],
  "confidence_score": 0.0 to 1.0,
  "people_involved": ["GitHub usernames"]
}
"""

    def __init__(self, api_key: str, model: str = "openai/gpt-oss-120b"):
        self.api_key = api_key
        self.model = model
        self._is_placeholder = not api_key or api_key in ("GROQ_API_KEY", "YOUR_GROQ_API_KEY")
        self.client = None
        if not self._is_placeholder:
            try:
                self.client = Groq(api_key=api_key)
                logger.info(f"Initialized Groq client with model: {model}")
            except Exception as e:
                logger.warning(f"Could not initialize Groq client: {e}")

    def extract_from_text(self, text: str, source_type: str, source_id: str) -> Optional[ExtractedDecision]:
        """Sends discussion text to Groq LLM with retries, falling back to heuristics if offline."""
        if not text or len(text.strip()) < 30:
            return None

        if self.client and not self._is_placeholder:
            retries = 3
            backoff = 2
            for attempt in range(retries):
                try:
                    response = self.client.chat.completions.create(
                        model=self.model,
                        messages=[
                            {"role": "system", "content": self.SYSTEM_PROMPT},
                            {"role": "user", "content": f"Context for {source_type} {source_id}:\n{text}"},
                        ],
                        response_format={"type": "json_object"},
                        max_tokens=4096,  # Generous headroom for reasoning tokens + JSON content
                        temperature=0.1,
                    )
                    content = response.choices[0].message.content or ""
                    clean_content = content.strip()
                    if clean_content.startswith("```json"):
                        clean_content = clean_content[7:-3].strip()
                    elif clean_content.startswith("```"):
                        clean_content = clean_content[3:-3].strip()

                    data = json.loads(clean_content)
                    if data.get("confidence_score", 0.0) < 0.3:
                        return None

                    return ExtractedDecision.model_validate(data)

                except Exception as e:
                    logger.debug(f"Groq extraction attempt {attempt + 1} for {source_type} {source_id}: {e}")
                    if attempt < retries - 1:
                        time.sleep(backoff)
                        backoff *= 2

        # Fallback heuristic extractor
        return self._heuristic_extract(text, source_type, source_id)

    def extract_from_pr(self, pr: PRData, collator: ContextCollator) -> Optional[ExtractedDecision]:
        """Extracts decision from a PR thread."""
        text = collator.collate_pr(pr)
        return self.extract_from_text(text, "pull_request", str(pr.number))

    def extract_from_commits_batch(
        self, commits: List[CommitData], collator: ContextCollator
    ) -> List[ExtractedDecision]:
        """Extracts decisions from a batch of commits."""
        text = collator.collate_batch(commits)
        batch_id = ",".join(c.sha[:7] for c in commits[:3]) + ("..." if len(commits) > 3 else "")
        decision = self.extract_from_text(text, "commit_batch", batch_id)
        return [decision] if decision else []

    def extract_from_issue(self, issue: IssueData, collator: ContextCollator) -> Optional[ExtractedDecision]:
        """Extracts decision from an issue thread."""
        text = collator.collate_issue(issue)
        return self.extract_from_text(text, "issue", str(issue.number))

    def _heuristic_extract(self, text: str, source_type: str, source_id: str) -> Optional[ExtractedDecision]:
        """Rule-based extractor for architectural decisions when LLM is unavailable or for offline testing."""
        decision_keywords = [
            "decided to",
            "chosen to",
            "in order to",
            "switch to",
            "replace",
            "deprecate",
            "support async",
            "performance",
            "architecture",
            "refactor",
            "design",
        ]
        text_lower = text.lower()
        matched_kw = [kw for kw in decision_keywords if kw in text_lower]

        if not matched_kw:
            return None

        # Extract title from first line
        lines = [line.strip() for line in text.splitlines() if line.strip()]
        first_line = lines[0] if lines else f"Update in {source_type} #{source_id}"
        clean_title = re.sub(r"^(PR Title:|Commit Message:|Issue Title:)\s*", "", first_line)

        # Infer decision type
        if "perf" in text_lower or "speed" in text_lower or "benchmark" in text_lower:
            dec_type = DecisionType.PERFORMANCE
        elif "security" in text_lower or "auth" in text_lower or "vulnerability" in text_lower:
            dec_type = DecisionType.SECURITY
        elif "deprecat" in text_lower or "remove" in text_lower:
            dec_type = DecisionType.DEPRECATION
        elif "config" in text_lower:
            dec_type = DecisionType.CONFIGURATION
        else:
            dec_type = DecisionType.ARCHITECTURE

        return ExtractedDecision(
            title=clean_title[:120],
            decision_type=dec_type,
            problem_context=f"Recorded during repository evolution in {source_type} #{source_id}",
            decision_made=clean_title,
            rationale=f"Selected due to {matched_kw[0]} as documented in project discussions.",
            confidence_score=0.6,
            people_involved=[],
            impacted_entities=[],
        )
