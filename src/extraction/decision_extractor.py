"""
Architectural decision extractor powered by Google Gemini (gemini-3.5-flash-lite) with rule-based fallback.
Parses PR discussions, commit messages, and issues to extract structured ArchitecturalDecisionRecord models.
"""

import json
import logging
import re
import time
from typing import Any, List, Optional

logger = logging.getLogger(__name__)

# Modern google.genai SDK
try:
    from google import genai
    from google.genai import types
    HAS_NEW_GENAI = True
except ImportError:
    HAS_NEW_GENAI = False

# Legacy fallback
try:
    import warnings
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", FutureWarning)
        import google.generativeai as legacy_genai
    HAS_LEGACY_GENAI = True
except ImportError:
    HAS_LEGACY_GENAI = False

from src.extraction.context_collator import ContextCollator
from src.models import (
    Alternative,
    CommitData,
    DecisionType,
    ExtractedDecision,
    IssueData,
    PRData,
)


class DecisionExtractor:
    """Uses Google Gemini (gemini-3.5-flash-lite) to extract structured architectural decisions."""

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

    def __init__(
        self,
        api_key: str,
        model: str = "gemini-3.5-flash-lite",
        fallback_models: Optional[List[str]] = None,
        groq_api_key: Optional[str] = None,
        groq_fallback_models: Optional[List[str]] = None,
        enable_cross_provider_fallback: bool = True,
        cooldown_seconds: int = 60,
    ):
        self.api_key = api_key
        self.cooldown_seconds = cooldown_seconds
        self.all_models_exhausted: bool = False
        self._is_placeholder = not api_key or api_key in ("GOOGLE_API_KEY", "YOUR_GOOGLE_API_KEY")

        # Build prioritized Gemini model list
        def _clean_model(m: str) -> str:
            cleaned = m.strip().lower().replace(" ", "-")
            if cleaned.startswith("models/"):
                cleaned = cleaned[7:]
            return cleaned

        primary = _clean_model(model)
        self.models: List[str] = [primary]
        if fallback_models:
            for m in fallback_models:
                cm = _clean_model(m)
                if cm and cm not in self.models:
                    self.models.append(cm)

        self.active_model_idx: int = 0
        self.client: Optional[Any] = None

        if not self._is_placeholder:
            if HAS_NEW_GENAI:
                try:
                    self.client = genai.Client(api_key=api_key)
                    logger.info(
                        f"Initialized google.genai Client with primary model: {primary} "
                        f"and {len(self.models)-1} fallback models: {self.models[1:]}"
                    )
                except Exception as e:
                    logger.warning(f"Could not initialize google.genai Client: {e}")
            elif HAS_LEGACY_GENAI:
                try:
                    legacy_genai.configure(api_key=api_key)
                    logger.info(f"Initialized legacy genai with model cascade: {self.models}")
                except Exception as e:
                    logger.warning(f"Could not initialize legacy genai: {e}")

        # Cross-provider Groq fallback setup
        self.enable_cross_provider_fallback = enable_cross_provider_fallback
        self.groq_models: List[str] = ["openai/gpt-oss-120b", "openai/gpt-oss-20b"]
        if groq_fallback_models:
            cleaned_g = [m.strip() for m in groq_fallback_models if m.strip()]
            if cleaned_g:
                self.groq_models = cleaned_g
        self.active_groq_idx: int = 0
        self.groq_client: Optional[Any] = None

        if (
            enable_cross_provider_fallback
            and groq_api_key
            and groq_api_key not in ("GROQ_API_KEY", "YOUR_GROQ_API_KEY")
        ):
            try:
                from groq import Groq
                self.groq_client = Groq(api_key=groq_api_key)
                logger.info(f"Initialized Groq cross-provider fallback client with models: {self.groq_models}")
            except Exception as e:
                logger.debug(f"Groq fallback client initialization note: {e}")

    @property
    def current_model(self) -> str:
        """Returns the currently active Gemini model name."""
        if self.active_model_idx < len(self.models):
            return self.models[self.active_model_idx]
        return self.models[-1]

    def extract_from_text(self, text: str, source_type: str, source_id: str) -> Optional[ExtractedDecision]:
        """Sends discussion text to LLM with sticky multi-model cascading on rate limit / daily quota exhaustion."""
        if not text or len(text.strip()) < 30:
            return None

        if not self._is_placeholder:
            # 1. Cascade through configured Gemini models (Primary + Fallbacks)
            while self.active_model_idx < len(self.models):
                curr_model = self.models[self.active_model_idx]
                retries = 2
                backoff = 2.0
                model_failed = False

                for attempt in range(retries):
                    try:
                        content_str = ""
                        # Modern google.genai SDK
                        if HAS_NEW_GENAI and self.client:
                            config = types.GenerateContentConfig(
                                temperature=0.1,
                                response_mime_type="application/json",
                                system_instruction=self.SYSTEM_PROMPT,
                            )
                            prompt = f"Context for {source_type} #{source_id}:\n\n{text}"
                            resp = self.client.models.generate_content(
                                model=curr_model,
                                contents=prompt,
                                config=config,
                            )
                            content_str = resp.text or ""

                        # Legacy google.generativeai SDK
                        elif HAS_LEGACY_GENAI:
                            legacy_model = legacy_genai.GenerativeModel(model_name=curr_model)
                            prompt = f"{self.SYSTEM_PROMPT}\n\nContext for {source_type} #{source_id}:\n\n{text}"
                            resp = legacy_model.generate_content(prompt)
                            content_str = resp.text or ""

                        if content_str:
                            clean_content = content_str.strip()
                            if clean_content.startswith("```json"):
                                clean_content = clean_content[7:-3].strip()
                            elif clean_content.startswith("```"):
                                clean_content = clean_content[3:-3].strip()

                            data = json.loads(clean_content)
                            if data.get("confidence_score", 0.0) < 0.3:
                                return None

                            return ExtractedDecision.model_validate(data)

                    except Exception as e:
                        err_msg = str(e).lower()
                        is_quota = (
                            "quota" in err_msg
                            or "daily" in err_msg
                            or "per day" in err_msg
                            or "queries per day" in err_msg
                        )
                        is_rate = "429" in err_msg or "resource_exhausted" in err_msg

                        if is_quota or (is_rate and attempt == retries - 1):
                            logger.warning(
                                f"Rate limit / daily quota reached for Gemini model '{curr_model}' on {source_type} #{source_id}. "
                                f"Switching to next model in cascade."
                            )
                            self.active_model_idx += 1
                            model_failed = True
                            if self.active_model_idx < len(self.models):
                                logger.info(
                                    f"Active Gemini extraction model switched to: '{self.models[self.active_model_idx]}'"
                                )
                            break
                        elif is_rate:
                            logger.info(f"Gemini {curr_model} per-minute rate limit, sleeping {backoff:.1f}s...")
                            time.sleep(backoff)
                            backoff *= 2
                        else:
                            logger.debug(f"Gemini {curr_model} attempt {attempt + 1} for {source_type} #{source_id}: {e}")
                            time.sleep(1.0)

                if not model_failed:
                    # Non-rate-limit exception or valid response processed
                    break

            # 2. Cross-provider fallback to Groq if all Gemini models are exhausted
            if self.enable_cross_provider_fallback and self.groq_client:
                while self.active_groq_idx < len(self.groq_models):
                    g_model = self.groq_models[self.active_groq_idx]
                    try:
                        logger.info(f"All Gemini models exhausted. Falling back to Groq model '{g_model}' for extraction.")
                        prompt = f"Context for {source_type} #{source_id}:\n\n{text}"
                        resp = self.groq_client.chat.completions.create(
                            model=g_model,
                            messages=[
                                {"role": "system", "content": self.SYSTEM_PROMPT},
                                {"role": "user", "content": prompt},
                            ],
                            response_format={"type": "json_object"},
                            temperature=0.1,
                        )
                        content_str = resp.choices[0].message.content or ""
                        clean_content = content_str.strip()
                        if clean_content.startswith("```json"):
                            clean_content = clean_content[7:-3].strip()
                        elif clean_content.startswith("```"):
                            clean_content = clean_content[3:-3].strip()

                        data = json.loads(clean_content)
                        if data.get("confidence_score", 0.0) < 0.3:
                            return None

                        return ExtractedDecision.model_validate(data)
                    except Exception as g_err:
                        g_msg = str(g_err).lower()
                        is_groq_rate = "429" in g_msg or "rate limit" in g_msg or "quota" in g_msg
                        if is_groq_rate:
                            logger.warning(f"Groq model '{g_model}' rate limited. Advancing to next Groq fallback.")
                            self.active_groq_idx += 1
                        else:
                            logger.warning(f"Groq extraction fallback with '{g_model}' failed: {g_err}")
                            break

            # 3. If ALL models (all Gemini + all Groq) are exhausted, enforce rate limit cooldown!
            all_gemini_exhausted = self.active_model_idx >= len(self.models)
            all_groq_exhausted = not self.groq_client or self.active_groq_idx >= len(self.groq_models)

            if all_gemini_exhausted and all_groq_exhausted:
                self.all_models_exhausted = True
                logger.warning(
                    f"🚨 ALL models across Google Gemini ({len(self.models)} models) and Groq ({len(self.groq_models)} models) "
                    f"are exhausted due to rate limits! Enforcing rate limit cool-down of {self.cooldown_seconds}s."
                )
                if self.cooldown_seconds > 0:
                    time.sleep(self.cooldown_seconds)

        # 4. Deterministic rule-based heuristic extractor fallback
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
