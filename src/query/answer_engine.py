"""
Answer generation engine powered by Groq (openai/gpt-oss-120b) and GraphRAG.
Synthesizes graph traversal results into structured answers with confidence signals,
evidence chains, supersession warnings, and recommended points of contact.
"""

import json
import logging
from typing import Any, List, Optional

from groq import Groq

from src.models import Answer, Confidence, EvidenceItem, SupersessionInfo
from src.query.retriever import WhyRetriever

logger = logging.getLogger(__name__)


class AnswerEngine:
    """Uses Groq openai/gpt-oss-120b to reason over graph context and generate verifiable answers."""

    SYSTEM_PROMPT = """
You are a senior software architect and code historian answering 'why is the code like this' questions.
Your goal is to explain why specific architectural decisions, configurations, and patterns were chosen, using ONLY the provided knowledge graph evidence.

Strict Rules:
1. Every claim MUST be grounded in the provided context (cite specific PRs, commits, issues, ADRs, or people).
2. If a decision was superseded or reversed later, explicitly state so and explain the evolutionary arc.
3. If the evidence is insufficient or absent in the context, do NOT invent an explanation. State: "I don't have enough evidence in the graph to answer this with certainty." Suggest who to ask based on connected contributors or where to investigate.
4. Assign an objective confidence level:
   - HIGH: Direct primary evidence found (PR discussion, commit rationale, ADR).
   - MEDIUM: Inferred from related decisions, touched components, or partial evidence.
   - LOW: Insufficient or tangential evidence; educated conjecture.
5. You MUST return ONLY a valid JSON object matching this schema:
{
  "answer_text": "Detailed explanation of why the code is like this...",
  "confidence": "HIGH" | "MEDIUM" | "LOW",
  "confidence_explanation": "Brief sentence explaining why this confidence was assigned...",
  "evidence": [
    {
      "source_type": "pull_request" | "commit" | "issue" | "decision",
      "source_id": "PR # / SHA / Issue #",
      "title": "Title or headline of the evidence item",
      "summary": "Key excerpt or rationale from this item",
      "person": "Relevant developer login if known",
      "url": "URL if available"
    }
  ],
  "supersession": {
    "is_superseded": true | false,
    "superseded_by": "Title or ID of the newer decision",
    "reason": "Why the newer decision superseded this one"
  },
  "ask_person": "GitHub username of the original decision maker or reviewer",
  "related_decisions": ["DEC-PR-...", "DEC-COMMIT-..."]
}
"""

    def __init__(
        self,
        api_key: str,
        model: str = "openai/gpt-oss-120b",
        retriever: Optional[WhyRetriever] = None,
        fallback_models: Optional[List[str]] = None,
        google_api_key: Optional[str] = None,
        gemini_reasoning_models: Optional[List[str]] = None,
        enable_cross_provider_fallback: bool = True,
        cooldown_seconds: int = 60,
    ):

        self.api_key = api_key
        self.retriever = retriever
        self._is_placeholder = not api_key or api_key in ("GROQ_API_KEY", "YOUR_GROQ_API_KEY")

        # Groq model cascade
        self.models: List[str] = [model]
        if fallback_models:
            for m in fallback_models:
                m_clean = m.strip()
                if m_clean and m_clean not in self.models:
                    self.models.append(m_clean)
        self.active_model_idx: int = 0

        self.client: Optional[Groq] = None
        if not self._is_placeholder:
            try:
                self.client = Groq(api_key=api_key)
                logger.info(f"Initialized Groq client with model cascade: {self.models}")
            except Exception as e:
                logger.warning(f"Could not initialize Groq client: {e}")

        # Google Gemini cross-provider fallback
        self.enable_cross_provider_fallback = enable_cross_provider_fallback
        self.cooldown_seconds = cooldown_seconds
        self.all_models_exhausted: bool = False
        self.gemini_client: Optional[Any] = None
        self.gemini_models: List[str] = [
            "gemini-3.5-flash-lite",
            "gemini-3.7-flash",
            "gemini-3.6-flash",
            "gemini-3-flash",
            "gemini-2.5-flash",
            "gemini-3.1-flash-lite",
            "gemini-2.5-flash-lite",
        ]
        if gemini_reasoning_models:
            cleaned_gm = [
                m.strip().lower().replace(" ", "-")
                for m in gemini_reasoning_models
                if m.strip()
            ]
            if cleaned_gm:
                # Ensure gemini-3.5-flash-lite is first if present
                ordered_gm = []
                if "gemini-3.5-flash-lite" in cleaned_gm:
                    ordered_gm.append("gemini-3.5-flash-lite")
                for m in cleaned_gm:
                    if m not in ordered_gm:
                        ordered_gm.append(m)
                self.gemini_models = ordered_gm

        self.active_gemini_idx: int = 0

        if (
            enable_cross_provider_fallback
            and google_api_key
            and google_api_key not in ("GOOGLE_API_KEY", "YOUR_GOOGLE_API_KEY")
        ):
            try:
                from google import genai
                self.gemini_client = genai.Client(api_key=google_api_key)
                logger.info(f"Initialized Gemini reasoning fallback client with models: {self.gemini_models}")
            except Exception as e:
                logger.debug(f"Gemini fallback client initialization note: {e}")

    @property
    def current_model(self) -> str:
        """Returns the currently active Groq model."""
        if self.active_model_idx < len(self.models):
            return self.models[self.active_model_idx]
        return self.models[-1]

    def answer(self, question: str) -> Answer:
        """Generates an evidence-grounded answer to a 'why' question with automatic failover."""
        if not self.retriever:
            logger.error("Retriever is not configured.")
            return self._build_insufficient_evidence_answer(question)

        results = self.retriever.retrieve(question)
        if not results:
            logger.info("No relevant context found in graph.")
            return self._build_insufficient_evidence_answer(question)

        context = self.retriever.format_context(results)

        if self._is_placeholder or not self.client:
            return self._build_heuristic_answer(question, results, context)

        prompt = f"QUESTION: {question}\n\nKNOWLEDGE GRAPH CONTEXT:\n{context}\n\nProvide the required JSON response:"

        # 1. Try Groq model cascade
        while self.active_model_idx < len(self.models):
            curr_model = self.models[self.active_model_idx]
            try:
                response = self.client.chat.completions.create(
                    model=curr_model,
                    messages=[
                        {"role": "system", "content": self.SYSTEM_PROMPT},
                        {"role": "user", "content": prompt},
                    ],
                    response_format={"type": "json_object"},
                    max_tokens=4096,
                    temperature=0.1,
                )
                response_text = response.choices[0].message.content or ""
                return self._parse_response(response_text, results, question)

            except Exception as e:
                err_msg = str(e).lower()
                is_rate = "429" in err_msg or "rate limit" in err_msg or "rate_limit_exceeded" in err_msg
                is_quota = "quota" in err_msg or "tokens per day" in err_msg

                if is_rate or is_quota:
                    logger.warning(
                        f"Groq rate limit reached for model '{curr_model}'. "
                        f"Switching to next model in cascade."
                    )
                    self.active_model_idx += 1
                    if self.active_model_idx < len(self.models):
                        logger.info(f"Active Groq model is now: '{self.models[self.active_model_idx]}'")
                        continue
                    else:
                        logger.warning("All Groq models exhausted.")
                        break
                else:
                    logger.warning(f"Groq API call ({curr_model}) failed: {e}")
                    break

        # 2. Try Google Gemini cross-provider reasoning fallback
        if self.enable_cross_provider_fallback and self.gemini_client:
            from google.genai import types
            while self.active_gemini_idx < len(self.gemini_models):
                g_model = self.gemini_models[self.active_gemini_idx]
                try:
                    logger.info(f"Attempting Gemini reasoning fallback with model: '{g_model}'")
                    config = types.GenerateContentConfig(
                        temperature=0.1,
                        response_mime_type="application/json",
                        system_instruction=self.SYSTEM_PROMPT,
                    )
                    resp = self.gemini_client.models.generate_content(
                        model=g_model,
                        contents=prompt,
                        config=config,
                    )
                    response_text = resp.text or ""
                    if response_text:
                        return self._parse_response(response_text, results, question)
                except Exception as gem_e:
                    gem_msg = str(gem_e).lower()
                    if "429" in gem_msg or "resource_exhausted" in gem_msg or "quota" in gem_msg:
                        logger.warning(f"Gemini fallback model '{g_model}' rate limited. Advancing to next Gemini fallback.")
                        self.active_gemini_idx += 1
                    else:
                        logger.warning(f"Gemini reasoning fallback with '{g_model}' failed: {gem_e}")
                        break

        # 3. If ALL models (Groq + Gemini) are exhausted, trigger rate limit cooldown!
        all_groq_exhausted = self.active_model_idx >= len(self.models)
        all_gemini_exhausted = not self.gemini_client or self.active_gemini_idx >= len(self.gemini_models)

        if all_groq_exhausted and all_gemini_exhausted:
            self.all_models_exhausted = True
            logger.warning(
                f"🚨 ALL models across Groq ({len(self.models)} models) and Google Gemini ({len(self.gemini_models)} models) "
                f"are exhausted due to rate limits! Enforcing rate limit cool-down of {self.cooldown_seconds}s."
            )
            if self.cooldown_seconds > 0:
                import time
                time.sleep(self.cooldown_seconds)

        # 4. Fallback to graph traversal heuristic synthesis
        logger.info("Falling back to direct graph context synthesis.")
        ans = self._build_heuristic_answer(question, results, context)
        if self.all_models_exhausted:
            ans.confidence_explanation = (
                "Synthesized directly from verified knowledge graph nodes. "
                "(Notice: All cloud LLMs temporarily throttled by API rate limits)."
            )
        return ans



    def _build_insufficient_evidence_answer(self, question: str) -> Answer:
        """Default response when no evidence exists in the graph."""
        return Answer(
            question=question,
            answer_text=(
                "I don't have enough evidence in the knowledge graph to answer why this is the way it is. "
                "No relevant decisions, PR discussions, or commit rationales were found for this topic."
            ),
            confidence=Confidence.LOW,
            confidence_explanation="Refused to invent an answer because the knowledge graph contains no matching decisions.",
            evidence=[],
            supersession=None,
            ask_person=None,
            related_decisions=[],
        )

    def _build_heuristic_answer(self, question: str, results: list, context: str) -> Answer:
        """Synthesizes an answer directly from graph traversal records when LLM API call is pending."""
        top_res = results[0]
        decision = top_res.get("decision", {})
        if not isinstance(decision, dict):
            decision = dict(decision) if hasattr(decision, "items") else {}

        title = decision.get("title", "Found Decision")
        rationale = decision.get("rationale", "No detailed rationale recorded.")
        decision_made = decision.get("decision_made", "")
        status = decision.get("status", "ACTIVE")
        is_superseded = status == "SUPERSEDED" or bool(top_res.get("superseded_by"))

        people = [
            p.get("name") or p.get("login")
            for p in top_res.get("people", [])
            if isinstance(p, dict)
        ]
        author = people[0] if people else None

        evidence_items = []
        for pr in top_res.get("pull_requests", []):
            if isinstance(pr, dict):
                evidence_items.append(
                    EvidenceItem(
                        source_type="pull_request",
                        source_id=str(pr.get("number", "")),
                        title=pr.get("title", ""),
                        summary="PR Discussion regarding this architectural choice",
                        url=pr.get("url"),
                        person=author,
                    )
                )

        sup_info = None
        if is_superseded:
            newer = top_res.get("superseded_by", [])
            newer_title = newer[0].get("title", "newer decision") if newer and isinstance(newer[0], dict) else "a later update"
            sup_info = SupersessionInfo(
                superseded_decision=title,
                superseded_by=newer_title,
                reason="Overridden by subsequent repository changes.",
            )

        answer_text = (
            f"Based on repository decisions in the knowledge graph, this was established in: **{title}**.\n\n"
            f"**Decision Made**: {decision_made or 'Implemented per design'}\n\n"
            f"**Rationale**: {rationale}\n"
        )
        if is_superseded:
            answer_text += f"\n⚠️ **Note**: This decision was subsequently superseded."

        return Answer(
            question=question,
            answer_text=answer_text,
            confidence=Confidence.MEDIUM if results else Confidence.LOW,
            confidence_explanation="Synthesized directly from verified knowledge graph nodes and relationships.",
            evidence=evidence_items,
            supersession=sup_info,
            ask_person=author,
            related_decisions=[decision.get("id")] if decision.get("id") else [],
        )

    def _parse_response(self, response_text: str, retrieval_results: list, question: str) -> Answer:
        """Parses structured JSON response into an Answer model."""
        try:
            text = response_text.strip()
            if text.startswith("```json"):
                text = text[7:]
            elif text.startswith("```"):
                text = text[3:]
            if text.endswith("```"):
                text = text[:-3]
            text = text.strip()

            data = json.loads(text)

            evidence_items = []
            for item in data.get("evidence", []):
                evidence_items.append(
                    EvidenceItem(
                        source_type=item.get("source_type", "decision"),
                        source_id=str(item.get("source_id", "")),
                        title=item.get("title", ""),
                        summary=item.get("summary", ""),
                        person=item.get("person"),
                        url=item.get("url"),
                    )
                )

            sup_data = data.get("supersession")
            supersession = None
            if sup_data and isinstance(sup_data, dict) and sup_data.get("is_superseded"):
                supersession = SupersessionInfo(
                    superseded_decision=sup_data.get("superseded_decision", "Previous decision"),
                    superseded_by=sup_data.get("superseded_by", "Newer decision"),
                    reason=sup_data.get("reason", "Overridden by subsequent policy or refactor"),
                )

            conf_str = str(data.get("confidence", "MEDIUM")).upper()
            if "HIGH" in conf_str:
                confidence = Confidence.HIGH
            elif "LOW" in conf_str:
                confidence = Confidence.LOW
            else:
                confidence = Confidence.MEDIUM

            return Answer(
                question=question,
                answer_text=data.get("answer_text", "No detailed answer generated."),
                confidence=confidence,
                confidence_explanation=data.get("confidence_explanation", "Evaluated from graph provenance."),
                evidence=evidence_items,
                supersession=supersession,
                ask_person=data.get("ask_person"),
                related_decisions=data.get("related_decisions", []),
            )

        except Exception as e:
            logger.warning(f"Failed to parse LLM JSON response: {e}. Falling back to graph summary.")
            return self._build_heuristic_answer(question, retrieval_results, "")
