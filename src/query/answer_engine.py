"""
Answer generation engine powered by Google Gemini and GraphRAG.
Synthesizes graph traversal results into structured answers with confidence signals,
evidence chains, supersession warnings, and recommended points of contact.
"""

import json
import logging
from typing import List, Optional

import google.generativeai as genai

from src.models import Answer, Confidence, EvidenceItem, SupersessionInfo
from src.query.retriever import WhyRetriever

logger = logging.getLogger(__name__)


class AnswerEngine:
    """Uses Gemini to reason over graph context and generate verifiable, evidence-backed answers."""

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
        model: str = "gemini-2.0-flash",
        retriever: Optional[WhyRetriever] = None,
    ):
        self.api_key = api_key
        self.model_name = model
        self.retriever = retriever
        self._is_placeholder = not api_key or api_key in ("GOOGLE_API_KEY", "YOUR_GOOGLE_API_KEY")

        if not self._is_placeholder:
            try:
                genai.configure(api_key=api_key)
                self.model = genai.GenerativeModel(model_name=model)
            except Exception as e:
                logger.warning(f"Could not initialize Gemini model: {e}")
                self.model = None
        else:
            self.model = None

    def answer(self, question: str) -> Answer:
        """Generates an evidence-grounded answer to a 'why' question."""
        if not self.retriever:
            logger.error("Retriever is not configured.")
            return self._build_insufficient_evidence_answer(question)

        results = self.retriever.retrieve(question)
        if not results:
            logger.info("No relevant context found in graph.")
            return self._build_insufficient_evidence_answer(question)

        context = self.retriever.format_context(results)

        if not self.model or self._is_placeholder:
            return self._build_heuristic_answer(question, results, context)

        prompt = f"QUESTION: {question}\n\nKNOWLEDGE GRAPH CONTEXT:\n{context}\n\nProvide the required JSON response:"

        try:
            response = self.model.generate_content([self.SYSTEM_PROMPT, prompt])
            response_text = response.text or ""
            return self._parse_response(response_text, results, question)
        except Exception as e:
            logger.warning(f"Gemini API call failed: {e}. Falling back to graph evidence synthesis.")
            return self._build_heuristic_answer(question, results, context)

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
        """Synthesizes an answer directly from graph traversal records when LLM API key is pending."""
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
        """Parses structured JSON response from Gemini into an Answer model."""
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
