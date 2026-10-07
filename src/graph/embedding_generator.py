"""
Embedding generator using Google Gemini embedding models.
Converts decision texts and user queries into 768-dimensional vector embeddings for Neo4j vector search.
"""

import hashlib
import logging
import math
import time
from typing import List

import google.generativeai as genai

from src.models import ExtractedDecision

logger = logging.getLogger(__name__)


class EmbeddingGenerator:
    """Generates embeddings using Google Gemini with rate-limit and placeholder handling."""

    def __init__(self, api_key: str, model: str = "models/text-embedding-004"):
        self.api_key = api_key
        self.model = model
        self._is_placeholder = not api_key or api_key in ("GOOGLE_API_KEY", "YOUR_GOOGLE_API_KEY")
        if not self._is_placeholder:
            try:
                genai.configure(api_key=api_key)
            except Exception as e:
                logger.warning(f"Could not configure genai with provided key: {e}")

    def embed_text(self, text: str) -> List[float]:
        """Embed a single text string into a 768-dim float vector."""
        if not text or not text.strip():
            text = "empty text"

        if self._is_placeholder:
            # Deterministic pseudo-embedding for testing/mocking when API key is a placeholder
            return self._generate_fallback_embedding(text)

        try:
            result = genai.embed_content(
                model=self.model,
                content=text,
                task_type="retrieval_document",
            )
            return result["embedding"]
        except Exception as e:
            err_msg = str(e)
            if "429" in err_msg or "RESOURCE_EXHAUSTED" in err_msg:
                logger.warning("Gemini rate limit hit. Sleeping 5 seconds before retrying...")
                time.sleep(5)
                return self.embed_text(text)
            elif "API_KEY_INVALID" in err_msg or "INVALID_ARGUMENT" in err_msg or "key" in err_msg.lower():
                logger.warning(f"Gemini API key issue: {e}. Using deterministic fallback embedding.")
                return self._generate_fallback_embedding(text)
            logger.error(f"Error embedding text: {e}")
            return self._generate_fallback_embedding(text)

    def generate(self, text: str) -> List[float]:
        """Alias for embed_text to maintain interface compatibility."""
        return self.embed_text(text)

    def embed_decision(self, decision: ExtractedDecision) -> List[float]:
        """Combines decision title, rationale, context, and outcome into a unified embedding."""
        combined_text = (
            f"Title: {decision.title}\n"
            f"Rationale: {decision.rationale}\n"
            f"Context: {decision.problem_context}\n"
            f"Decision: {decision.decision_made}"
        )
        return self.embed_text(combined_text)

    def embed_batch(self, texts: List[str]) -> List[List[float]]:
        """Batch embed multiple texts while adhering to Gemini RPM quotas."""
        embeddings = []
        for i, text in enumerate(texts):
            embeddings.append(self.embed_text(text))
            if not self._is_placeholder and (i + 1) % 10 == 0:
                time.sleep(2)
        return embeddings

    def _generate_fallback_embedding(self, text: str, dimensions: int = 768) -> List[float]:
        """Generates a deterministic unit-normalized pseudo-embedding for testing or offline mode."""
        seed = int(hashlib.sha256(text.encode("utf-8")).hexdigest()[:8], 16)
        vec = []
        norm_sq = 0.0
        for i in range(dimensions):
            val = math.sin(seed + i * 0.1)
            vec.append(val)
            norm_sq += val * val
        norm = math.sqrt(norm_sq) or 1.0
        return [v / norm for v in vec]
