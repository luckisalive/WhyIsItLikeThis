"""
Embedding generator using Gemini Embedding 2 at 768 dimensions.
Supports the modern google.genai SDK with fallback to legacy google.generativeai and offline mock.
"""

import hashlib
import logging
import math
import time
from typing import List, Optional

logger = logging.getLogger(__name__)

# Try modern google-genai SDK first
try:
    from google import genai
    from google.genai import types
    HAS_NEW_GENAI = True
except ImportError:
    HAS_NEW_GENAI = False

HAS_LEGACY_GENAI = False
if not HAS_NEW_GENAI:
    try:
        import warnings
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", FutureWarning)
            import google.generativeai as legacy_genai
        HAS_LEGACY_GENAI = True
    except ImportError:
        HAS_LEGACY_GENAI = False

from src.models import ExtractedDecision


class EmbeddingGenerator:
    """Generates 768-dimensional embeddings using Gemini Embedding 2."""

    def __init__(
        self,
        api_key: str,
        model: str = "gemini-embedding-2",
        dimensions: int = 768,
    ):
        self.api_key = api_key
        self.model = model
        self.dimensions = dimensions
        self._is_placeholder = not api_key or api_key in ("GOOGLE_API_KEY", "YOUR_GOOGLE_API_KEY")

        self.client: Optional[Any] = None
        if not self._is_placeholder:
            if HAS_NEW_GENAI:
                try:
                    self.client = genai.Client(api_key=api_key)
                    logger.info("Initialized modern google.genai Client for embeddings.")
                except Exception as e:
                    logger.warning(f"Failed to initialize google.genai Client: {e}")
            elif HAS_LEGACY_GENAI:
                try:
                    legacy_genai.configure(api_key=api_key)
                    logger.info("Initialized legacy google.generativeai for embeddings.")
                except Exception as e:
                    logger.warning(f"Failed to configure legacy genai: {e}")

    def embed_text(self, text: str) -> List[float]:
        """Embed a single text string into a 768-dim float vector."""
        if not text or not text.strip():
            text = "empty text"

        if self._is_placeholder:
            return self._generate_fallback_embedding(text, dimensions=self.dimensions)

        # 1. Try modern google.genai SDK
        if HAS_NEW_GENAI and self.client:
            try:
                config = types.EmbedContentConfig(
                    output_dimensionality=self.dimensions,
                    task_type="RETRIEVAL_DOCUMENT",
                )
                resp = self.client.models.embed_content(
                    model=self.model,
                    contents=text,
                    config=config,
                )
                if resp.embeddings and len(resp.embeddings) > 0:
                    values = resp.embeddings[0].values
                    if len(values) == self.dimensions:
                        return values
                    elif len(values) > self.dimensions:
                        return values[: self.dimensions]
                    return values
            except Exception as e:
                err_msg = str(e)
                if "429" in err_msg or "RESOURCE_EXHAUSTED" in err_msg:
                    logger.warning("Gemini rate limit hit. Sleeping 5 seconds before retrying...")
                    time.sleep(5)
                    return self.embed_text(text)
                logger.warning(f"google.genai embedding failed: {e}. Trying fallback.")

        # 2. Try legacy google.generativeai SDK
        if HAS_LEGACY_GENAI:
            try:
                legacy_model = self.model
                if not legacy_model.startswith("models/"):
                    legacy_model = f"models/{legacy_model}"
                result = legacy_genai.embed_content(
                    model=legacy_model,
                    content=text,
                    task_type="retrieval_document",
                    output_dimensionality=self.dimensions,
                )
                values = result.get("embedding", [])
                if values:
                    return values[: self.dimensions]
            except Exception as e:
                logger.warning(f"Legacy genai embedding failed: {e}")

        # 3. Deterministic fallback for offline testing or unrecoverable error
        return self._generate_fallback_embedding(text, dimensions=self.dimensions)

    def generate(self, text: str) -> List[float]:
        """Alias for embed_text."""
        return self.embed_text(text)

    def embed_decision(self, decision: ExtractedDecision) -> List[float]:
        """Combines decision title, rationale, context, and decision made into a single 768d embedding."""
        combined_text = (
            f"Title: {decision.title}\n"
            f"Rationale: {decision.rationale}\n"
            f"Context: {decision.problem_context}\n"
            f"Decision: {decision.decision_made}"
        )
        return self.embed_text(combined_text)

    def embed_batch(self, texts: List[str]) -> List[List[float]]:
        """Batch embed multiple texts while adhering to API rate quotas."""
        embeddings = []
        for i, text in enumerate(texts):
            embeddings.append(self.embed_text(text))
            if not self._is_placeholder and (i + 1) % 15 == 0:
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
