from types import SimpleNamespace
from unittest.mock import MagicMock

import src.graph.embedding_generator as embedding_generator
from src.graph.embedding_generator import EmbeddingGenerator


def _use_modern_embedding(monkeypatch, values):
    response = SimpleNamespace(
        embeddings=[SimpleNamespace(values=values)]
    )
    client = SimpleNamespace(
        models=SimpleNamespace(embed_content=MagicMock(return_value=response))
    )
    monkeypatch.setattr(embedding_generator, "HAS_NEW_GENAI", True)
    monkeypatch.setattr(embedding_generator, "HAS_LEGACY_GENAI", False)
    monkeypatch.setattr(
        embedding_generator,
        "types",
        SimpleNamespace(EmbedContentConfig=lambda **kwargs: kwargs),
        raising=False,
    )
    embedder = EmbeddingGenerator(api_key="test-key", dimensions=3)
    embedder.client = client
    return embedder


def test_short_modern_embedding_uses_deterministic_fallback(monkeypatch, caplog):
    embedder = _use_modern_embedding(monkeypatch, [0.1, 0.2])

    result = embedder.embed_text("short vector")

    assert len(result) == embedder.dimensions
    assert "returned an embedding with 2 dimensions" in caplog.text


def test_modern_embedding_preserves_exact_and_oversized_handling(monkeypatch):
    exact = [0.1, 0.2, 0.3]
    embedder = _use_modern_embedding(monkeypatch, exact)
    assert embedder.embed_text("exact vector") == exact

    oversized = [0.1, 0.2, 0.3, 0.4]
    embedder = _use_modern_embedding(monkeypatch, oversized)
    assert embedder.embed_text("oversized vector") == oversized[:3]


def test_short_legacy_embedding_uses_deterministic_fallback(monkeypatch, caplog):
    legacy = SimpleNamespace(
        configure=MagicMock(),
        embed_content=MagicMock(return_value={"embedding": [0.1, 0.2]}),
    )
    monkeypatch.setattr(embedding_generator, "HAS_NEW_GENAI", False)
    monkeypatch.setattr(embedding_generator, "HAS_LEGACY_GENAI", True)
    monkeypatch.setattr(
        embedding_generator, "legacy_genai", legacy, raising=False
    )
    embedder = EmbeddingGenerator(api_key="test-key", dimensions=3)

    result = embedder.embed_text("short legacy vector")

    assert len(result) == embedder.dimensions
    assert "returned an embedding with 2 dimensions" in caplog.text

    exact = [0.1, 0.2, 0.3]
    legacy.embed_content.return_value = {"embedding": exact}
    assert embedder.embed_text("exact legacy vector") == exact

    oversized = [0.1, 0.2, 0.3, 0.4]
    legacy.embed_content.return_value = {"embedding": oversized}
    assert embedder.embed_text("oversized legacy vector") == oversized[:3]
