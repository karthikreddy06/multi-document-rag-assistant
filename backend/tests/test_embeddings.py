"""
Tests for EmbeddingService.
"""

import pytest
from app.embeddings.service import EmbeddingService


def test_embedding_service_health():
    service = EmbeddingService()
    # If Ollama is running, check_health should be True
    health = service.check_health()
    assert health is True


def test_embed_text_valid():
    service = EmbeddingService()
    emb = service.embed_text("FastAPI backend microservices")
    assert isinstance(emb, list)
    assert len(emb) > 0
    assert all(isinstance(x, float) for x in emb)


def test_embed_empty_text():
    service = EmbeddingService()
    emb = service.embed_text("   ")
    assert emb == []


def test_embed_batch():
    service = EmbeddingService()
    texts = ["Python and Java", "Machine Learning models"]
    embs = service.embed_batch(texts)
    assert len(embs) == 2
    assert len(embs[0]) == len(embs[1])
