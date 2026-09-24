"""
Comprehensive Unit Tests for Production Provider Architecture:
- Groq Cloud LLM provider (configuration, non-streaming generation, SSE streaming)
- Local CPU ONNX embedding provider (initialization, dimensionality, batching)
- Provider selection & resolution (Groq, Local, Ollama, Gemini)
- Ollama & Gemini regression testing
- Main FastAPI entrypoint compatibility (uvicorn app.main:app)
"""

import json
from unittest.mock import MagicMock
import httpx
import pytest

from app.config import Settings
from app.embeddings.service import EmbeddingService
from app.generation.generator import LLMGenerator
from app.models import RetrievedChunk


# ==============================================================================
# 1. Groq Configuration Tests
# ==============================================================================

def test_groq_config_resolution(monkeypatch):
    """Verify Groq environment variables resolve to Groq API URL and production GPT-OSS model."""
    monkeypatch.setenv("LLM_PROVIDER", "groq")
    monkeypatch.setenv("GROQ_API_KEY", "gsk_test_secret_key_12345")
    monkeypatch.delenv("LLM_MODEL", raising=False)
    monkeypatch.delenv("LLM_API_URL", raising=False)

    cfg = Settings()
    assert cfg.llm_provider == "groq"
    assert cfg.effective_llm_api_key == "gsk_test_secret_key_12345"
    assert cfg.effective_llm_api_url == "https://api.groq.com/openai/v1"
    assert cfg.effective_llm_model == "openai/gpt-oss-20b"


def test_groq_config_custom_model(monkeypatch):
    """Verify custom model override is respected for Groq provider."""
    monkeypatch.setenv("LLM_PROVIDER", "groq")
    monkeypatch.setenv("GROQ_API_KEY", "gsk_test_secret_key_12345")
    monkeypatch.setenv("LLM_MODEL", "openai/gpt-oss-120b")

    cfg = Settings()
    assert cfg.llm_provider == "groq"
    assert cfg.effective_llm_model == "openai/gpt-oss-120b"


# ==============================================================================
# 2. Groq Non-Streaming Generation Tests
# ==============================================================================

def test_groq_non_streaming_generation(monkeypatch):
    """Verify Groq non-streaming request sends standard OpenAI Bearer auth without x-goog-api-key."""
    captured_request = {}

    def mock_post(url, headers=None, json=None):
        captured_request["url"] = url
        captured_request["headers"] = headers
        captured_request["json"] = json

        resp = MagicMock()
        resp.raise_for_status = MagicMock()
        resp.json.return_value = {
            "choices": [
                {
                    "message": {
                        "role": "assistant",
                        "content": "Groq inference is extremely fast and accurate."
                    }
                }
            ],
            "model": "openai/gpt-oss-20b"
        }
        return resp

    mock_client = MagicMock()
    mock_client.post.side_effect = mock_post
    mock_client.__enter__.return_value = mock_client
    mock_client.__exit__.return_value = None

    monkeypatch.setattr(httpx, "Client", lambda **kwargs: mock_client)

    generator = LLMGenerator(
        provider="groq",
        model="openai/gpt-oss-20b",
        api_url="https://api.groq.com/openai/v1",
        api_key="gsk_test_auth_key",
    )

    chunk = RetrievedChunk(
        text="Groq operates LPUs tailored for tensor streaming.",
        metadata={"filename": "groq_overview.txt", "chunk_id": "chk_1"},
        score=0.95
    )

    answer = generator.generate_answer(
        question="What hardware does Groq use?",
        chunks=[chunk]
    )

    assert "Groq inference is extremely fast" in answer
    assert captured_request["url"] == "https://api.groq.com/openai/v1/chat/completions"
    assert captured_request["headers"]["Authorization"] == "Bearer gsk_test_auth_key"
    assert "x-goog-api-key" not in captured_request["headers"]
    assert captured_request["json"]["model"] == "openai/gpt-oss-20b"
    assert len(captured_request["json"]["messages"]) == 1


# ==============================================================================
# 3. Groq Streaming Generation Tests
# ==============================================================================

def test_groq_streaming_generation(monkeypatch):
    """Verify Groq SSE streaming chunks are parsed and yielded as token strings."""
    captured_stream = {}

    class MockStreamResponse:
        def __init__(self, lines):
            self.lines = lines

        def raise_for_status(self):
            pass

        def iter_lines(self):
            return iter(self.lines)

        def __enter__(self):
            return self

        def __exit__(self, *args):
            pass

    def mock_stream(method, url, headers=None, json=None):
        captured_stream["method"] = method
        captured_stream["url"] = url
        captured_stream["headers"] = headers
        captured_stream["json"] = json

        lines = [
            'data: {"choices": [{"delta": {"content": "Groq "}}]}',
            'data: {"choices": [{"delta": {"content": "streaming "}}]}',
            'data: {"choices": [{"delta": {"content": "works!"}}]}',
            'data: [DONE]',
        ]
        return MockStreamResponse(lines)

    mock_client = MagicMock()
    mock_client.stream.side_effect = mock_stream
    mock_client.__enter__.return_value = mock_client
    mock_client.__exit__.return_value = None

    monkeypatch.setattr(httpx, "Client", lambda **kwargs: mock_client)

    generator = LLMGenerator(
        provider="groq",
        model="openai/gpt-oss-20b",
        api_url="https://api.groq.com/openai/v1",
        api_key="gsk_test_auth_key",
    )

    chunk = RetrievedChunk(
        text="Real-time streaming test document.",
        metadata={"filename": "streaming.txt", "chunk_id": "chk_2"},
        score=0.91
    )

    tokens = list(generator.generate_answer_stream(
        question="How does streaming work?",
        chunks=[chunk]
    ))

    full_output = "".join(tokens)
    assert full_output == "Groq streaming works!"
    assert captured_stream["url"] == "https://api.groq.com/openai/v1/chat/completions"
    assert captured_stream["json"]["stream"] is True
    assert captured_stream["headers"]["Authorization"] == "Bearer gsk_test_auth_key"
    assert "x-goog-api-key" not in captured_stream["headers"]


# ==============================================================================
# 4. Local CPU ONNX Embedding Provider Tests
# ==============================================================================

def test_local_embedding_init_and_config(monkeypatch):
    """Verify local embedding service defaults to all-MiniLM-L6-v2 and 384 dimensions."""
    monkeypatch.setenv("EMBEDDING_PROVIDER", "local")
    monkeypatch.delenv("EMBEDDING_MODEL", raising=False)
    monkeypatch.delenv("EMBEDDING_DIMENSIONS", raising=False)

    cfg = Settings()
    assert cfg.embedding_provider == "local"
    assert cfg.effective_embedding_model == "all-MiniLM-L6-v2"
    assert cfg.effective_embedding_dimensions == 384

    service = EmbeddingService(provider="local")
    assert service.provider == "local"
    assert service.model == "all-MiniLM-L6-v2"
    assert service.dimensions == 384
    assert service.check_health() is True


def test_local_embedding_single_and_batch_execution():
    """Verify real local ONNX CPU inference generates exactly 384-dimensional vectors."""
    service = EmbeddingService(
        provider="local",
        model="all-MiniLM-L6-v2",
        dimensions=384,
    )

    # Test single embedding
    text = "The quick brown fox jumps over the lazy dog."
    vec = service.embed_text(text)
    assert isinstance(vec, list)
    assert len(vec) == 384
    assert all(isinstance(x, float) for x in vec)

    # Test batch embedding
    texts = [
        "First test document about neural networks.",
        "Second test document about retrieval augmented generation."
    ]
    batch_vecs = service.embed_batch(texts)
    assert len(batch_vecs) == 2
    assert len(batch_vecs[0]) == 384
    assert len(batch_vecs[1]) == 384
    assert all(isinstance(x, float) for x in batch_vecs[0])
    assert all(isinstance(x, float) for x in batch_vecs[1])


def test_local_embedding_dimension_validation_mismatch():
    """Verify ValueError is raised if configured dimension does not match returned vector."""
    service = EmbeddingService(
        provider="local",
        model="all-MiniLM-L6-v2",
        dimensions=768,  # Deliberately wrong dimension (MiniLM produces 384)
    )

    with pytest.raises(ValueError, match="Expected embedding dimension 768"):
        service.embed_text("Dimension mismatch trigger test.")


# ==============================================================================
# 5. Provider Selection & Multi-Provider Regression Tests
# ==============================================================================

def test_provider_selection_matrix(monkeypatch):
    """Verify all 4 supported provider combinations resolve cleanly without cross-contamination."""
    # Scenario A: Production (Groq + Local CPU ONNX)
    monkeypatch.setenv("LLM_PROVIDER", "groq")
    monkeypatch.setenv("GROQ_API_KEY", "gsk_prod_key")
    monkeypatch.setenv("EMBEDDING_PROVIDER", "local")
    cfg_prod = Settings()
    assert cfg_prod.effective_llm_model == "openai/gpt-oss-20b"
    assert cfg_prod.effective_llm_api_url == "https://api.groq.com/openai/v1"
    assert cfg_prod.effective_llm_api_key == "gsk_prod_key"
    assert cfg_prod.effective_embedding_model == "all-MiniLM-L6-v2"
    assert cfg_prod.effective_embedding_dimensions == 384

    # Scenario B: Local Dev (Ollama + Ollama)
    monkeypatch.setenv("LLM_PROVIDER", "ollama")
    monkeypatch.setenv("EMBEDDING_PROVIDER", "ollama")
    monkeypatch.delenv("LLM_MODEL", raising=False)
    monkeypatch.delenv("EMBEDDING_MODEL", raising=False)
    monkeypatch.delenv("EMBEDDING_DIMENSIONS", raising=False)
    cfg_ollama = Settings()
    assert cfg_ollama.effective_llm_model == "llama3.2:1b"
    assert cfg_ollama.effective_embedding_model == "nomic-embed-text"
    assert cfg_ollama.effective_embedding_dimensions == 768

    # Scenario C: Gemini Cloud (Gemini Flash + Gemini Embeddings)
    monkeypatch.delenv("GROQ_API_KEY", raising=False)
    monkeypatch.delenv("LLM_API_KEY", raising=False)
    monkeypatch.setenv("LLM_PROVIDER", "cloud")
    monkeypatch.setenv("EMBEDDING_PROVIDER", "cloud")
    monkeypatch.setenv("GEMINI_API_KEY", "ai_gemini_key")
    cfg_gemini = Settings()
    assert cfg_gemini.effective_llm_model == "gemini-3.6-flash"
    assert cfg_gemini.effective_embedding_model == "gemini-embedding-2"
    assert cfg_gemini.effective_llm_api_key == "ai_gemini_key"
    assert cfg_gemini.effective_embedding_api_key == "ai_gemini_key"
    assert cfg_gemini.effective_llm_api_url == "https://generativelanguage.googleapis.com/v1beta/openai/"


# ==============================================================================
# 6. Render Entrypoint Test (uvicorn app.main:app)
# ==============================================================================

def test_uvicorn_entrypoint_export():
    """Verify app.main exports the FastAPI application instance required by Render."""
    from app.main import app as fastapi_app
    from fastapi import FastAPI

    assert isinstance(fastapi_app, FastAPI)
    assert fastapi_app.title == "Multi-Document RAG Assistant API"
