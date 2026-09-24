"""
Unit and integration tests for Google Gemini cloud provider integration and embedding dimensionality.
"""

import json
from unittest.mock import MagicMock
import httpx
import pytest

from app.config import Settings
from app.embeddings.service import EmbeddingService
from app.generation.generator import LLMGenerator
from app.models import RetrievedChunk


def test_gemini_config_defaults(monkeypatch):
    """Verify default configuration maintains local Ollama defaults for development."""
    monkeypatch.delenv("GEMINI_API_KEY", raising=False)
    cfg = Settings(_env_file=None)
    assert cfg.llm_provider == "ollama"
    assert cfg.llm_model == "llama3.2:1b"
    assert cfg.embedding_provider == "ollama"
    assert cfg.embedding_model == "nomic-embed-text"
    assert cfg.embedding_dimensions == 768
    assert cfg.gemini_api_key == ""


def test_gemini_config_cloud_resolution(monkeypatch):
    """Verify cloud provider environment variables resolve to Gemini endpoints and models."""
    monkeypatch.setenv("LLM_PROVIDER", "cloud")
    monkeypatch.setenv("EMBEDDING_PROVIDER", "cloud")
    monkeypatch.setenv("GEMINI_API_KEY", "test_gemini_secret_key")
    monkeypatch.setenv("EMBEDDING_MODEL", "gemini-embedding-2")
    monkeypatch.setenv("EMBEDDING_DIMENSIONS", "768")

    cfg = Settings()
    assert cfg.llm_provider == "cloud"
    assert cfg.embedding_provider == "cloud"
    assert cfg.effective_llm_api_key == "test_gemini_secret_key"
    assert cfg.effective_embedding_api_key == "test_gemini_secret_key"
    assert cfg.effective_llm_api_url == "https://generativelanguage.googleapis.com/v1beta/openai/"
    assert cfg.effective_embedding_api_url == "https://generativelanguage.googleapis.com/v1beta/openai/"
    assert cfg.effective_llm_model == "gemini-3.6-flash"
    assert cfg.effective_embedding_model == "gemini-embedding-2"
    assert cfg.embedding_dimensions == 768


def test_gemini_embedding_service_init():
    """Verify EmbeddingService properly defaults to Gemini configurations when provider is cloud."""
    service = EmbeddingService(
        provider="cloud",
        api_key="test_api_key",
    )
    assert service.provider == "cloud"
    assert service.model == "gemini-embedding-2"
    assert service.api_url == "https://generativelanguage.googleapis.com/v1beta/openai/"
    assert service.dimensions == 768
    assert service.api_key == "test_api_key"
    assert service.check_health() is True


def test_gemini_embedding_single_text(monkeypatch):
    """Verify single text embedding makes correct OpenAI-compatible request with dimensionality 768."""
    captured_request = {}

    def mock_post(url, headers=None, json=None):
        captured_request["url"] = url
        captured_request["headers"] = headers
        captured_request["json"] = json

        resp = MagicMock()
        resp.raise_for_status = MagicMock()
        # Return a simulated 768-dimensional embedding
        mock_embedding = [0.01 * i for i in range(768)]
        resp.json.return_value = {
            "data": [
                {
                    "object": "embedding",
                    "embedding": mock_embedding,
                    "index": 0
                }
            ],
            "model": "gemini-embedding-2"
        }
        return resp

    mock_client = MagicMock()
    mock_client.post.side_effect = mock_post
    mock_client.__enter__.return_value = mock_client
    mock_client.__exit__.return_value = None

    monkeypatch.setattr(httpx, "Client", lambda **kwargs: mock_client)

    service = EmbeddingService(
        provider="cloud",
        model="gemini-embedding-2",
        api_url="https://generativelanguage.googleapis.com/v1beta/openai/",
        api_key="test_key_123",
        dimensions=768,
    )

    vector = service.embed_text("Test single query embedding")

    assert len(vector) == 768
    assert captured_request["url"] == "https://generativelanguage.googleapis.com/v1beta/openai/embeddings"
    assert captured_request["headers"]["Authorization"] == "Bearer test_key_123"
    assert captured_request["headers"]["x-goog-api-key"] == "test_key_123"
    assert captured_request["json"]["model"] == "gemini-embedding-2"
    assert captured_request["json"]["input"] == "Test single query embedding"
    assert captured_request["json"]["dimensions"] == 768


def test_gemini_embedding_batch_texts(monkeypatch):
    """Verify batch text embedding makes correct batch request and returns ordered vectors."""
    captured_request = {}

    def mock_post(url, headers=None, json=None):
        captured_request["url"] = url
        captured_request["json"] = json

        resp = MagicMock()
        resp.raise_for_status = MagicMock()
        resp.json.return_value = {
            "data": [
                {"embedding": [0.1] * 768, "index": 0},
                {"embedding": [0.2] * 768, "index": 1},
            ],
            "model": "gemini-embedding-2"
        }
        return resp

    mock_client = MagicMock()
    mock_client.post.side_effect = mock_post
    mock_client.__enter__.return_value = mock_client
    mock_client.__exit__.return_value = None

    monkeypatch.setattr(httpx, "Client", lambda **kwargs: mock_client)

    service = EmbeddingService(
        provider="cloud",
        model="gemini-embedding-2",
        api_key="test_key",
        dimensions=768,
    )

    vectors = service.embed_batch(["First document text", "Second document text"])

    assert len(vectors) == 2
    assert len(vectors[0]) == 768
    assert len(vectors[1]) == 768
    assert vectors[0][0] == 0.1
    assert vectors[1][0] == 0.2
    assert captured_request["json"]["dimensions"] == 768
    assert captured_request["json"]["input"] == ["First document text", "Second document text"]


def test_gemini_embedding_dimension_validation(monkeypatch):
    """Verify that if upstream returns unexpected dimension (e.g. 3072), a clear ValueError is raised."""
    def mock_post(url, headers=None, json=None):
        resp = MagicMock()
        resp.raise_for_status = MagicMock()
        # Return 3072 dimensions instead of requested 768
        resp.json.return_value = {
            "data": [{"embedding": [0.05] * 3072, "index": 0}]
        }
        return resp

    mock_client = MagicMock()
    mock_client.post.side_effect = mock_post
    mock_client.__enter__.return_value = mock_client
    mock_client.__exit__.return_value = None

    monkeypatch.setattr(httpx, "Client", lambda **kwargs: mock_client)

    service = EmbeddingService(
        provider="cloud",
        dimensions=768,
    )

    with pytest.raises(ValueError, match="Expected embedding dimension 768"):
        service.embed_text("Sample input")


def test_gemini_generator_non_streaming(monkeypatch):
    """Verify LLMGenerator non-streaming generation via Gemini OpenAI-compatible endpoint."""
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
                        "content": "Alice found a golden key on the glass table."
                    }
                }
            ]
        }
        return resp

    mock_client = MagicMock()
    mock_client.post.side_effect = mock_post
    mock_client.__enter__.return_value = mock_client
    mock_client.__exit__.return_value = None

    monkeypatch.setattr(httpx, "Client", lambda **kwargs: mock_client)

    generator = LLMGenerator(
        provider="cloud",
        model="gemini-3.6-flash",
        api_url="https://generativelanguage.googleapis.com/v1beta/openai/",
        api_key="test_gemini_key",
    )

    chunks = [
        RetrievedChunk(
            text="Alice discovered a little three-legged table with a golden key on it.",
            metadata={"filename": "alice.pdf", "chunk_id": "chk1", "section": "Chapter 1"},
            score=0.95,
        )
    ]

    answer = generator.generate_answer("What was on the table?", chunks)

    assert "golden key" in answer.lower()
    assert captured_request["url"] == "https://generativelanguage.googleapis.com/v1beta/openai/chat/completions"
    assert captured_request["headers"]["Authorization"] == "Bearer test_gemini_key"
    assert captured_request["headers"]["x-goog-api-key"] == "test_gemini_key"
    assert captured_request["json"]["model"] == "gemini-3.6-flash"


def test_gemini_generator_streaming(monkeypatch):
    """Verify LLMGenerator streaming generation via Gemini OpenAI-compatible endpoint."""
    captured_request = {}

    sse_lines = [
        b'data: {"choices": [{"delta": {"content": "The "}}]}',
        b'data: {"choices": [{"delta": {"content": "key "}}]}',
        b'data: {"choices": [{"delta": {"content": "was gold."}}]}',
        b'data: [DONE]',
    ]

    class MockStreamResponse:
        def raise_for_status(self):
            pass

        def iter_lines(self):
            for line in sse_lines:
                yield line.decode("utf-8")

        def __enter__(self):
            return self

        def __exit__(self, exc_type, exc_val, exc_tb):
            pass

    mock_client = MagicMock()
    def mock_stream(method, url, headers=None, json=None):
        captured_request["url"] = url
        captured_request["json"] = json
        return MockStreamResponse()

    mock_client.stream.side_effect = mock_stream
    mock_client.__enter__.return_value = mock_client
    mock_client.__exit__.return_value = None

    monkeypatch.setattr(httpx, "Client", lambda **kwargs: mock_client)

    generator = LLMGenerator(
        provider="cloud",
        model="gemini-3.6-flash",
        api_key="test_gemini_key",
    )

    chunks = [
        RetrievedChunk(
            text="The key was gold and sat on the glass table.",
            metadata={"filename": "alice.pdf", "chunk_id": "chk1", "section": "Chapter 1"},
            score=0.9,
        )
    ]

    tokens = list(generator.generate_answer_stream("What was the key?", chunks))
    full_text = "".join(tokens)

    assert "The key was gold." in full_text
    assert captured_request["json"]["stream"] is True
    assert captured_request["json"]["model"] == "gemini-3.6-flash"


def test_ollama_local_mode_remains_intact():
    """Verify local Ollama provider instantiation is untouched."""
    generator = LLMGenerator(provider="ollama")
    assert generator.provider == "ollama"
    assert generator.model == "llama3.2:1b"
    assert generator.client is not None

    service = EmbeddingService(provider="ollama")
    assert service.provider == "ollama"
    assert service.model == "nomic-embed-text"
    assert service.client is not None
