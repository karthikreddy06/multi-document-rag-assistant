"""
Tests for FastAPI HTTP API Layer.
Validates health, chat, documents, and ingestion endpoints using FastAPI TestClient.
"""

from unittest.mock import MagicMock
import pytest
from fastapi.testclient import TestClient

from app.api.routes import app, get_rag_app
from app.models import RetrievedChunk


@pytest.fixture
def client():
    """FastAPI TestClient fixture."""
    return TestClient(app)


def test_api_health(client):
    """Test GET /api/health endpoint."""
    response = client.get("/api/health")
    assert response.status_code == 200
    data = response.json()
    assert data == {"status": "ok"}


def test_api_documents(client):
    """Test GET /api/documents endpoint."""
    response = client.get("/api/documents")
    assert response.status_code == 200
    data = response.json()
    assert "total_documents" in data
    assert "total_chunks" in data
    assert "documents" in data
    assert isinstance(data["documents"], list)


def test_api_chat_empty_query(client):
    """Test POST /api/chat validation for empty string."""
    response = client.post("/api/chat", json={"query": ""})
    # Pydantic min_length=1 returns 422 Unprocessable Entity
    assert response.status_code in (400, 422)


def test_api_chat_whitespace_query(client):
    """Test POST /api/chat validation for whitespace string."""
    response = client.post("/api/chat", json={"query": "   "})
    assert response.status_code == 400
    assert "empty" in response.json()["detail"].lower()


def test_api_chat_with_mock_rag(client):
    """Test POST /api/chat using dependency override for fast non-LLM testing."""
    mock_rag = MagicMock()
    mock_chunk = RetrievedChunk(
        text="Santiago was an old fisherman who fished in the Gulf Stream.",
        metadata={"filename": "oldmansea.pdf", "page_number": 2, "section": "Chapter 1", "chunk_id": "c_1"},
        score=0.92,
    )
    mock_rag.retriever.retrieve.return_value = [mock_chunk]
    mock_rag.generator.generate_answer.return_value = "The main character is Santiago, an old fisherman."

    # Override dependency
    app.dependency_overrides[get_rag_app] = lambda: mock_rag

    try:
        response = client.post(
            "/api/chat",
            json={"query": "Who is the main character in The Old Man and the Sea?", "show_context": True},
        )
        assert response.status_code == 200
        data = response.json()
        assert "answer" in data
        assert "sources" in data
        assert "Santiago" in data["answer"]
        assert len(data["sources"]) == 1
        assert data["sources"][0]["filename"] == "oldmansea.pdf"
        assert data["sources"][0]["page"] == 2
        assert data["sources"][0]["score"] == 0.92
        assert "Gulf Stream" in data["sources"][0]["text"]
    finally:
        app.dependency_overrides.clear()


def test_api_ingest_with_mock_rag(client):
    """Test POST /api/ingest using dependency override."""
    mock_rag = MagicMock()
    mock_rag.pipeline.ingest_directory.return_value = {"documents": 3, "chunks": 45}

    app.dependency_overrides[get_rag_app] = lambda: mock_rag
    try:
        response = client.post("/api/ingest")
        assert response.status_code == 200
        data = response.json()
        assert data["status"] == "success"
        assert data["documents_processed"] == 3
        assert data["chunks_stored"] == 45
    finally:
        app.dependency_overrides.clear()


def test_api_reindex_with_mock_rag(client):
    """Test POST /api/reindex using dependency override."""
    mock_rag = MagicMock()
    mock_rag.pipeline.ingest_directory.return_value = {"documents": 3, "chunks": 45}

    app.dependency_overrides[get_rag_app] = lambda: mock_rag
    try:
        response = client.post("/api/reindex")
        assert response.status_code == 200
        data = response.json()
        assert data["status"] == "success"
        assert data["documents_processed"] == 3
        assert data["chunks_stored"] == 45
        mock_rag.vector_store.clear.assert_called_once()
    finally:
        app.dependency_overrides.clear()
