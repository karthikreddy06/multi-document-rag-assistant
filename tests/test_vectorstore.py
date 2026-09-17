"""
Tests for ChromaDB VectorStore.
"""

import pytest
from app.models import Chunk
from app.vectorstore.store import VectorStore


def test_vectorstore_init():
    vs = VectorStore()
    assert vs.client is not None
    assert vs.collection is not None


def test_vectorstore_upsert_idempotency(tmp_path):
    # Use a temporary collection to avoid modifying the main database
    test_store = VectorStore(
        persist_path=str(tmp_path / "test_chroma"),
        collection_name="test_collection"
    )

    chunks = [
        Chunk(text="Python backend with FastAPI", metadata={"chunk_id": "test_1", "section": "Backend"}),
        Chunk(text="React frontend with Tailwind", metadata={"chunk_id": "test_2", "section": "Frontend"}),
    ]
    # Fake 4-dimensional embeddings for fast testing
    fake_embeddings = [[0.1, 0.2, 0.3, 0.4], [0.5, 0.6, 0.7, 0.8]]

    # First insert
    count1 = test_store.upsert_chunks(chunks, fake_embeddings)
    assert count1 == 2
    assert test_store.count() == 2

    # Second insert with same IDs (idempotency check)
    count2 = test_store.upsert_chunks(chunks, fake_embeddings)
    assert count2 == 2
    assert test_store.count() == 2  # Count must remain 2, not 4!


def test_vectorstore_query(tmp_path):
    test_store = VectorStore(
        persist_path=str(tmp_path / "test_chroma_query"),
        collection_name="test_query_collection"
    )
    chunks = [
        Chunk(text="PostgreSQL and MongoDB databases", metadata={"chunk_id": "db_1", "section": "Databases"}),
    ]
    test_store.upsert_chunks(chunks, [[1.0, 0.0, 0.0, 0.0]])

    res = test_store.query(query_embedding=[1.0, 0.0, 0.0, 0.0], top_k=1)
    assert len(res["ids"][0]) == 1
    assert res["ids"][0][0] == "db_1"
    assert "Databases" in res["metadatas"][0][0]["section"]
