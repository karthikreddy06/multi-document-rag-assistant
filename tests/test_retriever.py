"""
Tests for HybridRetriever.
"""

import pytest
from app.retrieval.retriever import HybridRetriever, RetrievedChunk
from app.vectorstore.store import VectorStore


def test_retriever_empty_query():
    retriever = HybridRetriever()
    results = retriever.retrieve("   ")
    assert results == []


def test_retriever_live_query():
    vs = VectorStore()
    if vs.count() == 0:
        pytest.skip("ChromaDB has 0 chunks. Ingestion required.")

    retriever = HybridRetriever(top_k=5)
    results = retriever.retrieve("What are all my skills?")
    assert len(results) > 0
    assert any("SKILLS" in r.section.upper() for r in results)


def test_retriever_project_query():
    vs = VectorStore()
    if vs.count() == 0:
        pytest.skip("ChromaDB has 0 chunks. Ingestion required.")

    retriever = HybridRetriever(top_k=5)
    results = retriever.retrieve("Tell me about TravelTrack")
    assert len(results) > 0
    # Top chunk should be TravelTrack
    assert any("TRAVELTRACK" in r.section.upper() or "TRAVELTRACK" in r.text.upper() for r in results)
