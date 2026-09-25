"""
Tests for PgVectorStore with real Supabase PostgreSQL connection.
Verifies upsert, query, get, and delete operations against the document_chunks table.
"""

import pytest
from app.config import settings
from app.models import Chunk
from app.vectorstore.pgvector_store import PgVectorStore, parse_where_clause

pytestmark = pytest.mark.skipif(
    not settings.is_postgres,
    reason="Supabase PostgreSQL connection required for pgvector tests"
)


def test_parse_where_clause():
    where = {
        "$and": [
            {"filename": "doc.pdf"},
            {"user_id": "usr_test"},
            {"page_number": 2},
            {"custom_field": "val123"},
        ]
    }
    clause, params = parse_where_clause(where)
    assert "filename = %s" in clause
    assert "user_id = %s" in clause
    assert "page_number = %s" in clause
    assert "(metadata->>%s) = %s" in clause
    assert "doc.pdf" in params
    assert "usr_test" in params
    assert 2 in params
    assert "custom_field" in params
    assert "val123" in params


def test_pgvector_crud_lifecycle():
    from app.db.repository import create_document, delete_document

    store = PgVectorStore()
    test_id = "chunk_test_pgvector_lifecycle_001"
    test_doc_id = "doc_test_lifecycle_001"
    test_user_id = "legacy_user"
    test_filename = "lifecycle_test.pdf"

    # Clean up any leftover from previous runs
    store.delete_by_document_id(test_doc_id, user_id=test_user_id)
    try:
        delete_document(test_doc_id, user_id=test_user_id)
    except Exception:
        pass

    # Ensure parent document row exists for FK
    create_document(
        filename=test_filename,
        file_hash="hash_test_123_unique",
        file_size=1024,
        user_id=test_user_id,
        document_id=test_doc_id,
        status="ready",
    )

    dummy_vec = [0.05] * 384
    chunk = Chunk(
        text="This is a test chunk for pgvector lifecycle testing.",
        metadata={
            "chunk_id": test_id,
            "document_id": test_doc_id,
            "user_id": test_user_id,
            "filename": test_filename,
            "page_number": 1,
            "section": "Introduction",
            "file_hash": "hash_test_123",
        },
    )

    # 1. Upsert
    added = store.upsert_chunks([chunk], [dummy_vec])
    assert added == 1

    # 2. Get with where filter
    get_res = store.get(where={"$and": [{"document_id": test_doc_id}, {"user_id": test_user_id}]})
    assert len(get_res["ids"]) >= 1
    assert test_id in get_res["ids"]
    assert get_res["metadatas"][0]["filename"] == test_filename

    # 3. Query with cosine similarity
    query_res = store.query(
        query_embedding=dummy_vec,
        top_k=5,
        where={"user_id": test_user_id},
    )
    assert len(query_res["ids"][0]) >= 1
    assert test_id in query_res["ids"][0]
    # Cosine distance to itself should be ~ 0.0
    assert abs(query_res["distances"][0][0]) < 1e-4

    # 4. Collection shim get()
    shim_res = store.collection.get(where={"filename": test_filename})
    assert test_id in shim_res["ids"]

    # 5. Delete by document_id
    deleted = store.delete_by_document_id(test_doc_id, user_id=test_user_id)
    assert deleted >= 1

    # Verify deleted
    after_get = store.get(where={"document_id": test_doc_id})
    assert len(after_get["ids"]) == 0

    # Clean up test document
    delete_document(test_doc_id, user_id=test_user_id)
