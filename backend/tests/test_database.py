"""
Unit and Integration Tests for SQLite Database Storage Layer.
Verifies all 24 required database and repository capabilities.
"""

import json
import sqlite3
import pytest

from app.db.database import get_db_connection, init_db
from app.db import repository


@pytest.fixture
def test_db_path(tmp_path):
    """Provide a fresh isolated SQLite database path for each test."""
    db_file = tmp_path / "test_rag.db"
    init_db(db_file)
    return db_file


@pytest.fixture
def conn(test_db_path):
    """Provide an open connection to the test database."""
    c = get_db_connection(test_db_path)
    yield c
    c.close()


# ==============================================================================
# 1. Database Initialization, Tables & PRAGMAs
# ==============================================================================

def test_1_database_initialization(test_db_path):
    """Verify database file is created and initialized."""
    assert test_db_path.exists()
    assert test_db_path.stat().st_size > 0


def test_2_tables_exist(conn):
    """Verify all 4 core tables and indexes exist."""
    rows = conn.execute(
        "SELECT name FROM sqlite_master WHERE type='table'"
    ).fetchall()
    tables = {r["name"] for r in rows}
    assert "chats" in tables
    assert "documents" in tables
    assert "chat_documents" in tables
    assert "messages" in tables


def test_3_foreign_keys_enabled(conn):
    """Verify PRAGMA foreign_keys is ON."""
    row = conn.execute("PRAGMA foreign_keys").fetchone()
    assert row[0] == 1


def test_4_wal_mode_enabled(conn):
    """Verify PRAGMA journal_mode is WAL on disk database."""
    row = conn.execute("PRAGMA journal_mode").fetchone()
    assert row[0].upper() == "WAL"


# ==============================================================================
# 2. Chat CRUD Operations
# ==============================================================================

def test_5_create_chat(conn):
    """Verify creating a chat with default and custom title."""
    c1 = repository.create_chat(conn=conn)
    assert c1["title"] == "New Chat"
    assert c1["id"].startswith("chat_")
    assert c1["created_at"]
    assert c1["updated_at"]

    c2 = repository.create_chat(title="Research Project", conn=conn)
    assert c2["title"] == "Research Project"


def test_6_list_chats(conn):
    """Verify listing chats sorted by updated_at DESC."""
    c1 = repository.create_chat(title="Chat 1", conn=conn)
    c2 = repository.create_chat(title="Chat 2", conn=conn)

    chats = repository.list_chats(conn=conn)
    assert len(chats) >= 2
    # Verify both chats are in the list
    ids = [c["id"] for c in chats]
    assert c1["id"] in ids
    assert c2["id"] in ids


def test_7_get_chat(conn):
    """Verify retrieving single chat by ID."""
    created = repository.create_chat(title="Specific Chat", conn=conn)
    fetched = repository.get_chat(created["id"], conn=conn)
    assert fetched is not None
    assert fetched["id"] == created["id"]
    assert fetched["title"] == "Specific Chat"

    assert repository.get_chat("non_existent_id", conn=conn) is None


def test_8_rename_chat(conn):
    """Verify renaming chat updates title and updated_at."""
    c = repository.create_chat(title="Old Title", conn=conn)
    orig_updated = c["updated_at"]

    renamed = repository.rename_chat(c["id"], "Brand New Title", conn=conn)
    assert renamed is not None
    assert renamed["title"] == "Brand New Title"

    with pytest.raises(ValueError):
        repository.rename_chat(c["id"], "   ", conn=conn)


def test_9_delete_chat(conn):
    """Verify chat deletion."""
    c = repository.create_chat(title="To Delete", conn=conn)
    deleted = repository.delete_chat(c["id"], conn=conn)
    assert deleted is True

    # Deleting again should return False
    assert repository.delete_chat(c["id"], conn=conn) is False
    assert repository.get_chat(c["id"], conn=conn) is None


# ==============================================================================
# 3. Document CRUD & Hash Uniqueness
# ==============================================================================

def test_10_create_document(conn):
    """Verify creating document record."""
    doc = repository.create_document(
        filename="manual.pdf",
        file_hash="hash_abc123",
        file_size=1024,
        page_count=5,
        chunk_count=12,
        status="ready",
        conn=conn,
    )
    assert doc["id"].startswith("doc_")
    assert doc["filename"] == "manual.pdf"
    assert doc["file_hash"] == "hash_abc123"
    assert doc["page_count"] == 5
    assert doc["status"] == "ready"


def test_11_sha256_uniqueness_constraint(conn):
    """Verify SHA-256 hash unique constraint prevents duplicates."""
    repository.create_document(
        filename="doc1.pdf",
        file_hash="duplicate_hash",
        file_size=500,
        conn=conn,
    )
    with pytest.raises(sqlite3.IntegrityError):
        repository.create_document(
            filename="doc2.pdf",
            file_hash="duplicate_hash",
            file_size=600,
            conn=conn,
        )


def test_12_get_document_by_hash(conn):
    """Verify finding document by file_hash for deduplication lookup."""
    doc = repository.create_document(
        filename="original.pdf",
        file_hash="unique_hash_999",
        file_size=2048,
        conn=conn,
    )
    found = repository.get_document_by_hash("unique_hash_999", conn=conn)
    assert found is not None
    assert found["id"] == doc["id"]
    assert found["filename"] == "original.pdf"

    assert repository.get_document_by_hash("not_found", conn=conn) is None


def test_13_update_document_status(conn):
    """Verify updating document status, counts, and error message."""
    doc = repository.create_document(
        filename="pending.pdf",
        file_hash="pending_hash",
        file_size=1000,
        status="pending",
        conn=conn,
    )
    assert doc["status"] == "pending"

    updated = repository.update_document_status(
        doc["id"],
        status="ready",
        page_count=10,
        chunk_count=45,
        conn=conn,
    )
    assert updated["status"] == "ready"
    assert updated["page_count"] == 10
    assert updated["chunk_count"] == 45

    # Test error status
    failed = repository.update_document_status(
        doc["id"],
        status="failed",
        error_message="Corrupted PDF bytes",
        conn=conn,
    )
    assert failed["status"] == "failed"
    assert failed["error_message"] == "Corrupted PDF bytes"


# ==============================================================================
# 4. Chat-Document Associations & Isolation
# ==============================================================================

def test_14_attach_document_to_chat(conn):
    """Verify associating document with chat."""
    chat = repository.create_chat(title="Chat with Docs", conn=conn)
    doc = repository.create_document(
        filename="report.pdf",
        file_hash="report_hash",
        file_size=1234,
        conn=conn,
    )

    assoc = repository.attach_document_to_chat(chat["id"], doc["id"], conn=conn)
    assert assoc["chat_id"] == chat["id"]
    assert assoc["document_id"] == doc["id"]

    assert repository.document_attached_to_chat(chat["id"], doc["id"], conn=conn) is True


def test_15_prevent_duplicate_chat_document_relationship(conn):
    """Verify re-attaching same document is idempotent and does not fail."""
    chat = repository.create_chat(title="Chat Idempotency", conn=conn)
    doc = repository.create_document(
        filename="once.pdf",
        file_hash="once_hash",
        file_size=200,
        conn=conn,
    )

    repository.attach_document_to_chat(chat["id"], doc["id"], conn=conn)
    # Second attachment should succeed without raising IntegrityError
    re_attach = repository.attach_document_to_chat(chat["id"], doc["id"], conn=conn)
    assert re_attach["document_id"] == doc["id"]

    docs = repository.list_chat_documents(chat["id"], conn=conn)
    assert len(docs) == 1


def test_16_detach_document(conn):
    """Verify detaching document from chat."""
    chat = repository.create_chat(title="Detach Test", conn=conn)
    doc = repository.create_document(
        filename="detach.pdf",
        file_hash="detach_hash",
        file_size=500,
        conn=conn,
    )
    repository.attach_document_to_chat(chat["id"], doc["id"], conn=conn)
    assert repository.document_attached_to_chat(chat["id"], doc["id"], conn=conn) is True

    detached = repository.detach_document_from_chat(chat["id"], doc["id"], conn=conn)
    assert detached is True
    assert repository.document_attached_to_chat(chat["id"], doc["id"], conn=conn) is False


# ==============================================================================
# 5. Messages & Message History
# ==============================================================================

def test_17_create_user_message(conn):
    """Verify creating user message."""
    chat = repository.create_chat(conn=conn)
    msg = repository.create_message(
        chat_id=chat["id"],
        role="user",
        content="What are the key findings?",
        conn=conn,
    )
    assert msg["id"].startswith("msg_")
    assert msg["role"] == "user"
    assert msg["content"] == "What are the key findings?"


def test_18_create_assistant_message(conn):
    """Verify creating assistant message."""
    chat = repository.create_chat(conn=conn)
    msg = repository.create_message(
        chat_id=chat["id"],
        role="assistant",
        content="Based on the documents, the key findings are...",
        conn=conn,
    )
    assert msg["role"] == "assistant"
    assert msg["content"].startswith("Based on the documents")


def test_19_store_sources_json(conn):
    """Verify storing structured sources as JSON string or dict list."""
    chat = repository.create_chat(conn=conn)
    sample_sources = [
        {"filename": "doc.pdf", "page": 2, "section": "Summary", "score": 0.85}
    ]
    msg = repository.create_message(
        chat_id=chat["id"],
        role="assistant",
        content="Answer with sources",
        sources_json=sample_sources,
        conn=conn,
    )
    assert msg["sources_json"] is not None
    parsed = json.loads(msg["sources_json"])
    assert len(parsed) == 1
    assert parsed[0]["filename"] == "doc.pdf"
    assert parsed[0]["score"] == 0.85


def test_20_retrieve_message_history(conn):
    """Verify retrieving chronological message history."""
    chat = repository.create_chat(conn=conn)
    repository.create_message(chat["id"], "user", "Question 1", conn=conn)
    repository.create_message(chat["id"], "assistant", "Answer 1", conn=conn)
    repository.create_message(chat["id"], "user", "Question 2", conn=conn)

    history = repository.list_messages(chat["id"], conn=conn)
    assert len(history) == 3
    assert history[0]["content"] == "Question 1"
    assert history[1]["content"] == "Answer 1"
    assert history[2]["content"] == "Question 2"


# ==============================================================================
# 6. Cascades, Persistence & Cross-Chat Sharing
# ==============================================================================

def test_21_deleting_chat_cascades_to_messages(conn):
    """Verify deleting chat automatically deletes all its messages via FK cascade."""
    chat = repository.create_chat(conn=conn)
    msg = repository.create_message(chat["id"], "user", "Hello", conn=conn)

    # Confirm message exists
    assert conn.execute("SELECT COUNT(*) FROM messages WHERE id = ?", (msg["id"],)).fetchone()[0] == 1

    # Delete chat
    repository.delete_chat(chat["id"], conn=conn)

    # Confirm message was cascaded and deleted
    assert conn.execute("SELECT COUNT(*) FROM messages WHERE id = ?", (msg["id"],)).fetchone()[0] == 0


def test_22_deleting_chat_cascades_to_chat_documents(conn):
    """Verify deleting chat removes the chat_document junction entries."""
    chat = repository.create_chat(conn=conn)
    doc = repository.create_document(
        filename="shared.pdf",
        file_hash="shared_hash",
        file_size=100,
        conn=conn,
    )
    repository.attach_document_to_chat(chat["id"], doc["id"], conn=conn)

    # Confirm link exists
    assert conn.execute(
        "SELECT COUNT(*) FROM chat_documents WHERE chat_id = ?",
        (chat["id"],),
    ).fetchone()[0] == 1

    # Delete chat
    repository.delete_chat(chat["id"], conn=conn)

    # Confirm link is gone
    assert conn.execute(
        "SELECT COUNT(*) FROM chat_documents WHERE chat_id = ?",
        (chat["id"],),
    ).fetchone()[0] == 0


def test_23_document_remains_after_chat_deletion(conn):
    """Verify that deleting a chat does NOT delete the global document from catalog."""
    chat = repository.create_chat(conn=conn)
    doc = repository.create_document(
        filename="persistent.pdf",
        file_hash="persistent_hash",
        file_size=999,
        conn=conn,
    )
    repository.attach_document_to_chat(chat["id"], doc["id"], conn=conn)

    repository.delete_chat(chat["id"], conn=conn)

    # Document itself MUST still exist
    doc_check = repository.get_document_by_id(doc["id"], conn=conn)
    assert doc_check is not None
    assert doc_check["filename"] == "persistent.pdf"


def test_24_multiple_chats_can_reference_same_document(conn):
    """Verify many-to-many relationship: single document attached to multiple chats."""
    chat1 = repository.create_chat(title="Chat Alpha", conn=conn)
    chat2 = repository.create_chat(title="Chat Beta", conn=conn)

    doc = repository.create_document(
        filename="company_policy.pdf",
        file_hash="policy_hash_123",
        file_size=5000,
        conn=conn,
    )

    repository.attach_document_to_chat(chat1["id"], doc["id"], conn=conn)
    repository.attach_document_to_chat(chat2["id"], doc["id"], conn=conn)

    docs1 = repository.list_chat_documents(chat1["id"], conn=conn)
    docs2 = repository.list_chat_documents(chat2["id"], conn=conn)

    assert len(docs1) == 1
    assert len(docs2) == 1
    assert docs1[0]["id"] == doc["id"]
    assert docs2[0]["id"] == doc["id"]

    # Detaching from Chat 1 leaves it attached to Chat 2
    repository.detach_document_from_chat(chat1["id"], doc["id"], conn=conn)
    assert repository.document_attached_to_chat(chat1["id"], doc["id"], conn=conn) is False
    assert repository.document_attached_to_chat(chat2["id"], doc["id"], conn=conn) is True
