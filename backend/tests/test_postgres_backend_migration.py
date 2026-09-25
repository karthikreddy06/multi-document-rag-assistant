"""
Tests for Dual Database Engine Backend (PostgreSQL / SQLite).
Verifies:
1. SQLite remains default and functional when DATABASE_URL is not set.
2. PostgreSQL configuration path and safe credential masking.
3. SQL translation layer (? -> %s, ON CONFLICT, COLLATE NOCASE, jsonb).
4. PostgreSQL repository operations using simulated/mocked database layer.
5. User registration/login, chats, documents, messages, attachments, user isolation, and duplicate protection.
6. Row normalization (datetime -> ISO string, sources_json preservation).
"""

from datetime import datetime, timezone
import json
import re
from unittest.mock import MagicMock, patch
import pytest

from app.config import Settings, settings
from app.db.database import (
    get_db_connection,
    get_db_path,
    get_pg_pool,
    close_db,
    get_safe_db_info,
    init_db,
)
from app.db import repository
from app.db.repository import (
    _translate_query,
    _row_to_dict,
    _rows_to_dicts,
    DBConnectionWrapper,
    create_user,
    get_user_by_email,
    get_user_by_id,
    list_users,
    create_chat,
    list_chats,
    get_chat,
    rename_chat,
    delete_chat,
    create_document,
    get_document_by_id,
    get_document_by_hash,
    update_document_status,
    list_documents,
    delete_document,
    attach_document_to_chat,
    detach_document_from_chat,
    list_chat_documents,
    document_attached_to_chat,
    create_message,
    list_messages,
)


# ==============================================================================
# 1. CONFIGURATION AND SAFE LOGGING TESTS
# ==============================================================================

def test_database_url_default_is_none():
    """Verify DATABASE_URL defaults to None and is_postgres is False in standard config."""
    s = Settings(DATABASE_URL=None)
    assert s.database_url is None
    assert s.is_postgres is False


def test_database_url_set_activates_postgres():
    """Verify setting DATABASE_URL activates is_postgres."""
    test_url = "postgresql://user:pass@aws-0-us-west-1.pooler.supabase.com:5432/postgres"
    s = Settings(DATABASE_URL=test_url)
    assert s.database_url == test_url
    assert s.is_postgres is True


def test_safe_db_info_masks_credentials():
    """Verify get_safe_db_info never prints database passwords or usernames."""
    secret_pass = "SuperSecretP@ssword123!"
    secret_user = "postgres_admin"
    url = f"postgresql://{secret_user}:{secret_pass}@aws-0-us-west-1.pooler.supabase.com:5432/postgres"

    safe_str = get_safe_db_info(url)
    assert secret_pass not in safe_str
    assert secret_user not in safe_str
    assert "aws-0-us-west-1.pooler.supabase.com:5432" in safe_str
    assert "postgres" in safe_str


def test_safe_db_info_sqlite_fallback():
    """Verify get_safe_db_info reports SQLite file path when url is None."""
    safe_str = get_safe_db_info(None)
    assert "SQLite" in safe_str
    assert "rag_app.db" in safe_str


# ==============================================================================
# 2. SQL DIALECT TRANSLATION TESTS
# ==============================================================================

def test_translate_query_sqlite_unchanged():
    """Verify that when is_postgres is False, query is returned unaltered."""
    sql = "SELECT * FROM users WHERE email = ? COLLATE NOCASE"
    assert _translate_query(sql, is_postgres=False) == sql


def test_translate_query_postgres_placeholders():
    """Verify question marks are converted to %s for PostgreSQL."""
    sql = "SELECT * FROM chats WHERE id = ? AND user_id = ?"
    translated = _translate_query(sql, is_postgres=True)
    assert translated == "SELECT * FROM chats WHERE id = %s AND user_id = %s"
    assert "?" not in translated


def test_translate_query_postgres_collate_nocase_removed():
    """Verify COLLATE NOCASE is safely removed for PostgreSQL CITEXT column."""
    sql = "SELECT * FROM users WHERE email = ? COLLATE NOCASE"
    translated = _translate_query(sql, is_postgres=True)
    assert translated == "SELECT * FROM users WHERE email = %s"
    assert "COLLATE" not in translated


def test_translate_query_postgres_insert_or_ignore_chat_documents():
    """Verify INSERT OR IGNORE INTO chat_documents translates to ON CONFLICT DO NOTHING."""
    sql = """
    INSERT OR IGNORE INTO chat_documents (chat_id, document_id, created_at)
    VALUES (?, ?, ?)
    """
    translated = _translate_query(sql, is_postgres=True)
    assert "INSERT INTO chat_documents (chat_id, document_id, created_at)" in translated
    assert "ON CONFLICT (chat_id, document_id) DO NOTHING" in translated
    assert "%s, %s, %s" in translated
    assert "?" not in translated


def test_translate_query_postgres_insert_messages_jsonb():
    """Verify message insertion adapts sources_json parameter to %s::jsonb."""
    sql = """
    INSERT INTO messages (id, chat_id, role, content, sources_json, created_at)
    VALUES (?, ?, ?, ?, ?, ?)
    """
    translated = _translate_query(sql, is_postgres=True)
    assert "VALUES (%s, %s, %s, %s, %s::jsonb, %s)" in translated


# ==============================================================================
# 3. ROW NORMALIZATION TESTS (DATETIME AND JSONB)
# ==============================================================================

def test_row_to_dict_normalizes_datetimes():
    """Verify datetime objects from PostgreSQL are converted to ISO-8601 strings."""
    now = datetime(2026, 9, 25, 12, 0, 0, tzinfo=timezone.utc)
    raw_row = {
        "id": "chat_123",
        "title": "Test Chat",
        "created_at": now,
        "updated_at": now,
    }
    normalized = _row_to_dict(raw_row)
    assert isinstance(normalized["created_at"], str)
    assert normalized["created_at"] == "2026-09-25T12:00:00+00:00"
    assert isinstance(normalized["updated_at"], str)


def test_row_to_dict_normalizes_sources_json():
    """Verify parsed JSONB objects/lists from PostgreSQL are converted to JSON strings."""
    sources_data = [{"section": "Summary", "filename": "doc.pdf"}]
    raw_row = {
        "id": "msg_123",
        "role": "assistant",
        "content": "Hello",
        "sources_json": sources_data,
        "created_at": "2026-09-25T12:00:00Z",
    }
    normalized = _row_to_dict(raw_row)
    assert isinstance(normalized["sources_json"], str)
    assert json.loads(normalized["sources_json"]) == sources_data


# ==============================================================================
# 4. MOCKED POSTGRESQL REPOSITORY OPERATIONS
# ==============================================================================

class MockPgCursor:
    """Mock psycopg cursor for simulating PostgreSQL queries and responses."""

    def __init__(self, data_store: dict):
        self.data_store = data_store
        self.last_query = ""
        self.last_params = ()
        self.rowcount = 0
        self._results = []

    def execute(self, query: str, params: tuple = ()):
        self.last_query = query
        self.last_params = params
        self.rowcount = 1

        clean = " ".join(query.split())

        # Simulate USERS table
        if clean.startswith("INSERT INTO users"):
            uid, email, pw_hash, created_at = params
            user_row = {"id": uid, "email": email, "password_hash": pw_hash, "created_at": datetime.now(timezone.utc)}
            self.data_store["users"][uid] = user_row
            self._results = []
        elif clean.startswith("SELECT * FROM users WHERE id = %s"):
            uid = params[0]
            user = self.data_store["users"].get(uid)
            self._results = [user] if user else []
        elif clean.startswith("SELECT * FROM users WHERE email = %s"):
            email = params[0].lower()
            matching = [u for u in self.data_store["users"].values() if u["email"].lower() == email]
            self._results = matching[:1]
        elif clean.startswith("SELECT * FROM users ORDER BY created_at ASC"):
            self._results = list(self.data_store["users"].values())

        # Simulate CHATS table
        elif clean.startswith("INSERT INTO chats"):
            cid, uid, title, created_at, updated_at = params
            chat_row = {"id": cid, "user_id": uid, "title": title, "created_at": datetime.now(timezone.utc), "updated_at": datetime.now(timezone.utc)}
            self.data_store["chats"][cid] = chat_row
            self._results = []
        elif re.search(r"SELECT\s+(?:\*|id)\s+FROM\s+chats\s+WHERE\s+id\s*=\s*%s\s+AND\s+user_id\s*=\s*%s", clean):
            cid, uid = params
            chat = self.data_store["chats"].get(cid)
            self._results = [chat] if (chat and chat["user_id"] == uid) else []
        elif re.search(r"SELECT\s+(?:\*|id)\s+FROM\s+chats\s+WHERE\s+id\s*=\s*%s", clean):
            cid = params[0]
            chat = self.data_store["chats"].get(cid)
            self._results = [chat] if chat else []
        elif clean.startswith("SELECT * FROM chats WHERE user_id = %s"):
            uid = params[0]
            self._results = [c for c in self.data_store["chats"].values() if c["user_id"] == uid]
        elif clean.startswith("SELECT * FROM chats ORDER BY updated_at DESC"):
            self._results = list(self.data_store["chats"].values())
        elif clean.startswith("UPDATE chats SET title = %s"):
            title, updated_at, cid = params[:3]
            if cid in self.data_store["chats"]:
                self.data_store["chats"][cid]["title"] = title
                self.rowcount = 1
            else:
                self.rowcount = 0
            self._results = []
        elif clean.startswith("UPDATE chats SET updated_at = %s WHERE id = %s"):
            updated_at, cid = params[:2]
            if cid in self.data_store["chats"]:
                self.data_store["chats"][cid]["updated_at"] = updated_at
                self.rowcount = 1
            else:
                self.rowcount = 0
            self._results = []
        elif clean.startswith("DELETE FROM chats"):
            cid = params[0]
            if cid in self.data_store["chats"]:
                del self.data_store["chats"][cid]
                self.rowcount = 1
            else:
                self.rowcount = 0
            self._results = []

        # Simulate DOCUMENTS table
        elif clean.startswith("INSERT INTO documents"):
            doc_id, uid, fn, fh, fs, pc, cc, sp, st, em, ca = params
            for existing in self.data_store["documents"].values():
                if existing["user_id"] == uid and existing["file_hash"] == fh:
                    raise RuntimeError("UniqueViolation: duplicate key value violates unique constraint 'uq_user_document_hash'")
            doc_row = {
                "id": doc_id, "user_id": uid, "filename": fn, "file_hash": fh,
                "file_size": fs, "page_count": pc, "chunk_count": cc,
                "storage_path": sp, "status": st, "error_message": em,
                "created_at": datetime.now(timezone.utc),
            }
            self.data_store["documents"][doc_id] = doc_row
            self._results = []
        elif re.search(r"SELECT\s+(?:\*|id)\s+FROM\s+documents\s+WHERE\s+id\s*=\s*%s\s+AND\s+user_id\s*=\s*%s", clean):
            doc_id, uid = params
            doc = self.data_store["documents"].get(doc_id)
            self._results = [doc] if (doc and doc["user_id"] == uid) else []
        elif re.search(r"SELECT\s+(?:\*|id)\s+FROM\s+documents\s+WHERE\s+id\s*=\s*%s", clean):
            doc_id = params[0]
            doc = self.data_store["documents"].get(doc_id)
            self._results = [doc] if doc else []
        elif clean.startswith("SELECT * FROM documents WHERE file_hash = %s AND user_id = %s"):
            fh, uid = params
            matches = [d for d in self.data_store["documents"].values() if d["file_hash"] == fh and d["user_id"] == uid]
            self._results = matches[:1]
        elif clean.startswith("SELECT * FROM documents WHERE user_id = %s"):
            uid = params[0]
            self._results = [d for d in self.data_store["documents"].values() if d["user_id"] == uid]
        elif clean.startswith("DELETE FROM documents WHERE id = %s AND user_id = %s"):
            doc_id, uid = params
            if doc_id in self.data_store["documents"] and self.data_store["documents"][doc_id]["user_id"] == uid:
                del self.data_store["documents"][doc_id]
                self.rowcount = 1
            else:
                self.rowcount = 0
            self._results = []
        elif clean.startswith("DELETE FROM documents"):
            doc_id = params[0]
            if doc_id in self.data_store["documents"]:
                del self.data_store["documents"][doc_id]
                self.rowcount = 1
            else:
                self.rowcount = 0
            self._results = []

        # Simulate CHAT_DOCUMENTS table
        elif "INSERT INTO chat_documents" in clean:
            cid, did, ca = params
            key = (cid, did)
            self.data_store["chat_documents"][key] = {"chat_id": cid, "document_id": did, "created_at": datetime.now(timezone.utc)}
            self._results = []
        elif clean.startswith("SELECT * FROM chat_documents WHERE chat_id = %s AND document_id = %s"):
            cid, did = params
            link = self.data_store["chat_documents"].get((cid, did))
            self._results = [link] if link else []
        elif clean.startswith("SELECT 1 FROM chat_documents"):
            cid, did = params
            link = self.data_store["chat_documents"].get((cid, did))
            self._results = [{"1": 1}] if link else []

        # Simulate MESSAGES table
        elif clean.startswith("INSERT INTO messages"):
            mid, cid, role, content, src, ca = params
            msg_row = {
                "id": mid, "chat_id": cid, "role": role, "content": content,
                "sources_json": json.loads(src) if (src and isinstance(src, str)) else src,
                "created_at": datetime.now(timezone.utc),
            }
            self.data_store["messages"][mid] = msg_row
            self._results = []
        elif clean.startswith("SELECT * FROM messages WHERE id = %s"):
            mid = params[0]
            msg = self.data_store["messages"].get(mid)
            self._results = [msg] if msg else []
        elif clean.startswith("SELECT * FROM messages WHERE chat_id = %s"):
            cid = params[0]
            self._results = [m for m in self.data_store["messages"].values() if m["chat_id"] == cid]

        return self

    def fetchone(self):
        if self._results:
            return self._results[0]
        return None

    def fetchall(self):
        return list(self._results)


class MockPgConnection:
    """Mock psycopg Connection object executing PostgreSQL queries."""

    def __init__(self):
        self.data_store = {
            "users": {},
            "chats": {},
            "documents": {},
            "chat_documents": {},
            "messages": {},
        }
        self.committed = False
        self.rolled_back = False

    def execute(self, query: str, params: tuple = ()):
        cur = MockPgCursor(self.data_store)
        return cur.execute(query, params)

    def commit(self):
        self.committed = True

    def rollback(self):
        self.rolled_back = True

    def close(self):
        pass


@pytest.fixture
def mock_pg_conn():
    """Provide a mock PostgreSQL connection configured for is_postgres=True."""
    return DBConnectionWrapper(MockPgConnection(), is_postgres=True)


def test_mock_pg_user_crud(mock_pg_conn):
    """Test user registration and lookup on PostgreSQL execution path."""
    user = create_user("Alice@Example.com", "hashed_pass_123", conn=mock_pg_conn)
    assert user["id"].startswith("usr_")
    assert user["email"] == "alice@example.com"
    assert isinstance(user["created_at"], str)

    # Email case-insensitive lookup
    found = get_user_by_email("ALICE@EXAMPLE.COM", conn=mock_pg_conn)
    assert found is not None
    assert found["id"] == user["id"]

    # ID lookup
    by_id = get_user_by_id(user["id"], conn=mock_pg_conn)
    assert by_id is not None
    assert by_id["email"] == "alice@example.com"


def test_mock_pg_chat_crud_and_isolation(mock_pg_conn):
    """Test chat creation, isolation, renaming, and listing on PostgreSQL path."""
    u1 = create_user("u1@test.com", "pw", conn=mock_pg_conn)["id"]
    u2 = create_user("u2@test.com", "pw", conn=mock_pg_conn)["id"]

    c1 = create_chat(title="Chat 1", user_id=u1, conn=mock_pg_conn)
    c2 = create_chat(title="Chat 2", user_id=u2, conn=mock_pg_conn)

    # Scoped listing
    u1_chats = list_chats(user_id=u1, conn=mock_pg_conn)
    assert len(u1_chats) == 1
    assert u1_chats[0]["id"] == c1["id"]

    u2_chats = list_chats(user_id=u2, conn=mock_pg_conn)
    assert len(u2_chats) == 1
    assert u2_chats[0]["id"] == c2["id"]

    # User isolation on get_chat
    assert get_chat(c1["id"], user_id=u1, conn=mock_pg_conn) is not None
    assert get_chat(c1["id"], user_id=u2, conn=mock_pg_conn) is None

    # Rename chat
    renamed = rename_chat(c1["id"], "Updated Title", user_id=u1, conn=mock_pg_conn)
    assert renamed is not None
    assert renamed["title"] == "Updated Title"

    # Delete chat
    assert delete_chat(c1["id"], user_id=u1, conn=mock_pg_conn) is True
    assert get_chat(c1["id"], conn=mock_pg_conn) is None


def test_mock_pg_document_and_chat_attachment(mock_pg_conn):
    """Test document cataloging and chat-document attachment on PostgreSQL path."""
    u1 = create_user("doc_user@test.com", "pw", conn=mock_pg_conn)["id"]
    chat = create_chat("Doc Chat", user_id=u1, conn=mock_pg_conn)

    doc = create_document(
        filename="report.pdf",
        file_hash="hash_abc123",
        file_size=1024,
        user_id=u1,
        conn=mock_pg_conn,
    )
    assert doc["id"].startswith("doc_")
    assert doc["filename"] == "report.pdf"

    # Lookup by hash
    by_hash = get_document_by_hash("hash_abc123", user_id=u1, conn=mock_pg_conn)
    assert by_hash is not None
    assert by_hash["id"] == doc["id"]

    # Attach document to chat
    rel = attach_document_to_chat(chat["id"], doc["id"], user_id=u1, conn=mock_pg_conn)
    assert rel["chat_id"] == chat["id"]
    assert rel["document_id"] == doc["id"]
    assert isinstance(rel["created_at"], str)

    # Check attached
    assert document_attached_to_chat(chat["id"], doc["id"], conn=mock_pg_conn) is True


def test_mock_pg_messages_and_sources_json(mock_pg_conn):
    """Test message creation and sources_json preservation on PostgreSQL path."""
    u1 = create_user("msg_user@test.com", "pw", conn=mock_pg_conn)["id"]
    chat = create_chat("Msg Chat", user_id=u1, conn=mock_pg_conn)

    sample_sources = [{"section": "Overview", "page": 1, "filename": "spec.pdf"}]
    msg = create_message(
        chat_id=chat["id"],
        role="assistant",
        content="Here is your answer.",
        sources_json=sample_sources,
        user_id=u1,
        conn=mock_pg_conn,
    )
    assert msg["id"].startswith("msg_")
    assert msg["role"] == "assistant"
    assert isinstance(msg["created_at"], str)

    # Verify sources_json is returned as a valid JSON string
    assert isinstance(msg["sources_json"], str)
    parsed = json.loads(msg["sources_json"])
    assert parsed == sample_sources

    # List messages
    messages = list_messages(chat["id"], user_id=u1, conn=mock_pg_conn)
    assert len(messages) == 1
    assert messages[0]["content"] == "Here is your answer."


def test_mock_pg_duplicate_document_protection(mock_pg_conn):
    """Verify duplicate (user_id, file_hash) insertion fails on PostgreSQL path."""
    u1 = create_user("dup_user@test.com", "pw", conn=mock_pg_conn)["id"]
    create_document("doc1.pdf", "hash_dup", 100, user_id=u1, conn=mock_pg_conn)

    with pytest.raises(Exception) as exc_info:
        create_document("doc2.pdf", "hash_dup", 100, user_id=u1, conn=mock_pg_conn)
    assert "uq_user_document_hash" in str(exc_info.value) or "UniqueViolation" in str(exc_info.value)


def test_mock_pg_document_and_message_user_isolation(mock_pg_conn):
    """Verify user isolation across documents and messages on PostgreSQL path."""
    u1 = create_user("owner@test.com", "pw", conn=mock_pg_conn)["id"]
    u2 = create_user("attacker@test.com", "pw", conn=mock_pg_conn)["id"]

    # u1 creates chat and doc
    c1 = create_chat("Private Chat", user_id=u1, conn=mock_pg_conn)
    d1 = create_document("secret.pdf", "hash_secret", 2048, user_id=u1, conn=mock_pg_conn)

    # u2 cannot access u1's document
    assert get_document_by_id(d1["id"], user_id=u2, conn=mock_pg_conn) is None
    assert get_document_by_hash("hash_secret", user_id=u2, conn=mock_pg_conn) is None
    assert len(list_documents(user_id=u2, conn=mock_pg_conn)) == 0
    assert delete_document(d1["id"], user_id=u2, conn=mock_pg_conn) is False

    # u2 cannot post to or list messages from u1's chat
    with pytest.raises(ValueError) as exc:
        create_message(c1["id"], "user", "Sneaky message", user_id=u2, conn=mock_pg_conn)
    assert "unauthorized" in str(exc.value).lower() or "not exist" in str(exc.value).lower()

    assert list_messages(c1["id"], user_id=u2, conn=mock_pg_conn) == []


def test_pg_pool_lifecycle(monkeypatch):
    """Verify PostgreSQL pool initialization and close_db teardown."""
    close_db()

    mock_pool = MagicMock()
    mock_pool.closed = False

    with patch("app.db.database.ConnectionPool", return_value=mock_pool) as mock_pool_cls:
        test_url = "postgresql://user:pass@aws-0-us-west-1.pooler.supabase.com:5432/postgres"
        pool = get_pg_pool(test_url)
        assert pool == mock_pool
        mock_pool_cls.assert_called_once()

        # Re-calling returns the same singleton
        same_pool = get_pg_pool(test_url)
        assert same_pool == pool
        assert mock_pool_cls.call_count == 1

        # close_db cleans up the pool
        close_db()
        mock_pool.close.assert_called_once()


def test_init_db_postgres_idempotent_verification(monkeypatch):
    """Verify init_db in PostgreSQL mode verifies tables without dropping or seeding legacy_user."""
    mock_conn = MagicMock()
    mock_cur = MagicMock()
    mock_conn.cursor.return_value.__enter__.return_value = mock_cur

    # Simulate existing tables in information_schema.tables
    mock_cur.fetchall.return_value = [
        {"table_name": "users"},
        {"table_name": "chats"},
        {"table_name": "documents"},
        {"table_name": "chat_documents"},
        {"table_name": "messages"},
    ]
    mock_cur.fetchone.return_value = {"count": 0}

    mock_pool = MagicMock()
    mock_pool.connection.return_value.__enter__.return_value = mock_conn

    monkeypatch.setattr("app.config.settings.database_url", "postgresql://u:p@aws-0-us-west-1.pooler.supabase.com:5432/postgres")
    monkeypatch.setattr("app.db.database.get_pg_pool", lambda *args, **kwargs: mock_pool)

    # Call init_db
    init_db()

    # Verify query checked information_schema.tables
    calls = [str(call) for call in mock_cur.execute.call_args_list]
    assert any("information_schema.tables" in call for call in calls)
    # Ensure no DROP or ALTER was executed
    assert not any("DROP" in call for call in calls)
    assert not any("legacy_user" in call for call in calls)

