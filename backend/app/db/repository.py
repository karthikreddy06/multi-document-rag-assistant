"""
Database Repository Module.
Provides CRUD operations for chats, documents, chat-document links, and messages.
Supports dual database backends: PostgreSQL (production/Supabase) and SQLite (local/tests).
All database access uses parameterized SQL and consistent UTC ISO-8601 timestamps.
"""

from contextlib import contextmanager
from datetime import datetime, timezone
import json
from pathlib import Path
import re
import sqlite3
from typing import Any, Dict, Generator, List, Optional
import uuid

from app.config import settings
from app.db.database import get_db_connection, get_pg_pool


def _now_utc_iso() -> str:
    """Return current UTC timestamp in ISO 8601 format."""
    return datetime.now(timezone.utc).isoformat()


def _translate_query(sql: str, is_postgres: bool) -> str:
    """
    Translate SQL statements between SQLite and PostgreSQL dialects safely.
    - Parameter placeholders: ? -> %s
    - Case-insensitive comparison: removes SQLite COLLATE NOCASE (PostgreSQL uses CITEXT)
    - Conflict handling: translates INSERT OR IGNORE to ON CONFLICT DO NOTHING
    - Native JSONB: casts sources_json placeholder in message inserts to %s::jsonb
    """
    if not is_postgres:
        return sql

    translated = sql

    # 1. Translate INSERT OR IGNORE to PostgreSQL ON CONFLICT DO NOTHING
    if re.search(r"INSERT\s+OR\s+IGNORE\s+INTO\s+chat_documents", translated, flags=re.IGNORECASE):
        translated = re.sub(
            r"INSERT\s+OR\s+IGNORE\s+INTO\s+chat_documents\s*\((.*?)\)\s*VALUES\s*\((.*?)\)",
            r"INSERT INTO chat_documents (\1) VALUES (\2) ON CONFLICT (chat_id, document_id) DO NOTHING",
            translated,
            flags=re.IGNORECASE | re.DOTALL,
        )
    elif re.search(r"INSERT\s+OR\s+IGNORE\s+INTO\s+users", translated, flags=re.IGNORECASE):
        translated = re.sub(
            r"INSERT\s+OR\s+IGNORE\s+INTO\s+users\s*\((.*?)\)\s*VALUES\s*\((.*?)\)",
            r"INSERT INTO users (\1) VALUES (\2) ON CONFLICT (id) DO NOTHING",
            translated,
            flags=re.IGNORECASE | re.DOTALL,
        )

    # 2. Remove SQLite-specific COLLATE NOCASE (PostgreSQL uses CITEXT for case-insensitive emails)
    translated = re.sub(r"\s+COLLATE\s+NOCASE", "", translated, flags=re.IGNORECASE)

    # 3. Explicit JSONB cast for messages sources_json column
    if re.search(r"INSERT\s+INTO\s+messages", translated, flags=re.IGNORECASE):
        translated = re.sub(
            r"VALUES\s*\(\s*\?,\s*\?,\s*\?,\s*\?,\s*\?,\s*\?\s*\)",
            r"VALUES (%s, %s, %s, %s, %s::jsonb, %s)",
            translated,
            flags=re.IGNORECASE,
        )

    # 4. Convert parameter placeholders: ? -> %s
    translated = translated.replace("?", "%s")

    return translated


class DBConnectionWrapper:
    """
    Lightweight wrapper providing cross-engine database execution.
    Transparently adapts parameter binding (? -> %s), dialect differences,
    and returns cursor results compatible with dictionary row factories.
    """

    def __init__(self, raw_conn: Any, is_postgres: Optional[bool] = None):
        self.raw_conn = raw_conn
        if is_postgres is not None:
            self.is_postgres = is_postgres
        elif isinstance(raw_conn, sqlite3.Connection):
            self.is_postgres = False
        else:
            self.is_postgres = getattr(settings, "is_postgres", False) or type(raw_conn).__module__.startswith("psycopg")

    def execute(self, sql: str, params: tuple | list = ()):
        translated_sql = _translate_query(sql, self.is_postgres)
        if params is None:
            params = ()
        elif not isinstance(params, (tuple, list)):
            params = tuple(params)
        return self.raw_conn.execute(translated_sql, params)

    def commit(self):
        if hasattr(self.raw_conn, "commit"):
            self.raw_conn.commit()

    def rollback(self):
        if hasattr(self.raw_conn, "rollback"):
            self.raw_conn.rollback()

    def close(self):
        if hasattr(self.raw_conn, "close"):
            self.raw_conn.close()

    def __getattr__(self, item):
        return getattr(self.raw_conn, item)


def _row_to_dict(row: Optional[Any]) -> Optional[Dict[str, Any]]:
    """
    Convert a database row (sqlite3.Row or psycopg dict) to a standard Python dictionary.
    Normalizes timestamps to ISO-8601 strings and sources_json to JSON strings for API consistency.
    """
    if row is None:
        return None
    d = dict(row)
    for k, v in d.items():
        if isinstance(v, datetime):
            d[k] = v.isoformat()
        elif k == "sources_json" and isinstance(v, (dict, list)):
            d[k] = json.dumps(v)
    return d


def _rows_to_dicts(rows: List[Any]) -> List[Dict[str, Any]]:
    """Convert a list of database rows to standard Python dictionaries."""
    return [_row_to_dict(r) for r in rows if r is not None]


@contextmanager
def _managed_connection(
    conn: Optional[Any] = None,
    custom_path: Optional[str | Path] = None,
) -> Generator[DBConnectionWrapper, None, None]:
    """
    Yield an active database connection wrapped for cross-engine compatibility.
    - If conn is provided, wrap and yield without closing (caller manages lifetime).
    - If conn is None and SQLite mode (default / no DATABASE_URL): open, commit, and close.
    - If conn is None and PostgreSQL mode (DATABASE_URL set): borrow from ConnectionPool with transaction.
    """
    if conn is not None:
        if isinstance(conn, DBConnectionWrapper):
            yield conn
        else:
            yield DBConnectionWrapper(conn)
    else:
        if custom_path is not None or not settings.is_postgres:
            new_conn = get_db_connection(custom_path)
            wrapper = DBConnectionWrapper(new_conn, is_postgres=False)
            try:
                yield wrapper
                new_conn.commit()
            except Exception:
                new_conn.rollback()
                raise
            finally:
                new_conn.close()
        else:
            pool = get_pg_pool()
            with pool.connection() as pg_conn:
                wrapper = DBConnectionWrapper(pg_conn, is_postgres=True)
                yield wrapper


# ==============================================================================
# 0. USER REPOSITORY
# ==============================================================================

def create_user(
    email: str,
    password_hash: str,
    user_id: Optional[str] = None,
    conn: Optional[Any] = None,
) -> Dict[str, Any]:
    """Create a new user with normalized email and password hash."""
    clean_email = email.strip().lower()
    if not clean_email or "@" not in clean_email:
        raise ValueError("Invalid email address.")
    if not password_hash:
        raise ValueError("Password hash cannot be empty.")

    uid = user_id or f"usr_{uuid.uuid4().hex[:16]}"
    now = _now_utc_iso()

    with _managed_connection(conn) as c:
        c.execute(
            "INSERT INTO users (id, email, password_hash, created_at) VALUES (?, ?, ?, ?)",
            (uid, clean_email, password_hash, now),
        )
        row = c.execute("SELECT * FROM users WHERE id = ?", (uid,)).fetchone()
        return _row_to_dict(row) or {}


def get_user_by_email(
    email: str,
    conn: Optional[Any] = None,
) -> Optional[Dict[str, Any]]:
    """Retrieve user record by email address (case-insensitive)."""
    clean_email = email.strip().lower()
    with _managed_connection(conn) as c:
        row = c.execute("SELECT * FROM users WHERE email = ? COLLATE NOCASE", (clean_email,)).fetchone()
        return _row_to_dict(row)


def get_user_by_id(
    user_id: str,
    conn: Optional[Any] = None,
) -> Optional[Dict[str, Any]]:
    """Retrieve user record by user ID."""
    with _managed_connection(conn) as c:
        row = c.execute("SELECT * FROM users WHERE id = ?", (user_id,)).fetchone()
        return _row_to_dict(row)


def list_users(conn: Optional[Any] = None) -> List[Dict[str, Any]]:
    """List all registered users ordered by registration date."""
    with _managed_connection(conn) as c:
        rows = c.execute("SELECT * FROM users ORDER BY created_at ASC").fetchall()
        return _rows_to_dicts(rows)


# ==============================================================================
# 1. CHAT REPOSITORY
# ==============================================================================

def create_chat(
    title: str = "New Chat",
    user_id: Optional[str] = None,
    chat_id: Optional[str] = None,
    conn: Optional[Any] = None,
) -> Dict[str, Any]:
    """Create a new chat session bound to a user (defaults to legacy_user)."""
    cid = chat_id or f"chat_{uuid.uuid4().hex[:16]}"
    now = _now_utc_iso()
    clean_title = title.strip() or "New Chat"
    uid = user_id or "legacy_user"

    with _managed_connection(conn) as c:
        c.execute(
            "INSERT INTO chats (id, user_id, title, created_at, updated_at) VALUES (?, ?, ?, ?, ?)",
            (cid, uid, clean_title, now, now),
        )
        row = c.execute("SELECT * FROM chats WHERE id = ?", (cid,)).fetchone()
        return _row_to_dict(row) or {}


def list_chats(
    user_id: Optional[str] = None,
    conn: Optional[Any] = None,
) -> List[Dict[str, Any]]:
    """List all chats ordered by most recently updated, scoped by user_id if provided."""
    with _managed_connection(conn) as c:
        if user_id:
            rows = c.execute(
                "SELECT * FROM chats WHERE user_id = ? ORDER BY updated_at DESC",
                (user_id,),
            ).fetchall()
        else:
            rows = c.execute(
                "SELECT * FROM chats ORDER BY updated_at DESC"
            ).fetchall()
        return _rows_to_dicts(rows)


def get_chat(
    chat_id: str,
    user_id: Optional[str] = None,
    conn: Optional[Any] = None,
) -> Optional[Dict[str, Any]]:
    """Retrieve a single chat by ID, verifying user_id ownership if provided."""
    with _managed_connection(conn) as c:
        if user_id:
            row = c.execute(
                "SELECT * FROM chats WHERE id = ? AND user_id = ?",
                (chat_id, user_id),
            ).fetchone()
        else:
            row = c.execute("SELECT * FROM chats WHERE id = ?", (chat_id,)).fetchone()
        return _row_to_dict(row)


def rename_chat(
    chat_id: str,
    new_title: str,
    user_id: Optional[str] = None,
    conn: Optional[Any] = None,
) -> Optional[Dict[str, Any]]:
    """Rename an existing chat and update its updated_at timestamp, scoped by user_id if provided."""
    clean_title = new_title.strip()
    if not clean_title:
        raise ValueError("Chat title cannot be empty.")

    now = _now_utc_iso()
    with _managed_connection(conn) as c:
        if user_id:
            cur = c.execute(
                "UPDATE chats SET title = ?, updated_at = ? WHERE id = ? AND user_id = ?",
                (clean_title, now, chat_id, user_id),
            )
            if cur.rowcount == 0:
                return None
            row = c.execute("SELECT * FROM chats WHERE id = ? AND user_id = ?", (chat_id, user_id)).fetchone()
        else:
            cur = c.execute(
                "UPDATE chats SET title = ?, updated_at = ? WHERE id = ?",
                (clean_title, now, chat_id),
            )
            if cur.rowcount == 0:
                return None
            row = c.execute("SELECT * FROM chats WHERE id = ?", (chat_id,)).fetchone()
        return _row_to_dict(row)


def delete_chat(
    chat_id: str,
    user_id: Optional[str] = None,
    conn: Optional[Any] = None,
) -> bool:
    """
    Delete a chat. Cascades automatically to messages and chat_documents links.
    Scopes by user_id if provided.
    """
    with _managed_connection(conn) as c:
        if user_id:
            cur = c.execute("DELETE FROM chats WHERE id = ? AND user_id = ?", (chat_id, user_id))
        else:
            cur = c.execute("DELETE FROM chats WHERE id = ?", (chat_id,))
        return cur.rowcount > 0


# ==============================================================================
# 2. DOCUMENT REPOSITORY
# ==============================================================================

VALID_DOCUMENT_STATUSES = {"pending", "processing", "ready", "failed"}


def create_document(
    filename: str,
    file_hash: str,
    file_size: int,
    user_id: Optional[str] = None,
    page_count: Optional[int] = None,
    chunk_count: Optional[int] = None,
    storage_path: Optional[str] = None,
    status: str = "pending",
    error_message: Optional[str] = None,
    document_id: Optional[str] = None,
    conn: Optional[Any] = None,
) -> Dict[str, Any]:
    """
    Register a document in the documents catalog.
    Enforces (user_id, file_hash) uniqueness per user.
    """
    if status not in VALID_DOCUMENT_STATUSES:
        raise ValueError(f"Invalid status '{status}'. Must be one of {VALID_DOCUMENT_STATUSES}")

    doc_id = document_id or f"doc_{uuid.uuid4().hex[:16]}"
    now = _now_utc_iso()
    uid = user_id or "legacy_user"

    with _managed_connection(conn) as c:
        c.execute(
            """
            INSERT INTO documents (
                id, user_id, filename, file_hash, file_size, page_count,
                chunk_count, storage_path, status, error_message, created_at
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                doc_id,
                uid,
                filename,
                file_hash,
                file_size,
                page_count,
                chunk_count,
                storage_path,
                status,
                error_message,
                now,
            ),
        )
        row = c.execute("SELECT * FROM documents WHERE id = ?", (doc_id,)).fetchone()
        return _row_to_dict(row) or {}


def get_document_by_id(
    document_id: str,
    user_id: Optional[str] = None,
    conn: Optional[Any] = None,
) -> Optional[Dict[str, Any]]:
    """Retrieve document record by document ID, scoped by user_id if provided."""
    with _managed_connection(conn) as c:
        if user_id:
            row = c.execute(
                "SELECT * FROM documents WHERE id = ? AND user_id = ?",
                (document_id, user_id),
            ).fetchone()
        else:
            row = c.execute("SELECT * FROM documents WHERE id = ?", (document_id,)).fetchone()
        return _row_to_dict(row)


def get_document_by_hash(
    file_hash: str,
    user_id: Optional[str] = None,
    conn: Optional[Any] = None,
) -> Optional[Dict[str, Any]]:
    """Retrieve document record by SHA-256 file content hash scoped by user_id."""
    with _managed_connection(conn) as c:
        if user_id:
            row = c.execute(
                "SELECT * FROM documents WHERE file_hash = ? AND user_id = ?",
                (file_hash, user_id),
            ).fetchone()
        else:
            row = c.execute("SELECT * FROM documents WHERE file_hash = ?", (file_hash,)).fetchone()
        return _row_to_dict(row)


def update_document_status(
    document_id: str,
    status: str,
    page_count: Optional[int] = None,
    chunk_count: Optional[int] = None,
    error_message: Optional[str] = None,
    storage_path: Optional[str] = None,
    user_id: Optional[str] = None,
    conn: Optional[Any] = None,
) -> Optional[Dict[str, Any]]:
    """Update processing status, counts, and error details of an indexed document."""
    if status not in VALID_DOCUMENT_STATUSES:
        raise ValueError(f"Invalid status '{status}'. Must be one of {VALID_DOCUMENT_STATUSES}")

    with _managed_connection(conn) as c:
        updates = ["status = ?"]
        params: List[Any] = [status]

        if page_count is not None:
            updates.append("page_count = ?")
            params.append(page_count)
        if chunk_count is not None:
            updates.append("chunk_count = ?")
            params.append(chunk_count)
        if error_message is not None:
            updates.append("error_message = ?")
            params.append(error_message)
        if storage_path is not None:
            updates.append("storage_path = ?")
            params.append(storage_path)

        params.append(document_id)
        if user_id:
            params.append(user_id)
            sql = f"UPDATE documents SET {', '.join(updates)} WHERE id = ? AND user_id = ?"
        else:
            sql = f"UPDATE documents SET {', '.join(updates)} WHERE id = ?"

        cur = c.execute(sql, tuple(params))
        if cur.rowcount == 0:
            return None

        if user_id:
            row = c.execute("SELECT * FROM documents WHERE id = ? AND user_id = ?", (document_id, user_id)).fetchone()
        else:
            row = c.execute("SELECT * FROM documents WHERE id = ?", (document_id,)).fetchone()
        return _row_to_dict(row)


def list_documents(
    user_id: Optional[str] = None,
    conn: Optional[Any] = None,
) -> List[Dict[str, Any]]:
    """List all documents in the catalog ordered by creation date, scoped by user_id if provided."""
    with _managed_connection(conn) as c:
        if user_id:
            rows = c.execute(
                "SELECT * FROM documents WHERE user_id = ? ORDER BY created_at DESC",
                (user_id,),
            ).fetchall()
        else:
            rows = c.execute("SELECT * FROM documents ORDER BY created_at DESC").fetchall()
        return _rows_to_dicts(rows)


def delete_document(
    document_id: str,
    user_id: Optional[str] = None,
    conn: Optional[Any] = None,
) -> bool:
    """Delete a document record scoped by user_id if provided."""
    with _managed_connection(conn) as c:
        if user_id:
            cur = c.execute("DELETE FROM documents WHERE id = ? AND user_id = ?", (document_id, user_id))
        else:
            cur = c.execute("DELETE FROM documents WHERE id = ?", (document_id,))
        return cur.rowcount > 0


# ==============================================================================
# 3. CHAT-DOCUMENT RELATIONSHIPS
# ==============================================================================

def attach_document_to_chat(
    chat_id: str,
    document_id: str,
    user_id: Optional[str] = None,
    conn: Optional[Any] = None,
) -> Dict[str, Any]:
    """
    Associate an existing document with a chat.
    Validates ownership of both chat and document when user_id is provided.
    Idempotent: Re-attaching an already attached document returns existing relationship.
    """
    now = _now_utc_iso()
    with _managed_connection(conn) as c:
        if user_id:
            chat_row = c.execute(
                "SELECT id FROM chats WHERE id = ? AND user_id = ?",
                (chat_id, user_id),
            ).fetchone()
            if not chat_row:
                raise ValueError(f"Chat '{chat_id}' does not exist or unauthorized.")

            doc_row = c.execute(
                "SELECT id FROM documents WHERE id = ? AND user_id = ?",
                (document_id, user_id),
            ).fetchone()
            if not doc_row:
                raise ValueError(f"Document '{document_id}' does not exist or unauthorized.")
        else:
            chat_row = c.execute("SELECT id FROM chats WHERE id = ?", (chat_id,)).fetchone()
            if not chat_row:
                raise ValueError(f"Chat '{chat_id}' does not exist.")

            doc_row = c.execute("SELECT id FROM documents WHERE id = ?", (document_id,)).fetchone()
            if not doc_row:
                raise ValueError(f"Document '{document_id}' does not exist.")

        c.execute(
            """
            INSERT OR IGNORE INTO chat_documents (chat_id, document_id, created_at)
            VALUES (?, ?, ?)
            """,
            (chat_id, document_id, now),
        )
        c.execute("UPDATE chats SET updated_at = ? WHERE id = ?", (now, chat_id))

        row = c.execute(
            "SELECT * FROM chat_documents WHERE chat_id = ? AND document_id = ?",
            (chat_id, document_id),
        ).fetchone()
        return _row_to_dict(row) or {}


def detach_document_from_chat(
    chat_id: str,
    document_id: str,
    user_id: Optional[str] = None,
    conn: Optional[Any] = None,
) -> bool:
    """
    Remove a document from a chat's scope.
    Validates chat ownership if user_id is provided.
    Does NOT delete the document from the user catalog or ChromaDB.
    """
    now = _now_utc_iso()
    with _managed_connection(conn) as c:
        if user_id:
            chat_row = c.execute("SELECT id FROM chats WHERE id = ? AND user_id = ?", (chat_id, user_id)).fetchone()
            if not chat_row:
                return False

        cur = c.execute(
            "DELETE FROM chat_documents WHERE chat_id = ? AND document_id = ?",
            (chat_id, document_id),
        )
        if cur.rowcount > 0:
            c.execute("UPDATE chats SET updated_at = ? WHERE id = ?", (now, chat_id))
            return True
        return False


def list_chat_documents(
    chat_id: str,
    user_id: Optional[str] = None,
    conn: Optional[Any] = None,
) -> List[Dict[str, Any]]:
    """
    List all documents attached to a specific chat, including document metadata.
    Validates chat ownership if user_id is provided.
    """
    with _managed_connection(conn) as c:
        if user_id:
            chat_row = c.execute("SELECT id FROM chats WHERE id = ? AND user_id = ?", (chat_id, user_id)).fetchone()
            if not chat_row:
                return []

        rows = c.execute(
            """
            SELECT 
                d.id,
                d.user_id,
                d.filename,
                d.file_hash,
                d.file_size,
                d.page_count,
                d.chunk_count,
                d.storage_path,
                d.status,
                d.error_message,
                d.created_at,
                cd.created_at as attached_at
            FROM chat_documents cd
            JOIN documents d ON cd.document_id = d.id
            WHERE cd.chat_id = ?
            ORDER BY cd.created_at ASC
            """,
            (chat_id,),
        ).fetchall()
        return _rows_to_dicts(rows)


def document_attached_to_chat(
    chat_id: str,
    document_id: str,
    conn: Optional[Any] = None,
) -> bool:
    """Check if a specific document is attached to a chat."""
    with _managed_connection(conn) as c:
        row = c.execute(
            "SELECT 1 FROM chat_documents WHERE chat_id = ? AND document_id = ?",
            (chat_id, document_id),
        ).fetchone()
        return row is not None


# ==============================================================================
# 4. MESSAGES REPOSITORY
# ==============================================================================

VALID_MESSAGE_ROLES = {"user", "assistant", "system"}


def create_message(
    chat_id: str,
    role: str,
    content: str,
    sources_json: Optional[str | List[Dict[str, Any]]] = None,
    message_id: Optional[str] = None,
    user_id: Optional[str] = None,
    conn: Optional[Any] = None,
) -> Dict[str, Any]:
    """
    Create and persist a chat message.
    Updates the parent chat's updated_at timestamp.
    Validates chat ownership when user_id is provided.
    """
    if role not in VALID_MESSAGE_ROLES:
        raise ValueError(f"Invalid message role '{role}'. Must be one of {VALID_MESSAGE_ROLES}")

    msg_id = message_id or f"msg_{uuid.uuid4().hex[:16]}"
    now = _now_utc_iso()

    serialized_sources = None
    if sources_json is not None:
        if isinstance(sources_json, str):
            serialized_sources = sources_json
        else:
            serialized_sources = json.dumps(sources_json)

    with _managed_connection(conn) as c:
        if user_id:
            chat = c.execute("SELECT id FROM chats WHERE id = ? AND user_id = ?", (chat_id, user_id)).fetchone()
            if not chat:
                raise ValueError(f"Chat '{chat_id}' does not exist or unauthorized.")
        else:
            chat = c.execute("SELECT id FROM chats WHERE id = ?", (chat_id,)).fetchone()
            if not chat:
                raise ValueError(f"Chat '{chat_id}' does not exist.")

        c.execute(
            """
            INSERT INTO messages (id, chat_id, role, content, sources_json, created_at)
            VALUES (?, ?, ?, ?, ?, ?)
            """,
            (msg_id, chat_id, role, content, serialized_sources, now),
        )
        c.execute("UPDATE chats SET updated_at = ? WHERE id = ?", (now, chat_id))

        row = c.execute("SELECT * FROM messages WHERE id = ?", (msg_id,)).fetchone()
        return _row_to_dict(row) or {}


def list_messages(
    chat_id: str,
    user_id: Optional[str] = None,
    conn: Optional[Any] = None,
) -> List[Dict[str, Any]]:
    """Retrieve all messages for a specific chat ordered chronologically, validating ownership if user_id is provided."""
    with _managed_connection(conn) as c:
        if user_id:
            chat = c.execute("SELECT id FROM chats WHERE id = ? AND user_id = ?", (chat_id, user_id)).fetchone()
            if not chat:
                return []

        rows = c.execute(
            "SELECT * FROM messages WHERE chat_id = ? ORDER BY created_at ASC",
            (chat_id,),
        ).fetchall()
        return _rows_to_dicts(rows)
