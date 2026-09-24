"""
Database Repository Module.
Provides CRUD operations for chats, documents, chat-document links, and messages.
All database access uses parameterized SQL and consistent UTC ISO-8601 timestamps.
"""

from contextlib import contextmanager
from datetime import datetime, timezone
import json
from pathlib import Path
import sqlite3
from typing import Any, Dict, Generator, List, Optional
import uuid

from app.db.database import get_db_connection


def _now_utc_iso() -> str:
    """Return current UTC timestamp in ISO 8601 format."""
    return datetime.now(timezone.utc).isoformat()


def _row_to_dict(row: Optional[sqlite3.Row]) -> Optional[Dict[str, Any]]:
    """Convert an sqlite3.Row object to a standard Python dictionary."""
    if row is None:
        return None
    return dict(row)


def _rows_to_dicts(rows: List[sqlite3.Row]) -> List[Dict[str, Any]]:
    """Convert a list of sqlite3.Row objects to standard Python dictionaries."""
    return [dict(r) for r in rows]


@contextmanager
def _managed_connection(
    conn: Optional[sqlite3.Connection] = None,
    custom_path: Optional[str | Path] = None,
) -> Generator[sqlite3.Connection, None, None]:
    """
    Yield an active SQLite connection.
    If a connection is provided, reuse it without closing; otherwise open, commit, and close.
    """
    if conn is not None:
        yield conn
    else:
        new_conn = get_db_connection(custom_path)
        try:
            yield new_conn
            new_conn.commit()
        finally:
            new_conn.close()


# ==============================================================================
# 0. USER REPOSITORY
# ==============================================================================

def create_user(
    email: str,
    password_hash: str,
    user_id: Optional[str] = None,
    conn: Optional[sqlite3.Connection] = None,
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
    conn: Optional[sqlite3.Connection] = None,
) -> Optional[Dict[str, Any]]:
    """Retrieve user record by email address (case-insensitive)."""
    clean_email = email.strip().lower()
    with _managed_connection(conn) as c:
        row = c.execute("SELECT * FROM users WHERE email = ? COLLATE NOCASE", (clean_email,)).fetchone()
        return _row_to_dict(row)


def get_user_by_id(
    user_id: str,
    conn: Optional[sqlite3.Connection] = None,
) -> Optional[Dict[str, Any]]:
    """Retrieve user record by user ID."""
    with _managed_connection(conn) as c:
        row = c.execute("SELECT * FROM users WHERE id = ?", (user_id,)).fetchone()
        return _row_to_dict(row)


def list_users(conn: Optional[sqlite3.Connection] = None) -> List[Dict[str, Any]]:
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
    conn: Optional[sqlite3.Connection] = None,
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
    conn: Optional[sqlite3.Connection] = None,
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
    conn: Optional[sqlite3.Connection] = None,
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
    conn: Optional[sqlite3.Connection] = None,
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
    conn: Optional[sqlite3.Connection] = None,
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
    conn: Optional[sqlite3.Connection] = None,
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
    conn: Optional[sqlite3.Connection] = None,
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
    conn: Optional[sqlite3.Connection] = None,
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
    conn: Optional[sqlite3.Connection] = None,
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
    conn: Optional[sqlite3.Connection] = None,
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
    conn: Optional[sqlite3.Connection] = None,
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
    conn: Optional[sqlite3.Connection] = None,
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
    conn: Optional[sqlite3.Connection] = None,
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
    conn: Optional[sqlite3.Connection] = None,
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
    conn: Optional[sqlite3.Connection] = None,
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
    conn: Optional[sqlite3.Connection] = None,
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
    conn: Optional[sqlite3.Connection] = None,
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

