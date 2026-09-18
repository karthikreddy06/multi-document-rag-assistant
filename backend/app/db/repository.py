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
# 1. CHAT REPOSITORY
# ==============================================================================

def create_chat(
    title: str = "New Chat",
    chat_id: Optional[str] = None,
    conn: Optional[sqlite3.Connection] = None,
) -> Dict[str, Any]:
    """Create a new chat session."""
    cid = chat_id or f"chat_{uuid.uuid4().hex[:16]}"
    now = _now_utc_iso()
    clean_title = title.strip() or "New Chat"

    with _managed_connection(conn) as c:
        c.execute(
            "INSERT INTO chats (id, title, created_at, updated_at) VALUES (?, ?, ?, ?)",
            (cid, clean_title, now, now),
        )
        row = c.execute("SELECT * FROM chats WHERE id = ?", (cid,)).fetchone()
        return _row_to_dict(row) or {}


def list_chats(conn: Optional[sqlite3.Connection] = None) -> List[Dict[str, Any]]:
    """List all chats ordered by most recently updated."""
    with _managed_connection(conn) as c:
        rows = c.execute(
            "SELECT * FROM chats ORDER BY updated_at DESC"
        ).fetchall()
        return _rows_to_dicts(rows)


def get_chat(
    chat_id: str,
    conn: Optional[sqlite3.Connection] = None,
) -> Optional[Dict[str, Any]]:
    """Retrieve a single chat by ID."""
    with _managed_connection(conn) as c:
        row = c.execute("SELECT * FROM chats WHERE id = ?", (chat_id,)).fetchone()
        return _row_to_dict(row)


def rename_chat(
    chat_id: str,
    new_title: str,
    conn: Optional[sqlite3.Connection] = None,
) -> Optional[Dict[str, Any]]:
    """Rename an existing chat and update its updated_at timestamp."""
    clean_title = new_title.strip()
    if not clean_title:
        raise ValueError("Chat title cannot be empty.")

    now = _now_utc_iso()
    with _managed_connection(conn) as c:
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
    conn: Optional[sqlite3.Connection] = None,
) -> bool:
    """
    Delete a chat. Cascades automatically to messages and chat_documents links.
    Does NOT delete the global documents.
    """
    with _managed_connection(conn) as c:
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
    Enforces file_hash uniqueness.
    """
    if status not in VALID_DOCUMENT_STATUSES:
        raise ValueError(f"Invalid status '{status}'. Must be one of {VALID_DOCUMENT_STATUSES}")

    doc_id = document_id or f"doc_{uuid.uuid4().hex[:16]}"
    now = _now_utc_iso()

    with _managed_connection(conn) as c:
        c.execute(
            """
            INSERT INTO documents (
                id, filename, file_hash, file_size, page_count,
                chunk_count, storage_path, status, error_message, created_at
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                doc_id,
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
    conn: Optional[sqlite3.Connection] = None,
) -> Optional[Dict[str, Any]]:
    """Retrieve document record by document ID."""
    with _managed_connection(conn) as c:
        row = c.execute("SELECT * FROM documents WHERE id = ?", (document_id,)).fetchone()
        return _row_to_dict(row)


def get_document_by_hash(
    file_hash: str,
    conn: Optional[sqlite3.Connection] = None,
) -> Optional[Dict[str, Any]]:
    """Retrieve document record by SHA-256 file content hash (deduplication lookup)."""
    with _managed_connection(conn) as c:
        row = c.execute("SELECT * FROM documents WHERE file_hash = ?", (file_hash,)).fetchone()
        return _row_to_dict(row)


def update_document_status(
    document_id: str,
    status: str,
    page_count: Optional[int] = None,
    chunk_count: Optional[int] = None,
    error_message: Optional[str] = None,
    storage_path: Optional[str] = None,
    conn: Optional[sqlite3.Connection] = None,
) -> Optional[Dict[str, Any]]:
    """Update processing status, counts, and error details of an indexed document."""
    if status not in VALID_DOCUMENT_STATUSES:
        raise ValueError(f"Invalid status '{status}'. Must be one of {VALID_DOCUMENT_STATUSES}")

    with _managed_connection(conn) as c:
        # Build dynamic update statement safely with parameters
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
        sql = f"UPDATE documents SET {', '.join(updates)} WHERE id = ?"
        cur = c.execute(sql, tuple(params))
        if cur.rowcount == 0:
            return None

        row = c.execute("SELECT * FROM documents WHERE id = ?", (document_id,)).fetchone()
        return _row_to_dict(row)


def list_documents(conn: Optional[sqlite3.Connection] = None) -> List[Dict[str, Any]]:
    """List all documents in the catalog ordered by creation date."""
    with _managed_connection(conn) as c:
        rows = c.execute("SELECT * FROM documents ORDER BY created_at DESC").fetchall()
        return _rows_to_dicts(rows)


# ==============================================================================
# 3. CHAT-DOCUMENT RELATIONSHIPS
# ==============================================================================

def attach_document_to_chat(
    chat_id: str,
    document_id: str,
    conn: Optional[sqlite3.Connection] = None,
) -> Dict[str, Any]:
    """
    Associate an existing document with a chat.
    Idempotent: Re-attaching an already attached document returns existing relationship.
    """
    now = _now_utc_iso()
    with _managed_connection(conn) as c:
        # Verify both chat and document exist
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
        # Update chat updated_at
        c.execute("UPDATE chats SET updated_at = ? WHERE id = ?", (now, chat_id))

        row = c.execute(
            "SELECT * FROM chat_documents WHERE chat_id = ? AND document_id = ?",
            (chat_id, document_id),
        ).fetchone()
        return _row_to_dict(row) or {}


def detach_document_from_chat(
    chat_id: str,
    document_id: str,
    conn: Optional[sqlite3.Connection] = None,
) -> bool:
    """
    Remove a document from a chat's scope.
    Does NOT delete the document from the global catalog or ChromaDB.
    """
    now = _now_utc_iso()
    with _managed_connection(conn) as c:
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
    conn: Optional[sqlite3.Connection] = None,
) -> List[Dict[str, Any]]:
    """
    List all documents attached to a specific chat, including document metadata.
    """
    with _managed_connection(conn) as c:
        rows = c.execute(
            """
            SELECT 
                d.id,
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
    conn: Optional[sqlite3.Connection] = None,
) -> Dict[str, Any]:
    """
    Create and persist a chat message.
    Updates the parent chat's updated_at timestamp.
    """
    if role not in VALID_MESSAGE_ROLES:
        raise ValueError(f"Invalid message role '{role}'. Must be one of {VALID_MESSAGE_ROLES}")

    msg_id = message_id or f"msg_{uuid.uuid4().hex[:16]}"
    now = _now_utc_iso()

    # Normalize sources to a valid JSON string if list provided
    serialized_sources = None
    if sources_json is not None:
        if isinstance(sources_json, str):
            serialized_sources = sources_json
        else:
            serialized_sources = json.dumps(sources_json)

    with _managed_connection(conn) as c:
        # Check chat exists to provide clean foreign key error
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
    conn: Optional[sqlite3.Connection] = None,
) -> List[Dict[str, Any]]:
    """Retrieve all messages for a specific chat ordered chronologically."""
    with _managed_connection(conn) as c:
        rows = c.execute(
            "SELECT * FROM messages WHERE chat_id = ? ORDER BY created_at ASC",
            (chat_id,),
        ).fetchall()
        return _rows_to_dicts(rows)
