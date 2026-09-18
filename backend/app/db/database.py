"""
SQLite Database Connection and Schema Management.
Configures connection pooling, WAL mode, foreign keys, and tables.
"""

from pathlib import Path
import sqlite3
from typing import Optional

from app.config import settings
from app.utils.logger import setup_logger

logger = setup_logger("db.database")

SCHEMA_SQL = """
CREATE TABLE IF NOT EXISTS chats (
    id TEXT PRIMARY KEY,
    title TEXT NOT NULL,
    created_at TEXT NOT NULL,
    updated_at TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS documents (
    id TEXT PRIMARY KEY,
    filename TEXT NOT NULL,
    file_hash TEXT NOT NULL UNIQUE,
    file_size INTEGER NOT NULL,
    page_count INTEGER,
    chunk_count INTEGER,
    storage_path TEXT,
    status TEXT NOT NULL,
    error_message TEXT,
    created_at TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS chat_documents (
    chat_id TEXT NOT NULL,
    document_id TEXT NOT NULL,
    created_at TEXT NOT NULL,
    PRIMARY KEY (chat_id, document_id),
    FOREIGN KEY (chat_id) REFERENCES chats(id) ON DELETE CASCADE,
    FOREIGN KEY (document_id) REFERENCES documents(id) ON DELETE CASCADE
);

CREATE TABLE IF NOT EXISTS messages (
    id TEXT PRIMARY KEY,
    chat_id TEXT NOT NULL,
    role TEXT NOT NULL,
    content TEXT NOT NULL,
    sources_json TEXT,
    created_at TEXT NOT NULL,
    FOREIGN KEY (chat_id) REFERENCES chats(id) ON DELETE CASCADE
);

CREATE INDEX IF NOT EXISTS idx_chats_updated_at ON chats(updated_at);
CREATE INDEX IF NOT EXISTS idx_documents_file_hash ON documents(file_hash);
CREATE INDEX IF NOT EXISTS idx_chat_documents_chat_id ON chat_documents(chat_id);
CREATE INDEX IF NOT EXISTS idx_chat_documents_document_id ON chat_documents(document_id);
CREATE INDEX IF NOT EXISTS idx_messages_chat_id ON messages(chat_id);
CREATE INDEX IF NOT EXISTS idx_messages_created_at ON messages(created_at);
"""


def get_db_path(custom_path: Optional[str | Path] = None) -> Path:
    """Return resolved database file path, ensuring parent directories exist."""
    path = Path(custom_path or settings.database_path).resolve()
    path.parent.mkdir(parents=True, exist_ok=True)
    return path


def get_db_connection(custom_path: Optional[str | Path] = None) -> sqlite3.Connection:
    """
    Open a connection to the SQLite database with 30s timeout, row factory,
    WAL mode, and foreign keys enabled.
    """
    db_path = get_db_path(custom_path)
    conn = sqlite3.connect(
        str(db_path),
        timeout=30.0,
        detect_types=sqlite3.PARSE_DECLTYPES | sqlite3.PARSE_COLNAMES,
    )
    conn.row_factory = sqlite3.Row

    # Enforce foreign key constraints
    conn.execute("PRAGMA foreign_keys = ON;")

    # Enable WAL mode for high concurrency
    # (Memory databases do not support WAL mode, so catch safely for memory tests)
    try:
        conn.execute("PRAGMA journal_mode = WAL;")
    except sqlite3.OperationalError:
        pass

    return conn


def init_db(custom_path: Optional[str | Path] = None) -> None:
    """Initialize database schema, tables, and indexes idempotently."""
    conn = get_db_connection(custom_path)
    try:
        with conn:
            conn.executescript(SCHEMA_SQL)
        logger.info(f"Database initialized successfully at {get_db_path(custom_path)}")
    finally:
        conn.close()
