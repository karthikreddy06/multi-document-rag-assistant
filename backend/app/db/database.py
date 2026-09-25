"""
Dual Database Engine Manager: PostgreSQL (Supabase) and SQLite.
Configures connection pooling, schema initialization, and safe connection lifecycle.
"""

from pathlib import Path
import sqlite3
import threading
from typing import Any, Optional
import urllib.parse

from app.config import settings
from app.utils.logger import setup_logger

try:
    import psycopg
    from psycopg.rows import dict_row
    from psycopg_pool import ConnectionPool
    PSYCOPG_AVAILABLE = True
except ImportError:
    PSYCOPG_AVAILABLE = False
    dict_row = None
    ConnectionPool = None

logger = setup_logger("db.database")

# Global pool singleton and thread lock for PostgreSQL
_pg_pool: Optional[Any] = None
_pg_pool_lock = threading.Lock()

SCHEMA_SQL = """
CREATE TABLE IF NOT EXISTS users (
    id TEXT PRIMARY KEY,
    email TEXT NOT NULL UNIQUE COLLATE NOCASE,
    password_hash TEXT NOT NULL,
    created_at TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS chats (
    id TEXT PRIMARY KEY,
    user_id TEXT REFERENCES users(id) ON DELETE CASCADE,
    title TEXT NOT NULL,
    created_at TEXT NOT NULL,
    updated_at TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS documents (
    id TEXT PRIMARY KEY,
    user_id TEXT REFERENCES users(id) ON DELETE CASCADE,
    filename TEXT NOT NULL,
    file_hash TEXT NOT NULL,
    file_size INTEGER NOT NULL,
    page_count INTEGER,
    chunk_count INTEGER,
    storage_path TEXT,
    status TEXT NOT NULL,
    error_message TEXT,
    created_at TEXT NOT NULL,
    UNIQUE(user_id, file_hash)
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

CREATE INDEX IF NOT EXISTS idx_users_email ON users(email);
CREATE INDEX IF NOT EXISTS idx_chats_updated_at ON chats(updated_at);
CREATE INDEX IF NOT EXISTS idx_documents_file_hash ON documents(file_hash);
CREATE INDEX IF NOT EXISTS idx_chat_documents_chat_id ON chat_documents(chat_id);
CREATE INDEX IF NOT EXISTS idx_chat_documents_document_id ON chat_documents(document_id);
CREATE INDEX IF NOT EXISTS idx_messages_chat_id ON messages(chat_id);
CREATE INDEX IF NOT EXISTS idx_messages_created_at ON messages(created_at);
"""


def normalize_database_url(url: Optional[str]) -> Optional[str]:
    """
    Safely normalize database URL if the password contains unencoded '@' or special characters.
    In standard URIs, password characters like '@' must be percent-encoded (%40).
    """
    if not url:
        return url
    if url.count("@") > 1 and "://" in url:
        prefix, rest = url.split("://", 1)
        last_at_idx = rest.rfind("@")
        credentials = rest[:last_at_idx]
        host_and_beyond = rest[last_at_idx + 1:]
        if ":" in credentials:
            user, password = credentials.split(":", 1)
            encoded_password = urllib.parse.quote(password, safe="")
            return f"{prefix}://{user}:{encoded_password}@{host_and_beyond}"
    return url


_UNSET = object()


def get_safe_db_info(url: Any = _UNSET) -> str:
    """
    Return sanitized database connection info without username, password, or secrets.
    Example: 'PostgreSQL (host: ldytwnvxskfajjwcxxtb.supabase.co:5432, db: postgres)'
    """
    raw_url = settings.database_url if url is _UNSET else url
    target_url = normalize_database_url(raw_url)
    if not target_url:
        return f"SQLite (local file: {get_db_path()})"
    try:
        parsed = urllib.parse.urlsplit(target_url)
        host = parsed.hostname or "unknown-host"
        port = f":{parsed.port}" if parsed.port else ":5432"
        dbname = parsed.path.lstrip("/") or "postgres"
        return f"PostgreSQL (host: {host}{port}, db: {dbname})"
    except Exception:
        return "PostgreSQL (credentials masked)"


def get_db_path(custom_path: Optional[str | Path] = None) -> Path:
    """
    Return resolved database file path, ensuring parent directories exist.
    If custom_path or settings.database_path points to a directory (or ends with a slash),
    automatically appends 'rag_app.db' to the target directory.
    """
    raw = custom_path or settings.database_path
    path = Path(raw).resolve()
    if path.is_dir() or str(raw).endswith(("/", "\\")):
        path = (path / "rag_app.db").resolve()
    path.parent.mkdir(parents=True, exist_ok=True)
    return path


def get_pg_pool(database_url: Optional[str] = None) -> Any:
    """
    Return active PostgreSQL connection pool (singleton).
    Thread-safe lazy initialization using psycopg_pool.ConnectionPool with dict_row factory.
    """
    global _pg_pool
    if not PSYCOPG_AVAILABLE or ConnectionPool is None:
        raise RuntimeError("psycopg and psycopg_pool must be installed to use PostgreSQL backend.")

    with _pg_pool_lock:
        if _pg_pool is None:
            raw_url = database_url or settings.database_url
            url = normalize_database_url(raw_url)
            if not url:
                raise RuntimeError("Cannot initialize PostgreSQL pool: DATABASE_URL is not set.")

            safe_info = get_safe_db_info(url)
            logger.info(f"Database pool: Initializing PostgreSQL ConnectionPool for {safe_info} (min=1, max=10)")
            _pg_pool = ConnectionPool(
                conninfo=url,
                min_size=1,
                max_size=10,
                timeout=30.0,
                kwargs={"row_factory": dict_row},
                open=True,
            )
        return _pg_pool


def close_db() -> None:
    """Safely close and clean up PostgreSQL connection pool on application shutdown."""
    global _pg_pool
    with _pg_pool_lock:
        if _pg_pool is not None:
            try:
                _pg_pool.close()
                logger.info("Database pool: PostgreSQL connection pool closed safely.")
            except Exception as e:
                logger.warning(f"Error closing PostgreSQL pool: {e}")
            _pg_pool = None


def get_db_connection(custom_path: Optional[str | Path] = None) -> Any:
    """
    Open/borrow a database connection.
    If custom_path is provided or settings.is_postgres is False, returns an SQLite connection
    configured with 30s timeout, dict-like sqlite3.Row factory, WAL mode, and foreign keys enabled.
    If settings.is_postgres is True and custom_path is None, returns a pooled PostgreSQL connection.
    """
    if custom_path is not None or not settings.is_postgres:
        db_path = get_db_path(custom_path)
        conn = sqlite3.connect(
            str(db_path),
            timeout=30.0,
            detect_types=sqlite3.PARSE_DECLTYPES | sqlite3.PARSE_COLNAMES,
        )
        conn.row_factory = sqlite3.Row
        conn.execute("PRAGMA foreign_keys = ON;")
        try:
            conn.execute("PRAGMA journal_mode = WAL;")
        except sqlite3.OperationalError:
            pass
        return conn
    else:
        pool = get_pg_pool()
        return pool.getconn()


def _init_sqlite_db(custom_path: Optional[str | Path] = None) -> None:
    """Initialize SQLite database schema, tables, indexes, and apply migrations idempotently."""
    db_path = get_db_path(custom_path)
    is_existing = db_path.exists() and db_path.stat().st_size > 0
    if is_existing:
        logger.info(
            f"Database persistence: Connecting to existing SQLite database at '{db_path}' "
            f"({db_path.stat().st_size} bytes)"
        )
    else:
        logger.info(f"Database persistence: Initializing new SQLite database file at '{db_path}'")

    conn = get_db_connection(custom_path)
    try:
        conn.execute("PRAGMA foreign_keys = OFF;")
        with conn:
            # 1. Ensure basic schema is created
            conn.executescript(SCHEMA_SQL)

            # 2. Check and perform safe migrations for existing tables if needed
            conn.execute(
                """
                INSERT OR IGNORE INTO users (id, email, password_hash, created_at)
                VALUES ('legacy_user', 'legacy@local.dev', '', '2026-01-01T00:00:00Z')
                """
            )

            # Check if 'user_id' exists in 'chats'
            chat_cols = [c[1] for c in conn.execute("PRAGMA table_info(chats)").fetchall()]
            if "user_id" not in chat_cols:
                conn.execute("ALTER TABLE chats ADD COLUMN user_id TEXT REFERENCES users(id) ON DELETE CASCADE")
                conn.execute("UPDATE chats SET user_id = 'legacy_user' WHERE user_id IS NULL")

            # Check if 'user_id' exists in 'documents'
            doc_cols = [c[1] for c in conn.execute("PRAGMA table_info(documents)").fetchall()]
            if "user_id" not in doc_cols:
                conn.execute(
                    """
                    CREATE TABLE IF NOT EXISTS documents_migrated (
                        id TEXT PRIMARY KEY,
                        user_id TEXT NOT NULL REFERENCES users(id) ON DELETE CASCADE,
                        filename TEXT NOT NULL,
                        file_hash TEXT NOT NULL,
                        file_size INTEGER NOT NULL,
                        page_count INTEGER,
                        chunk_count INTEGER,
                        storage_path TEXT,
                        status TEXT NOT NULL,
                        error_message TEXT,
                        created_at TEXT NOT NULL,
                        UNIQUE(user_id, file_hash)
                    )
                    """
                )
                conn.execute(
                    """
                    INSERT INTO documents_migrated (
                        id, user_id, filename, file_hash, file_size, page_count,
                        chunk_count, storage_path, status, error_message, created_at
                    )
                    SELECT id, 'legacy_user', filename, file_hash, file_size, page_count,
                           chunk_count, storage_path, status, error_message, created_at
                    FROM documents
                    """
                )
                conn.execute("DROP TABLE documents")
                conn.execute("ALTER TABLE documents_migrated RENAME TO documents")

            # Establish indexes
            conn.execute("CREATE INDEX IF NOT EXISTS idx_users_email ON users(email)")
            conn.execute("CREATE INDEX IF NOT EXISTS idx_chats_user_id ON chats(user_id)")
            conn.execute("CREATE INDEX IF NOT EXISTS idx_documents_user_id ON documents(user_id)")
            conn.execute("CREATE INDEX IF NOT EXISTS idx_documents_file_hash ON documents(file_hash)")

            user_count = conn.execute("SELECT COUNT(*) FROM users").fetchone()[0]
            chat_count = conn.execute("SELECT COUNT(*) FROM chats").fetchone()[0]
            doc_count = conn.execute("SELECT COUNT(*) FROM documents").fetchone()[0]

        logger.info(
            f"Database ready: SQLite active at '{db_path}' ({user_count} users, {chat_count} chats, {doc_count} documents present)."
        )
    finally:
        conn.close()


def _init_postgres_db() -> None:
    """
    Verify and idempotently initialize PostgreSQL schema.
    Verifies users, chats, documents, chat_documents, messages tables.
    Does NOT seed legacy_user and does NOT migrate existing SQLite data.
    """
    safe_info = get_safe_db_info()
    logger.info(f"Database persistence: Connecting to PostgreSQL ({safe_info})")

    pool = get_pg_pool()
    with pool.connection() as conn:
        with conn.cursor() as cur:
            # 1. Enable CITEXT extension if possible
            try:
                cur.execute("CREATE EXTENSION IF NOT EXISTS citext;")
            except Exception as e:
                logger.warning(f"Note on citext extension creation: {e}")

            # 2. Check existing tables in public schema
            cur.execute(
                """
                SELECT table_name FROM information_schema.tables 
                WHERE table_schema = 'public' 
                  AND table_name IN ('users', 'chats', 'documents', 'chat_documents', 'messages');
                """
            )
            existing_tables = {row["table_name"] for row in cur.fetchall()}
            required_tables = {"users", "chats", "documents", "chat_documents", "messages"}
            missing_tables = required_tables - existing_tables

            if missing_tables:
                logger.info(f"PostgreSQL schema missing tables {missing_tables}. Applying DDL idempotently...")
                schema_path = Path(__file__).resolve().parent.parent.parent / "sql" / "supabase_schema.sql"
                if schema_path.exists():
                    ddl_script = schema_path.read_text(encoding="utf-8")
                    cur.execute(ddl_script)
                else:
                    cur.execute(
                        """
                        CREATE TABLE IF NOT EXISTS users (
                            id TEXT PRIMARY KEY,
                            email CITEXT NOT NULL UNIQUE,
                            password_hash TEXT NOT NULL,
                            created_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
                        );
                        CREATE TABLE IF NOT EXISTS chats (
                            id TEXT PRIMARY KEY,
                            user_id TEXT NOT NULL REFERENCES users(id) ON DELETE CASCADE,
                            title TEXT NOT NULL DEFAULT 'New Chat',
                            created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
                            updated_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
                        );
                        CREATE TABLE IF NOT EXISTS documents (
                            id TEXT PRIMARY KEY,
                            user_id TEXT NOT NULL REFERENCES users(id) ON DELETE CASCADE,
                            filename TEXT NOT NULL,
                            file_hash TEXT NOT NULL,
                            file_size BIGINT NOT NULL,
                            page_count INTEGER,
                            chunk_count INTEGER,
                            storage_path TEXT,
                            status TEXT NOT NULL CHECK (status IN ('pending', 'processing', 'ready', 'failed')),
                            error_message TEXT,
                            created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
                            CONSTRAINT uq_user_document_hash UNIQUE (user_id, file_hash)
                        );
                        CREATE TABLE IF NOT EXISTS chat_documents (
                            chat_id TEXT NOT NULL REFERENCES chats(id) ON DELETE CASCADE,
                            document_id TEXT NOT NULL REFERENCES documents(id) ON DELETE CASCADE,
                            created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
                            PRIMARY KEY (chat_id, document_id)
                        );
                        CREATE TABLE IF NOT EXISTS messages (
                            id TEXT PRIMARY KEY,
                            chat_id TEXT NOT NULL REFERENCES chats(id) ON DELETE CASCADE,
                            role TEXT NOT NULL CHECK (role IN ('user', 'assistant', 'system')),
                            content TEXT NOT NULL,
                            sources_json JSONB,
                            created_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
                        );
                        CREATE INDEX IF NOT EXISTS idx_chats_user_id ON chats(user_id);
                        CREATE INDEX IF NOT EXISTS idx_chats_updated_at ON chats(updated_at DESC);
                        CREATE INDEX IF NOT EXISTS idx_documents_user_id ON documents(user_id);
                        CREATE INDEX IF NOT EXISTS idx_documents_file_hash ON documents(file_hash);
                        CREATE INDEX IF NOT EXISTS idx_chat_documents_chat_id ON chat_documents(chat_id);
                        CREATE INDEX IF NOT EXISTS idx_chat_documents_document_id ON chat_documents(document_id);
                        CREATE INDEX IF NOT EXISTS idx_messages_chat_id ON messages(chat_id);
                        CREATE INDEX IF NOT EXISTS idx_messages_created_at ON messages(created_at ASC);
                        """
                    )
                logger.info(f"PostgreSQL tables initialized successfully: {missing_tables}")
            else:
                logger.info("Database verification: All required PostgreSQL tables verified (users, chats, documents, chat_documents, messages).")

            # 3. Safe diagnostic counts
            cur.execute("SELECT COUNT(*) AS count FROM users")
            user_count = cur.fetchone()["count"]
            cur.execute("SELECT COUNT(*) AS count FROM chats")
            chat_count = cur.fetchone()["count"]
            cur.execute("SELECT COUNT(*) AS count FROM documents")
            doc_count = cur.fetchone()["count"]

    logger.info(
        f"Database ready: PostgreSQL active ({safe_info}) - "
        f"{user_count} users, {chat_count} chats, {doc_count} documents present."
    )


def init_db(custom_path: Optional[str | Path] = None) -> None:
    """
    Initialize database schema idempotently.
    Routes to PostgreSQL when settings.is_postgres is True and custom_path is None.
    Routes to SQLite otherwise.
    """
    if custom_path is None and settings.is_postgres:
        _init_postgres_db()
    else:
        _init_sqlite_db(custom_path)
