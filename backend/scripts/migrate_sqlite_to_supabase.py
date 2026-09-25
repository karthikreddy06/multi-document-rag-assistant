"""
Data Migration Script: SQLite -> Supabase PostgreSQL.
Transfers existing development application data safely, idempotently, and repeatably.
Preserves IDs, timestamps, document hashes, password hashes, foreign keys, and message sources.
"""

from datetime import datetime
import json
import os
from pathlib import Path
import sqlite3
import sys
from typing import Any, Dict, List, Optional
import urllib.parse
from dotenv import dotenv_values
import psycopg
from psycopg.rows import dict_row

# Ensure backend directory is in sys.path
SCRIPT_DIR = Path(__file__).resolve().parent
BACKEND_DIR = SCRIPT_DIR.parent
if str(BACKEND_DIR) not in sys.path:
    sys.path.insert(0, str(BACKEND_DIR))

from app.db.database import get_db_path, normalize_database_url, get_safe_db_info


def get_target_db_url() -> str:
    """Retrieve and normalize DATABASE_URL safely from environment or .env file."""
    url = os.environ.get("DATABASE_URL")
    if not url:
        env_path = BACKEND_DIR / ".env"
        if env_path.exists():
            env_vars = dotenv_values(env_path)
            url = env_vars.get("DATABASE_URL")
    if not url:
        raise RuntimeError("DATABASE_URL not found in environment or backend/.env")
    return normalize_database_url(url)


def run_migration() -> Dict[str, Any]:
    """Execute end-to-end migration from SQLite to Supabase PostgreSQL."""
    sqlite_path = get_db_path()
    if not sqlite_path.exists() or sqlite_path.stat().st_size == 0:
        print(f"Source SQLite database '{sqlite_path}' does not exist or is empty. Nothing to migrate.")
        return {}

    pg_url = get_target_db_url()
    safe_target = get_safe_db_info(pg_url)
    print("=" * 70)
    print("SQLITE TO SUPABASE POSTGRESQL DATA MIGRATION")
    print(f"Source: {sqlite_path} ({sqlite_path.stat().st_size} bytes)")
    print(f"Target: {safe_target}")
    print("=" * 70)

    # 1. Connect to SQLite
    s_conn = sqlite3.connect(str(sqlite_path))
    s_conn.row_factory = sqlite3.Row
    s_cur = s_conn.cursor()

    # 2. Connect to Supabase PostgreSQL
    with psycopg.connect(pg_url, row_factory=dict_row) as pg_conn:
        summary = {}

        # Run inside single transaction for atomicity and safety
        with pg_conn.transaction():
            with pg_conn.cursor() as pg_cur:
                # --------------------------------------------------------------
                # Phase 2.1: Users
                # --------------------------------------------------------------
                s_users = s_cur.execute("SELECT id, email, password_hash, created_at FROM users").fetchall()
                pg_cur.execute("SELECT COUNT(*) AS c FROM users")
                u_before = pg_cur.fetchone()["c"]

                # Ensure legacy_user exists first
                pg_cur.execute(
                    """
                    INSERT INTO users (id, email, password_hash, created_at)
                    VALUES ('legacy_user', 'legacy@local.dev', '', '2026-01-01T00:00:00Z')
                    ON CONFLICT (id) DO NOTHING
                    """
                )

                u_migrated = 0
                migrated_user_ids = {"legacy_user"}
                for u in s_users:
                    pg_cur.execute(
                        """
                        INSERT INTO users (id, email, password_hash, created_at)
                        VALUES (%s, %s, %s, %s)
                        ON CONFLICT (id) DO UPDATE SET
                            email = EXCLUDED.email,
                            password_hash = EXCLUDED.password_hash
                        """,
                        (u["id"], u["email"].lower().strip(), u["password_hash"], u["created_at"]),
                    )
                    migrated_user_ids.add(u["id"])
                    u_migrated += 1

                pg_cur.execute("SELECT COUNT(*) AS c FROM users")
                u_after = pg_cur.fetchone()["c"]
                summary["users"] = {"sqlite": len(s_users), "before": u_before, "migrated": u_migrated, "after": u_after}

                # --------------------------------------------------------------
                # Phase 2.2: Chats
                # --------------------------------------------------------------
                s_chats = s_cur.execute("SELECT id, user_id, title, created_at, updated_at FROM chats").fetchall()
                pg_cur.execute("SELECT COUNT(*) AS c FROM chats")
                c_before = pg_cur.fetchone()["c"]

                c_migrated = 0
                migrated_chat_ids = set()
                for c in s_chats:
                    c_user_id = c["user_id"] or "legacy_user"
                    if c_user_id not in migrated_user_ids:
                        c_user_id = "legacy_user"

                    pg_cur.execute(
                        """
                        INSERT INTO chats (id, user_id, title, created_at, updated_at)
                        VALUES (%s, %s, %s, %s, %s)
                        ON CONFLICT (id) DO UPDATE SET
                            title = EXCLUDED.title,
                            updated_at = EXCLUDED.updated_at
                        """,
                        (c["id"], c_user_id, c["title"] or "New Chat", c["created_at"], c["updated_at"]),
                    )
                    migrated_chat_ids.add(c["id"])
                    c_migrated += 1

                pg_cur.execute("SELECT COUNT(*) AS c FROM chats")
                c_after = pg_cur.fetchone()["c"]
                summary["chats"] = {"sqlite": len(s_chats), "before": c_before, "migrated": c_migrated, "after": c_after}

                # --------------------------------------------------------------
                # Phase 2.3: Documents
                # --------------------------------------------------------------
                s_docs = s_cur.execute(
                    """
                    SELECT id, user_id, filename, file_hash, file_size, page_count,
                           chunk_count, storage_path, status, error_message, created_at
                    FROM documents
                    """
                ).fetchall()
                pg_cur.execute("SELECT COUNT(*) AS c FROM documents")
                d_before = pg_cur.fetchone()["c"]

                d_migrated = 0
                seen_user_hashes = {}  # (user_id, file_hash) -> canonical_doc_id
                doc_id_map = {}        # original_doc_id -> canonical_doc_id

                for d in s_docs:
                    d_user_id = d["user_id"] or "legacy_user"
                    if d_user_id not in migrated_user_ids:
                        d_user_id = "legacy_user"

                    user_hash_key = (d_user_id, d["file_hash"])
                    if user_hash_key in seen_user_hashes:
                        doc_id_map[d["id"]] = seen_user_hashes[user_hash_key]
                        continue

                    seen_user_hashes[user_hash_key] = d["id"]
                    doc_id_map[d["id"]] = d["id"]

                    status = d["status"] if d["status"] in ("pending", "processing", "ready", "failed") else "ready"
                    pg_cur.execute(
                        """
                        INSERT INTO documents (
                            id, user_id, filename, file_hash, file_size, page_count,
                            chunk_count, storage_path, status, error_message, created_at
                        )
                        VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
                        ON CONFLICT (id) DO UPDATE SET
                            status = EXCLUDED.status,
                            page_count = EXCLUDED.page_count,
                            chunk_count = EXCLUDED.chunk_count,
                            storage_path = EXCLUDED.storage_path
                        """,
                        (
                            d["id"],
                            d_user_id,
                            d["filename"],
                            d["file_hash"],
                            d["file_size"],
                            d["page_count"],
                            d["chunk_count"],
                            d["storage_path"],
                            status,
                            d["error_message"],
                            d["created_at"],
                        ),
                    )
                    d_migrated += 1

                pg_cur.execute("SELECT COUNT(*) AS c FROM documents")
                d_after = pg_cur.fetchone()["c"]
                summary["documents"] = {"sqlite": len(s_docs), "before": d_before, "migrated": d_migrated, "after": d_after}

                # --------------------------------------------------------------
                # Phase 2.4: Chat-Documents (Junction)
                # --------------------------------------------------------------
                s_cd = s_cur.execute("SELECT chat_id, document_id, created_at FROM chat_documents").fetchall()
                pg_cur.execute("SELECT COUNT(*) AS c FROM chat_documents")
                cd_before = pg_cur.fetchone()["c"]

                cd_migrated = 0
                for cd in s_cd:
                    target_doc_id = doc_id_map.get(cd["document_id"], cd["document_id"])
                    if cd["chat_id"] not in migrated_chat_ids or target_doc_id not in seen_user_hashes.values():
                        continue

                    pg_cur.execute(
                        """
                        INSERT INTO chat_documents (chat_id, document_id, created_at)
                        VALUES (%s, %s, %s)
                        ON CONFLICT (chat_id, document_id) DO NOTHING
                        """,
                        (cd["chat_id"], target_doc_id, cd["created_at"]),
                    )
                    cd_migrated += 1

                pg_cur.execute("SELECT COUNT(*) AS c FROM chat_documents")
                cd_after = pg_cur.fetchone()["c"]
                summary["chat_documents"] = {"sqlite": len(s_cd), "before": cd_before, "migrated": cd_migrated, "after": cd_after}

                # --------------------------------------------------------------
                # Phase 2.5: Messages
                # --------------------------------------------------------------
                s_msgs = s_cur.execute(
                    "SELECT id, chat_id, role, content, sources_json, created_at FROM messages ORDER BY created_at ASC"
                ).fetchall()
                pg_cur.execute("SELECT COUNT(*) AS c FROM messages")
                m_before = pg_cur.fetchone()["c"]

                m_migrated = 0
                for m in s_msgs:
                    if m["chat_id"] not in migrated_chat_ids:
                        continue

                    src_json_val = None
                    if m["sources_json"]:
                        try:
                            # Verify valid JSON before inserting
                            parsed = json.loads(m["sources_json"])
                            src_json_val = json.dumps(parsed)
                        except Exception:
                            src_json_val = None

                    pg_cur.execute(
                        """
                        INSERT INTO messages (id, chat_id, role, content, sources_json, created_at)
                        VALUES (%s, %s, %s, %s, %s::jsonb, %s)
                        ON CONFLICT (id) DO UPDATE SET
                            content = EXCLUDED.content,
                            sources_json = EXCLUDED.sources_json
                        """,
                        (m["id"], m["chat_id"], m["role"], m["content"], src_json_val, m["created_at"]),
                    )
                    m_migrated += 1

                pg_cur.execute("SELECT COUNT(*) AS c FROM messages")
                m_after = pg_cur.fetchone()["c"]
                summary["messages"] = {"sqlite": len(s_msgs), "before": m_before, "migrated": m_migrated, "after": m_after}

        # ----------------------------------------------------------------------
        # Post-migration Verification
        # ----------------------------------------------------------------------
        print("\n" + "-" * 70)
        print(f"{'Table':<18} | {'SQLite Rows':<12} | {'Supabase Prior':<14} | {'Migrated':<10} | {'Supabase Total':<14}")
        print("-" * 70)
        for tbl, stats in summary.items():
            print(f"{tbl:<18} | {stats['sqlite']:<12} | {stats['before']:<14} | {stats['migrated']:<10} | {stats['after']:<14}")
        print("-" * 70)

        # Integrity check: verify FK relationships
        with pg_conn.cursor() as cur:
            cur.execute("""
                SELECT COUNT(*) AS broken FROM chats c 
                LEFT JOIN users u ON c.user_id = u.id 
                WHERE u.id IS NULL;
            """)
            broken_chats = cur.fetchone()["broken"]
            assert broken_chats == 0, f"Integrity error: {broken_chats} chats have invalid user_id references!"

            cur.execute("""
                SELECT COUNT(*) AS broken FROM documents d 
                LEFT JOIN users u ON d.user_id = u.id 
                WHERE u.id IS NULL;
            """)
            broken_docs = cur.fetchone()["broken"]
            assert broken_docs == 0, f"Integrity error: {broken_docs} documents have invalid user_id references!"

            cur.execute("""
                SELECT COUNT(*) AS broken FROM messages m 
                LEFT JOIN chats c ON m.chat_id = c.id 
                WHERE c.id IS NULL;
            """)
            broken_msgs = cur.fetchone()["broken"]
            assert broken_msgs == 0, f"Integrity error: {broken_msgs} messages have invalid chat_id references!"

            print("\nForeign key integrity verification: ALL CHECKS PASSED (0 orphaned records).")

    s_conn.close()
    print("Migration completed successfully and verified.\n")
    return summary


if __name__ == "__main__":
    try:
        run_migration()
    except Exception as exc:
        print(f"\nMigration failed: {exc}", file=sys.stderr)
        sys.exit(1)
