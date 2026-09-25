"""
Document File & PgVector Migration Script for Supabase.
Migrates local source document files to Supabase Storage (rag-files bucket)
and re-chunks & re-indexes them into Supabase pgvector (document_chunks table)
using the production 384-dimensional all-MiniLM-L6-v2 ONNX embedding model.

Idempotent, hash-aware, user-scoped, and transaction-safe.
Does not print or expose any secrets or confidential file contents.
"""

import hashlib
import json
import mimetypes
import os
from pathlib import Path
import sys
from typing import Any, Dict, List, Optional, Tuple

# Ensure backend root is in Python sys.path
SCRIPT_DIR = Path(__file__).resolve().parent
BACKEND_DIR = SCRIPT_DIR.parent
PROJECT_ROOT = BACKEND_DIR.parent
if str(BACKEND_DIR) not in sys.path:
    sys.path.insert(0, str(BACKEND_DIR))

from app.config import settings
from app.db.database import get_db_connection
from app.embeddings.service import EmbeddingService
from app.ingestion.chunker import DocumentChunker
from app.ingestion.parsers import ParserRegistry, UnsupportedFileTypeError
from app.services.storage import get_document_storage_path, SupabaseStorageService
from app.utils.logger import setup_logger

logger = setup_logger("scripts.migrate_documents_to_supabase")

# ─── Explicit test/fixture exclusion list ─────────────────────────────────────
# These files MUST NEVER be uploaded to Supabase Storage or indexed into pgvector.
# Verified against: backend/documents/ (Chroma test corpus) and test source code.
#
# To ADD a production document: do NOT add it here. Upload it through the RAG app.
# To EXCLUDE a new test fixture: add its filename here.
TEST_FIXTURE_EXCLUSION_LIST = frozenset({
    # ── backend/documents/ Chroma test corpus ──────────────────────────────
    "Alice_in_Wonderland.pdf",   # referenced in test_retrieval_generic.py, test_adaptive_retrieval.py
    "oldmansea.pdf",             # referenced in test_retrieval_generic.py, test_adaptive_retrieval.py
    "sample.pdf",                # referenced in test_retrieval_generic.py (pdflatex compilation)
    "company.pdf",               # referenced in test_adaptive_retrieval.py (company queries)
    # ── Benchmark / dataset files ──────────────────────────────────────────
    "Dataset.XLSX",
    "Dataset.xlsx",
    # ── Synthetic test files ───────────────────────────────────────────────
    "test.pdf",
    "test.txt",
    "notes.txt",
    "scores.csv",
    "staff.csv",
    "cascade_test.pdf",
})

# Filename patterns that indicate a test/benchmark file (case-insensitive)
TEST_FIXTURE_KEYWORDS = ("_test.", "test_", "dataset", "benchmark", "fixture", "dummy", "mock")


def is_test_fixture(filename: str) -> tuple:
    """
    Return (True, reason) if the file is a known test fixture or benchmark.
    Return (False, '') for production documents.

    This is the definitive exclusion gate — any file matching here will never
    be uploaded to Supabase Storage or indexed into pgvector by migration scripts.
    """
    if filename in TEST_FIXTURE_EXCLUSION_LIST:
        return True, f"explicit fixture exclusion list"
    fn_lower = filename.lower()
    for kw in TEST_FIXTURE_KEYWORDS:
        if kw in fn_lower:
            return True, f"keyword pattern '{kw}' in filename"
    return False, ""

def find_candidate_local_files() -> List[Path]:
    """Scan candidate local storage directories for uploaded source files."""
    candidate_dirs = [
        BACKEND_DIR / "data" / "uploads",
        BACKEND_DIR / "documents",
        BACKEND_DIR / "data",
        PROJECT_ROOT / "data" / "uploads",
        PROJECT_ROOT / "documents",
    ]
    files: List[Path] = []
    seen = set()

    for d in candidate_dirs:
        if d.exists() and d.is_dir():
            for p in d.rglob("*"):
                if p.is_file():
                    resolved = p.resolve()
                    if resolved not in seen:
                        # Exclude database files, caches, virtual environments
                        if not any(part.startswith(".") or part in ("node_modules", "__pycache__", ".venv", ".venv311") for part in resolved.parts):
                            if resolved.suffix.lower() not in (".db", ".db-journal", ".py", ".pyc", ".sh", ".bat", ".sql", ".log"):
                                files.append(resolved)
                                seen.add(resolved)
    return files


def build_local_file_index(files: List[Path]) -> Dict[str, Path]:
    """Build a mapping from SHA-256 hash to Path for all candidate source files."""
    hash_to_path: Dict[str, Path] = {}
    for f in files:
        try:
            hasher = hashlib.sha256()
            with open(f, "rb") as fh:
                for block in iter(lambda: fh.read(65536), b""):
                    hasher.update(block)
            file_hash = hasher.hexdigest()
            if file_hash not in hash_to_path:
                hash_to_path[file_hash] = f
        except Exception as e:
            logger.warning(f"Could not hash local file '{f.name}': {e}")
    return hash_to_path


def resolve_local_file(
    doc: Dict[str, Any],
    hash_to_path: Dict[str, Path],
) -> Optional[Path]:
    """
    Resolve local source file for a database document record.
    1. Prefer matching by SHA-256 file_hash.
    2. Fall back to stored storage_path if valid and hash matches.
    3. Never guess by filename alone.
    """
    file_hash = (doc.get("file_hash") or "").strip()
    storage_path = (doc.get("storage_path") or "").strip()

    # 1. Match by SHA-256
    if file_hash and file_hash in hash_to_path:
        return hash_to_path[file_hash]

    # 2. Check storage_path fallback
    if storage_path:
        candidates = [
            Path(storage_path),
            BACKEND_DIR / storage_path,
            PROJECT_ROOT / storage_path,
            BACKEND_DIR / "data" / "uploads" / storage_path,
        ]
        for c in candidates:
            if c.exists() and c.is_file():
                # Verify hash if available
                if file_hash:
                    hasher = hashlib.sha256()
                    try:
                        with open(c, "rb") as fh:
                            for block in iter(lambda: fh.read(65536), b""):
                                hasher.update(block)
                        if hasher.hexdigest() == file_hash:
                            return c
                    except Exception:
                        pass
                else:
                    return c

    return None


def run_migration() -> Dict[str, int]:
    """
    Execute storage migration and pgvector rebuild for all valid PostgreSQL documents.
    """
    print("=" * 60)
    print("STARTING SUPABASE DOCUMENT & PGVECTOR MIGRATION")
    print("=" * 60)

    # 1. Initialize Supabase Storage
    storage_service = SupabaseStorageService()
    storage_service._ensure_bucket()

    # 2. Scan and index local files
    candidate_files = find_candidate_local_files()
    hash_to_path = build_local_file_index(candidate_files)
    logger.info(f"Scanned {len(candidate_files)} candidate local files ({len(hash_to_path)} unique hashes).")

    # 3. Retrieve PostgreSQL documents
    with get_db_connection() as conn:
        with conn.cursor() as cur:
            cur.execute(
                """
                SELECT id, user_id, filename, file_hash, file_size, storage_path, status
                FROM documents
                ORDER BY created_at ASC;
                """
            )
            documents = cur.fetchall()

            # Build set of doc_ids that already have pgvector chunks
            cur.execute("SELECT DISTINCT document_id FROM document_chunks;")
            already_indexed_ids = {row["document_id"] for row in cur.fetchall()}

    total_discovered = len(documents)
    matched_count = 0
    uploaded_count = 0
    skipped_count = 0
    indexed_docs = 0
    chunks_created = 0
    missing_docs = 0
    upload_failures = 0
    index_failures = 0

    # 4. Prepare Parser, Chunker, and ONNX 384-d Embedding Service
    parser_registry = ParserRegistry()
    chunker = DocumentChunker()
    embedding_service = EmbeddingService(
        provider="local",
        model="all-MiniLM-L6-v2",
        dimensions=384,
    )

    print(f"Discovered {total_discovered} document records in PostgreSQL.")
    print(f"Already indexed in pgvector: {len(already_indexed_ids)} documents.")

    for doc in documents:
        doc_id = doc["id"]
        user_id = doc["user_id"] or "legacy_user"
        filename = doc["filename"]
        file_hash = doc["file_hash"]
        current_storage_path = (doc.get("storage_path") or "").strip()

        # --- Exclusion gate: test/fixture/benchmark files are NEVER migrated ---
        is_fixture, fixture_reason = is_test_fixture(filename)
        if is_fixture:
            logger.info(
                f"[EXCLUDED] Skipping test/fixture file: '{filename}' "
                f"(reason: {fixture_reason}). Not uploading to Supabase."
            )
            skipped_count += 1
            continue

        # --- Skip check: already fully migrated ---
        already_in_storage = current_storage_path.startswith("users/")
        already_in_pgvector = doc_id in already_indexed_ids
        if already_in_storage and already_in_pgvector:
            skipped_count += 1
            logger.info(f"Skipping already-migrated doc id={doc_id} ('{filename}')")
            continue

        local_file = resolve_local_file(doc, hash_to_path)

        if not local_file:
            missing_docs += 1
            logger.info(f"Missing local source for document id={doc_id}, filename='{filename}'")
            continue

        matched_count += 1
        logger.info(f"Matched document id={doc_id} to '{local_file.name}'")

        # ----------------------------------------------------------------------
        # Phase A: Supabase Storage Upload (skip if already in Supabase)
        # ----------------------------------------------------------------------
        if already_in_storage:
            logger.info(f"Storage already migrated for doc id={doc_id}, skipping upload.")
            # still need to re-index pgvector below
        else:
            target_storage_path = get_document_storage_path(
                user_id=user_id,
                document_id=doc_id,
                filename=filename,
            )

            try:
                file_bytes = local_file.read_bytes()
                mime = mimetypes.guess_type(filename)[0] or "application/octet-stream"

                # Upload to Supabase Storage
                clean_path = storage_service.upload(
                    file_bytes=file_bytes,
                    file_path=target_storage_path,
                    content_type=mime,
                )

                # Verify existence
                if not storage_service.exists(clean_path):
                    raise RuntimeError(f"Storage verification failed: object '{clean_path}' not found after upload.")

                # Update PostgreSQL storage_path
                with get_db_connection() as conn:
                    with conn.cursor() as cur:
                        cur.execute(
                            "UPDATE documents SET storage_path = %s WHERE id = %s;",
                            (clean_path, doc_id),
                        )
                    conn.commit()

                uploaded_count += 1
                logger.info(f"Uploaded and verified storage path for document id={doc_id}: {clean_path}")

            except Exception as e:
                upload_failures += 1
                logger.error(f"Failed to upload document id={doc_id} ('{filename}'): {e}")
                continue

        # ----------------------------------------------------------------------
        # Phase B: Rebuild pgvector from original file
        # ----------------------------------------------------------------------
        try:
            parsed_docs = parser_registry.parse(
                file_path=local_file,
                original_filename=filename,
                file_hash=file_hash,
            )

            if not parsed_docs:
                logger.warning(f"No content parsed from '{filename}' for document id={doc_id}.")
                continue

            # Populate metadata
            for p_doc in parsed_docs:
                p_doc.metadata["document_id"] = doc_id
                p_doc.metadata["user_id"] = user_id
                p_doc.metadata["filename"] = filename
                p_doc.metadata["file_hash"] = file_hash

            chunks = chunker.split_documents(parsed_docs)
            if not chunks:
                logger.warning(f"No chunks produced for '{filename}' (doc_id={doc_id}).")
                continue

            for idx, c in enumerate(chunks):
                c.metadata["document_id"] = doc_id
                c.metadata["user_id"] = user_id
                c.metadata["filename"] = filename
                c.metadata["file_hash"] = file_hash
                c.metadata["chunk_index"] = idx

            texts = [c.text for c in chunks]
            embeddings = embedding_service.embed_batch(texts)

            if len(embeddings) != len(chunks):
                raise ValueError(f"Embedding count ({len(embeddings)}) != chunk count ({len(chunks)})")

            for emb in embeddings:
                if len(emb) != 384:
                    raise ValueError(f"Invalid embedding dimension {len(emb)}, expected 384.")

            # Insert into document_chunks transactionally
            with get_db_connection() as conn:
                with conn.cursor() as cur:
                    # 1. Remove old chunks for this document
                    cur.execute("DELETE FROM document_chunks WHERE document_id = %s;", (doc_id,))

                    # 2. Insert new chunks
                    insert_sql = """
                        INSERT INTO document_chunks (
                            id, document_id, user_id, chunk_index, chunk_text,
                            embedding, filename, page_number, section, metadata, created_at
                        ) VALUES (
                            %s, %s, %s, %s, %s,
                            %s::vector(384), %s, %s, %s, %s, NOW()
                        )
                        ON CONFLICT (id) DO UPDATE SET
                            chunk_text = EXCLUDED.chunk_text,
                            embedding = EXCLUDED.embedding,
                            metadata = EXCLUDED.metadata,
                            page_number = EXCLUDED.page_number,
                            section = EXCLUDED.section,
                            filename = EXCLUDED.filename,
                            user_id = EXCLUDED.user_id,
                            document_id = EXCLUDED.document_id,
                            chunk_index = EXCLUDED.chunk_index;
                    """
                    for idx, (chunk, emb) in enumerate(zip(chunks, embeddings)):
                        meta = dict(chunk.metadata or {})
                        page_num = meta.get("page_number")
                        section = meta.get("section")
                        cur.execute(
                            insert_sql,
                            (
                                chunk.chunk_id,
                                doc_id,
                                user_id,
                                idx,
                                chunk.text,
                                str(emb),
                                filename,
                                page_num,
                                section,
                                json.dumps(meta),
                            ),
                        )

                    # 3. Update documents catalog row
                    cur.execute(
                        """
                        UPDATE documents
                        SET status = 'ready',
                            page_count = %s,
                            chunk_count = %s,
                            error_message = NULL
                        WHERE id = %s;
                        """,
                        (len(parsed_docs), len(chunks), doc_id),
                    )
                conn.commit()

            indexed_docs += 1
            chunks_created += len(chunks)
            logger.info(f"Indexed doc id={doc_id}: {len(chunks)} chunks with 384-d embeddings.")

        except Exception as e:
            index_failures += 1
            logger.error(f"Failed to reindex document id={doc_id}: {e}")

    # Final summary report
    print("\n" + "=" * 60)
    print("MIGRATION SUMMARY")
    print("=" * 60)
    print(f"Documents discovered: {total_discovered}")
    print(f"Already migrated (skipped): {skipped_count}")
    print(f"Files matched: {matched_count}")
    print(f"Files uploaded: {uploaded_count}")
    print(f"Documents indexed: {indexed_docs}")
    print(f"Chunks created: {chunks_created}")
    print(f"Documents missing local source: {missing_docs}")
    print(f"Upload failures: {upload_failures}")
    print(f"Index failures: {index_failures}")
    print("=" * 60 + "\n")

    return {
        "documents_discovered": total_discovered,
        "already_migrated_skipped": skipped_count,
        "files_matched": matched_count,
        "files_uploaded": uploaded_count,
        "documents_indexed": indexed_docs,
        "chunks_created": chunks_created,
        "documents_missing_local_source": missing_docs,
        "upload_failures": upload_failures,
        "index_failures": index_failures,
    }


if __name__ == "__main__":
    run_migration()
