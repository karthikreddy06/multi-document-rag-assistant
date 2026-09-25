"""
Batched pgvector indexer for Dataset.XLSX (doc_a32aa6fb75bf4aab).

Processes 17,311 chunks in configurable batch sizes (default 200) and commits
each batch to Supabase document_chunks immediately, so progress survives
interruptions. Re-running is safe: already-inserted chunk IDs are skipped
via ON CONFLICT DO NOTHING.

Does NOT print or expose secrets.
"""

import json
import os
import sys
from pathlib import Path

SCRIPT_DIR = Path(__file__).resolve().parent
BACKEND_DIR = SCRIPT_DIR.parent
PROJECT_ROOT = BACKEND_DIR.parent
if str(BACKEND_DIR) not in sys.path:
    sys.path.insert(0, str(BACKEND_DIR))

from app.config import settings
from app.db.database import get_db_connection
from app.ingestion.chunker import DocumentChunker
from app.ingestion.parsers import ParserRegistry
from app.utils.logger import setup_logger

logger = setup_logger("scripts.index_xlsx_batched")

DOC_ID    = "doc_a32aa6fb75bf4aab"
FILENAME  = "Dataset.XLSX"
FILE_HASH = "32d3dd6f45ad15eeffd3d484e8d07752f76ed9d0384bff9a346b67f786d91b9c"
USER_ID   = "legacy_user"
EMBED_BATCH = 200   # chunks per ONNX call – tune lower if OOM
DB_BATCH    = 200   # chunks per INSERT transaction


def find_xlsx_file() -> Path:
    candidate_dirs = [
        BACKEND_DIR / "data" / "uploads",
        BACKEND_DIR / "documents",
        BACKEND_DIR / "data",
        PROJECT_ROOT / "data" / "uploads",
        PROJECT_ROOT / "documents",
    ]
    # Primary: search by hash-named file (uploads store files as SHA-256 name)
    for d in candidate_dirs:
        if d.exists():
            for p in d.rglob(f"{FILE_HASH}.xlsx"):
                return p.resolve()
            for p in d.rglob(f"{FILE_HASH}.XLSX"):
                return p.resolve()
    # Fallback: search by original filename
    for d in candidate_dirs:
        if d.exists():
            for p in d.rglob("*"):
                if p.is_file() and p.suffix.lower() == ".xlsx" and p.name.upper() == FILENAME.upper():
                    return p.resolve()
    raise FileNotFoundError(f"Cannot find {FILENAME} (hash={FILE_HASH}) in candidate dirs.")


def get_already_indexed_chunk_ids() -> set:
    with get_db_connection() as conn:
        with conn.cursor() as cur:
            cur.execute(
                "SELECT id FROM document_chunks WHERE document_id = %s;",
                (DOC_ID,),
            )
            return {row["id"] for row in cur.fetchall()}


def insert_batch(chunks_with_embeddings):
    insert_sql = """
        INSERT INTO document_chunks (
            id, document_id, user_id, chunk_index, chunk_text,
            embedding, filename, page_number, section, metadata, created_at
        ) VALUES (
            %s, %s, %s, %s, %s,
            %s::vector(384), %s, %s, %s, %s, NOW()
        )
        ON CONFLICT (id) DO NOTHING;
    """
    with get_db_connection() as conn:
        with conn.cursor() as cur:
            for chunk, emb, idx in chunks_with_embeddings:
                meta = dict(chunk.metadata or {})
                cur.execute(
                    insert_sql,
                    (
                        chunk.chunk_id,
                        DOC_ID,
                        USER_ID,
                        idx,
                        chunk.text,
                        str(emb),
                        FILENAME,
                        meta.get("page_number"),
                        meta.get("section"),
                        json.dumps(meta),
                    ),
                )
        conn.commit()


def main():
    print("=" * 60)
    print("BATCHED XLSX PGVECTOR INDEXER")
    print(f"Document: {FILENAME} | doc_id: {DOC_ID}")
    print(f"Embed batch size: {EMBED_BATCH} | DB batch size: {DB_BATCH}")
    print("=" * 60)

    # 1. Find file
    xlsx_path = find_xlsx_file()
    logger.info(f"Found XLSX at: {xlsx_path}")

    # 2. Parse
    logger.info("Parsing XLSX (this may take ~30s)...")
    registry = ParserRegistry()
    parsed_docs = registry.parse(
        file_path=xlsx_path,
        original_filename=FILENAME,
        file_hash=FILE_HASH,
    )
    logger.info(f"Parsed {len(parsed_docs)} sheets/pages.")

    for p in parsed_docs:
        p.metadata["document_id"] = DOC_ID
        p.metadata["user_id"] = USER_ID
        p.metadata["filename"] = FILENAME

    # 3. Chunk
    logger.info("Chunking...")
    chunker = DocumentChunker()
    chunks = chunker.split_documents(parsed_docs)
    logger.info(f"Produced {len(chunks)} chunks.")

    for idx, c in enumerate(chunks):
        c.metadata["document_id"] = DOC_ID
        c.metadata["user_id"] = USER_ID
        c.metadata["filename"] = FILENAME
        c.metadata["chunk_index"] = idx

    # 4. Check which chunk IDs are already indexed
    already_done = get_already_indexed_chunk_ids()
    logger.info(f"Already indexed: {len(already_done)} chunks. Remaining: {len(chunks) - len(already_done)}.")

    pending = [(c, i) for i, c in enumerate(chunks) if c.chunk_id not in already_done]
    if not pending:
        print("All chunks already indexed. Nothing to do.")
        return

    # 5. Lazy-load ONNX model
    logger.info("Loading ONNX all-MiniLM-L6-v2 model...")
    from chromadb.utils.embedding_functions import ONNXMiniLM_L6_V2
    ef = ONNXMiniLM_L6_V2(preferred_providers=["CPUExecutionProvider"])
    logger.info("Model loaded.")

    # 6. Embed + insert in batches
    total = len(pending)
    inserted = 0
    errors = 0

    for batch_start in range(0, total, EMBED_BATCH):
        batch = pending[batch_start: batch_start + EMBED_BATCH]
        texts = [c.text.strip() for c, _ in batch]
        batch_num = batch_start // EMBED_BATCH + 1
        total_batches = (total + EMBED_BATCH - 1) // EMBED_BATCH

        logger.info(
            f"Batch {batch_num}/{total_batches}: embedding {len(texts)} chunks "
            f"(global {batch_start+1}-{batch_start+len(batch)}/{total})..."
        )

        try:
            raw_embs = ef(texts)
            embeddings = []
            for emb in raw_embs:
                vec = emb.tolist() if hasattr(emb, "tolist") else [float(x) for x in emb]
                if len(vec) != 384:
                    raise ValueError(f"Expected 384-d embedding, got {len(vec)}-d.")
                embeddings.append(vec)
        except Exception as e:
            logger.error(f"Embedding batch {batch_num} failed: {e}")
            errors += len(batch)
            continue

        # Build rows and insert
        rows = [(chunk, emb, idx) for (chunk, idx), emb in zip(batch, embeddings)]
        try:
            insert_batch(rows)
            inserted += len(rows)
            logger.info(f"Batch {batch_num}/{total_batches}: inserted {len(rows)} chunks. Total inserted: {inserted}/{total}.")
        except Exception as e:
            logger.error(f"DB insert batch {batch_num} failed: {e}")
            errors += len(rows)

    # 7. Update documents table
    with get_db_connection() as conn:
        with conn.cursor() as cur:
            cur.execute(
                """
                UPDATE documents
                SET status = 'ready',
                    page_count = %s,
                    chunk_count = %s,
                    error_message = NULL
                WHERE id = %s;
                """,
                (len(parsed_docs), inserted + len(already_done), DOC_ID),
            )
        conn.commit()

    print("\n" + "=" * 60)
    print("XLSX INDEXING SUMMARY")
    print("=" * 60)
    print(f"Total chunks:       {len(chunks)}")
    print(f"Already indexed:    {len(already_done)}")
    print(f"Newly inserted:     {inserted}")
    print(f"Errors:             {errors}")
    print(f"Final chunk total:  {inserted + len(already_done)}")
    print("=" * 60)


if __name__ == "__main__":
    main()
