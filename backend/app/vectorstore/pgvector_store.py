"""
PgVector Storage Module using PostgreSQL and pgvector.
Provides persistent vector storage for production on Supabase,
matching the ChromaDB VectorStore interface for drop-in compatibility.
"""

import json
from typing import Any, Dict, List, Optional, Tuple
from app.config import settings
from app.db.database import get_db_connection, get_pg_pool, PooledConnectionProxy
from app.models import Chunk
from app.utils.logger import setup_logger

logger = setup_logger("vectorstore.pgvector")

STANDARD_COLUMNS = {
    "id",
    "document_id",
    "user_id",
    "chunk_index",
    "filename",
    "page_number",
    "section",
}


def parse_where_clause(where: Dict[str, Any]) -> Tuple[str, List[Any]]:
    """
    Recursively parse a Chroma-style where dictionary into a SQL WHERE clause and parameter list.
    Supports $and, $or, $in, top-level column matches, and metadata JSONB lookups.
    """
    if not where:
        return "", []

    clauses: List[str] = []
    params: List[Any] = []

    for key, val in where.items():
        if key == "$and" and isinstance(val, list):
            sub_clauses = []
            for sub in val:
                c, p = parse_where_clause(sub)
                if c:
                    sub_clauses.append(f"({c})")
                    params.extend(p)
            if sub_clauses:
                clauses.append(" AND ".join(sub_clauses))

        elif key == "$or" and isinstance(val, list):
            sub_clauses = []
            for sub in val:
                c, p = parse_where_clause(sub)
                if c:
                    sub_clauses.append(f"({c})")
                    params.extend(p)
            if sub_clauses:
                clauses.append(" OR ".join(sub_clauses))

        elif isinstance(val, dict) and "$in" in val:
            in_vals = val["$in"]
            if key in STANDARD_COLUMNS:
                clauses.append(f"{key} = ANY(%s)")
                params.append(list(in_vals))
            else:
                clauses.append(f"(metadata->>%s) = ANY(%s)")
                params.extend([key, [str(v) for v in in_vals]])

        elif key in STANDARD_COLUMNS:
            clauses.append(f"{key} = %s")
            params.append(val)

        else:
            clauses.append(f"(metadata->>%s) = %s")
            params.extend([key, str(val)])

    if not clauses:
        return "", []

    return " AND ".join(clauses), params


from app.vectorstore.store import VectorStore


class PgVectorStore(VectorStore):
    """Production pgvector client for persistent chunk upserts and semantic querying."""

    def __init__(self, table_name: str = "document_chunks"):
        self.table_name = table_name
        self.collection = self  # Expose collection shim for collection.get() compatibility
        # Capture the pool at construction time so monkeypatching settings later
        # cannot accidentally cause get_db_connection() to return a SQLite conn.
        self._pool = get_pg_pool()
        logger.info(f"PgVectorStore initialized with table '{self.table_name}'.")

    def _get_conn(self):
        """Return a pooled PostgreSQL connection context manager."""
        return PooledConnectionProxy(self._pool)

    def count(self) -> int:
        """Return total chunks in document_chunks."""
        try:
            with self._get_conn() as conn:
                with conn.cursor() as cur:
                    cur.execute(f"SELECT COUNT(*) as cnt FROM {self.table_name}")
                    row = cur.fetchone()
                    return int(row["cnt"] if row else 0)
        except Exception as e:
            logger.error(f"Failed to count chunks in PgVectorStore: {e}")
            return 0

    def clear(self) -> None:
        """Clear all chunks from the table."""
        logger.info(f"Clearing all rows from '{self.table_name}'...")
        try:
            with self._get_conn() as conn:
                with conn.cursor() as cur:
                    cur.execute(f"TRUNCATE TABLE {self.table_name}")
                conn.commit()
            logger.info("PgVectorStore cleared successfully.")
        except Exception as e:
            logger.error(f"Failed to clear PgVectorStore: {e}")
            raise

    def upsert_chunks(self, chunks: List[Chunk], embeddings: List[List[float]]) -> int:
        """
        Idempotently insert or update chunks in PostgreSQL with pgvector embeddings.
        Returns the number of chunks added/updated.
        """
        if not chunks:
            logger.warning("No chunks provided for storage.")
            return 0

        if len(chunks) != len(embeddings):
            raise ValueError(f"Mismatch between chunks ({len(chunks)}) and embeddings ({len(embeddings)})")

        insert_sql = f"""
            INSERT INTO {self.table_name} (
                id, document_id, user_id, chunk_index, chunk_text,
                embedding, filename, page_number, section, metadata
            ) VALUES (
                %s, %s, %s, %s, %s,
                %s::vector(384), %s, %s, %s, %s
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

        try:
            with self._get_conn() as conn:
                with conn.cursor() as cur:
                    for chunk, emb in zip(chunks, embeddings):
                        meta = dict(chunk.metadata or {})
                        doc_id = meta.get("document_id") or None
                        user_id = meta.get("user_id") or "legacy_user"
                        chunk_idx = meta.get("chunk_index", 0)
                        filename = meta.get("filename", "")
                        page_num = meta.get("page_number")
                        section = meta.get("section")
                        emb_str = str(emb)
                        meta_json = json.dumps(meta)

                        cur.execute(insert_sql, (
                                chunk.chunk_id, doc_id, user_id, chunk_idx, chunk.text,
                                emb_str, filename, page_num, section, meta_json,
                            ))
                conn.commit()
            logger.info(f"Successfully upserted {len(chunks)} chunks into PgVectorStore.")
            return len(chunks)
        except Exception as e:
            logger.error(f"Failed to upsert chunks into PgVectorStore: {e}")
            raise RuntimeError(f"PgVectorStore upsert failure: {e}") from e

    def query(
        self,
        query_embedding: List[float],
        top_k: Optional[int] = None,
        where: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        """
        Query closest chunks using cosine distance (<=>).
        Returns a dict matching the Chroma query response schema:
        {"ids": [[...]], "documents": [[...]], "metadatas": [[...]], "distances": [[...]]}
        """
        k = top_k or settings.top_k
        emb_str = str(query_embedding)

        where_sql, params = parse_where_clause(where or {})
        where_part = f"WHERE {where_sql}" if where_sql else ""

        sql = f"""
            SELECT
                id,
                chunk_text,
                metadata,
                filename,
                page_number,
                section,
                document_id,
                user_id,
                (embedding <=> %s::vector(384)) AS distance
            FROM {self.table_name}
            {where_part}
            ORDER BY embedding <=> %s::vector(384) ASC
            LIMIT %s;
        """
        query_params = [emb_str, *params, emb_str, k]

        try:
            with self._get_conn() as conn:
                with conn.cursor() as cur:
                    cur.execute(sql, query_params)
                    rows = cur.fetchall()

            ids = []
            docs = []
            metas = []
            distances = []

            for row in rows:
                ids.append(row["id"])
                docs.append(row["chunk_text"])
                # Combine metadata JSONB with top-level fields
                meta = dict(row["metadata"] or {})
                meta["filename"] = row["filename"]
                if row["page_number"] is not None:
                    meta["page_number"] = row["page_number"]
                if row["section"]:
                    meta["section"] = row["section"]
                if row["document_id"]:
                    meta["document_id"] = row["document_id"]
                if row["user_id"]:
                    meta["user_id"] = row["user_id"]
                metas.append(meta)
                distances.append(float(row["distance"]))

            return {
                "ids": [ids],
                "documents": [docs],
                "metadatas": [metas],
                "distances": [distances],
            }
        except Exception as e:
            logger.error(f"Failed to query PgVectorStore: {e}")
            raise RuntimeError(f"PgVectorStore query failure: {e}") from e

    def get(
        self,
        where: Optional[Dict[str, Any]] = None,
        include: Optional[List[str]] = None,
        limit: Optional[int] = None,
    ) -> Dict[str, Any]:
        """
        Retrieve chunks matching filter, matching Chroma get() schema:
        {"ids": [...], "documents": [...], "metadatas": [...]}
        """
        where_sql, params = parse_where_clause(where or {})
        where_part = f"WHERE {where_sql}" if where_sql else ""
        limit_part = f"LIMIT {int(limit)}" if limit is not None else ""

        sql = f"""
            SELECT
                id,
                chunk_text,
                metadata,
                filename,
                page_number,
                section,
                document_id,
                user_id
            FROM {self.table_name}
            {where_part}
            ORDER BY chunk_index ASC
            {limit_part};
        """

        try:
            with self._get_conn() as conn:
                with conn.cursor() as cur:
                    cur.execute(sql, params)
                    rows = cur.fetchall()

            ids = []
            docs = []
            metas = []

            for row in rows:
                ids.append(row["id"])
                docs.append(row["chunk_text"])
                meta = dict(row["metadata"] or {})
                meta["filename"] = row["filename"]
                if row["page_number"] is not None:
                    meta["page_number"] = row["page_number"]
                if row["section"]:
                    meta["section"] = row["section"]
                if row["document_id"]:
                    meta["document_id"] = row["document_id"]
                if row["user_id"]:
                    meta["user_id"] = row["user_id"]
                metas.append(meta)

            return {
                "ids": ids,
                "documents": docs,
                "metadatas": metas,
            }
        except Exception as e:
            logger.error(f"Failed to get from PgVectorStore: {e}")
            raise RuntimeError(f"PgVectorStore get failure: {e}") from e

    def delete_by_filename(self, filename: str, user_id: Optional[str] = None) -> int:
        """Delete all chunks for a filename, scoped by user_id if provided."""
        if not filename:
            return 0
        try:
            with self._get_conn() as conn:
                with conn.cursor() as cur:
                    if user_id:
                        cur.execute(
                            f"DELETE FROM {self.table_name} WHERE filename = %s AND user_id = %s RETURNING id",
                            (filename, user_id),
                        )
                    else:
                        cur.execute(
                            f"DELETE FROM {self.table_name} WHERE filename = %s RETURNING id",
                            (filename,),
                        )
                    deleted = cur.fetchall()
                conn.commit()
                cnt = len(deleted)
                logger.info(f"Deleted {cnt} chunks for filename='{filename}' (user_id={user_id}) from PgVectorStore.")
                return cnt
        except Exception as e:
            logger.error(f"Failed to delete by filename from PgVectorStore: {e}")
            raise RuntimeError(f"PgVectorStore delete failure: {e}") from e

    def delete_by_document_id(self, document_id: str, user_id: Optional[str] = None) -> int:
        """Delete all chunks for a document_id, scoped by user_id if provided."""
        if not document_id:
            return 0
        try:
            with self._get_conn() as conn:
                with conn.cursor() as cur:
                    if user_id:
                        cur.execute(
                            f"DELETE FROM {self.table_name} WHERE document_id = %s AND user_id = %s RETURNING id",
                            (document_id, user_id),
                        )
                    else:
                        cur.execute(
                            f"DELETE FROM {self.table_name} WHERE document_id = %s RETURNING id",
                            (document_id,),
                        )
                    deleted = cur.fetchall()
                conn.commit()
                cnt = len(deleted)
                logger.info(f"Deleted {cnt} chunks for document_id='{document_id}' (user_id={user_id}) from PgVectorStore.")
                return cnt
        except Exception as e:
            logger.error(f"Failed to delete by document_id from PgVectorStore: {e}")
            raise RuntimeError(f"PgVectorStore delete failure: {e}") from e

    def get_indexed_files(self, user_id: Optional[str] = None) -> Dict[str, str]:
        """Return {filename: file_hash} mapping for indexed documents."""
        try:
            with self._get_conn() as conn:
                with conn.cursor() as cur:
                    if user_id:
                        cur.execute(
                            f"""
                            SELECT DISTINCT filename, (metadata->>'file_hash') as file_hash
                            FROM {self.table_name}
                            WHERE user_id = %s AND filename IS NOT NULL AND filename != ''
                            """,
                            (user_id,),
                        )
                    else:
                        cur.execute(
                            f"""
                            SELECT DISTINCT filename, (metadata->>'file_hash') as file_hash
                            FROM {self.table_name}
                            WHERE filename IS NOT NULL AND filename != ''
                            """
                        )
                    rows = cur.fetchall()

            file_map: Dict[str, str] = {}
            for row in rows:
                fn = row["filename"]
                fh = row["file_hash"] or ""
                if fn and fn not in file_map:
                    file_map[fn] = fh
            return file_map
        except Exception as e:
            logger.warning(f"Could not retrieve indexed files from PgVectorStore: {e}")
            return {}
