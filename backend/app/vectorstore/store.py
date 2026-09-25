"""
Vector Storage Module using ChromaDB.
Provides idempotent persistence, stable ID indexing, rich metadata storage,
and error-resilient collection querying.
"""

import os
from pathlib import Path
from typing import Any, Dict, List, Optional

# Disable ChromaDB telemetry errors cleanly
os.environ["ANONYMIZED_TELEMETRY"] = "False"
try:
    import posthog
    posthog.capture = lambda *args, **kwargs: None
except ImportError:
    pass

import chromadb
from chromadb.config import Settings as ChromaSettings

from app.config import settings
from app.models import Chunk
from app.utils.logger import setup_logger

logger = setup_logger("vectorstore.store")


class VectorStore:
    """Vector store abstraction routing to PgVectorStore (production) or ChromaVectorStore (local)."""

    def __new__(
        cls,
        persist_path: Optional[str] = None,
        collection_name: Optional[str] = None,
    ):
        if cls is VectorStore:
            if settings.is_pgvector:
                from app.vectorstore.pgvector_store import PgVectorStore
                return PgVectorStore()
            return ChromaVectorStore(persist_path=persist_path, collection_name=collection_name)
        return super().__new__(cls)


class ChromaVectorStore(VectorStore):
    """Production ChromaDB client supporting idempotent chunk upserts and metadata filtering."""

    def __init__(
        self,
        persist_path: Optional[str] = None,
        collection_name: Optional[str] = None,
    ):
        self.persist_path = persist_path or settings.chroma_path
        self.collection_name = collection_name or settings.collection_name

        Path(self.persist_path).mkdir(parents=True, exist_ok=True)

        chroma_settings = ChromaSettings(
            anonymized_telemetry=False,
            is_persistent=True
        )

        try:
            self.client = chromadb.PersistentClient(
                path=self.persist_path,
                settings=chroma_settings
            )
            self.collection = self.client.get_or_create_collection(
                name=self.collection_name
            )
            logger.info(
                f"ChromaDB initialized at '{self.persist_path}'. "
                f"Collection '{self.collection_name}' has {self.collection.count()} chunks."
            )
            self._migrate_chunks_user_id()
        except Exception as e:
            logger.error(f"Failed to initialize ChromaDB at '{self.persist_path}': {e}")
            raise RuntimeError(f"ChromaDB initialization error: {e}") from e

    def _migrate_chunks_user_id(self, default_user_id: str = "legacy_user") -> None:
        """Idempotently ensure all existing chunks carry a user_id metadata field."""
        try:
            total = self.collection.count()
            if total == 0:
                return
            raw = self.collection.get(include=["metadatas"])
            ids = raw.get("ids", []) or []
            metadatas = raw.get("metadatas", []) or []
            to_update_ids = []
            to_update_metas = []
            for cid, meta in zip(ids, metadatas):
                if meta is not None and "user_id" not in meta:
                    new_meta = dict(meta)
                    new_meta["user_id"] = default_user_id
                    to_update_ids.append(cid)
                    to_update_metas.append(new_meta)
            if to_update_ids:
                batch_size = 500
                for i in range(0, len(to_update_ids), batch_size):
                    self.collection.update(
                        ids=to_update_ids[i : i + batch_size],
                        metadatas=to_update_metas[i : i + batch_size],
                    )
                logger.info(f"Migrated {len(to_update_ids)} chunks to user_id='{default_user_id}'.")
        except Exception as e:
            logger.warning(f"Could not perform chunks user_id migration: {e}")

    def count(self) -> int:
        """Return total chunks in the collection."""
        return self.collection.count()

    def clear(self) -> None:
        """Delete all documents in the collection."""
        logger.info(f"Clearing collection '{self.collection_name}'...")
        try:
            self.client.delete_collection(self.collection_name)
            self.collection = self.client.get_or_create_collection(self.collection_name)
            logger.info("Collection cleared and recreated successfully.")
        except Exception as e:
            logger.error(f"Failed to clear collection: {e}")
            raise

    def upsert_chunks(self, chunks: List[Chunk], embeddings: List[List[float]]) -> int:
        """
        Idempotently insert or update chunks in ChromaDB with embeddings and primitive metadata.
        Returns the number of chunks added/updated.
        """
        if not chunks:
            logger.warning("No chunks provided for storage.")
            return 0

        if len(chunks) != len(embeddings):
            raise ValueError(f"Mismatch between chunks count ({len(chunks)}) and embeddings count ({len(embeddings)})")

        ids: List[str] = []
        documents: List[str] = []
        metadatas: List[Dict[str, Any]] = []

        for chunk in chunks:
            ids.append(chunk.chunk_id)
            documents.append(chunk.text)

            # ChromaDB requires metadata values to be primitive types: str, int, float, bool
            sanitized_meta: Dict[str, Any] = {}
            for k, v in chunk.metadata.items():
                if isinstance(v, (str, int, float, bool)):
                    sanitized_meta[k] = v
                elif v is None:
                    sanitized_meta[k] = ""
                else:
                    sanitized_meta[k] = str(v)
            metadatas.append(sanitized_meta)

        logger.info(f"Upserting {len(ids)} chunks into collection '{self.collection_name}'...")
        try:
            self.collection.upsert(
                ids=ids,
                documents=documents,
                embeddings=embeddings,
                metadatas=metadatas
            )
            logger.info(f"Successfully stored chunks. Total collection count now: {self.collection.count()}")
            return len(ids)
        except Exception as e:
            logger.error(f"Failed to upsert chunks into ChromaDB: {e}")
            raise RuntimeError(f"ChromaDB upsert failure: {e}") from e

    def query(
        self,
        query_embedding: List[float],
        top_k: Optional[int] = None,
        where: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        """
        Query the vector store for the closest chunks.
        """
        k = top_k or settings.top_k
        total = self.count()

        if total == 0:
            logger.warning("Query called on an empty ChromaDB collection.")
            return {"ids": [[]], "documents": [[]], "metadatas": [[]], "distances": [[]]}

        # n_results cannot exceed total items in collection
        n_results = min(k, total)

        query_args: Dict[str, Any] = {
            "query_embeddings": [query_embedding],
            "n_results": n_results,
        }
        if where:
            query_args["where"] = where

        try:
            results = self.collection.query(**query_args)
            return results
        except Exception as e:
            logger.error(f"Failed querying ChromaDB: {e}")
            raise RuntimeError(f"ChromaDB query failure: {e}") from e

    def get_all(self) -> Dict[str, Any]:
        """Retrieve all documents and metadata stored in the collection."""
        return self.collection.get()

    def get(
        self,
        where: Optional[Dict[str, Any]] = None,
        include: Optional[List[str]] = None,
        limit: Optional[int] = None,
    ) -> Dict[str, Any]:
        """Retrieve documents and metadata stored in the collection matching filters."""
        kwargs: Dict[str, Any] = {}
        if where:
            kwargs["where"] = where
        if include:
            kwargs["include"] = include
        if limit is not None:
            kwargs["limit"] = limit
        return self.collection.get(**kwargs)

    def delete_by_filename(self, filename: str, user_id: Optional[str] = None) -> int:
        """
        Delete all chunks associated with a specific document filename, scoped by user_id if provided.
        Returns the count of chunks removed.
        """
        if not filename:
            return 0
        try:
            where_filter: Dict[str, Any] = {"filename": filename}
            if user_id:
                where_filter = {"$and": [{"filename": filename}, {"user_id": user_id}]}
            records = self.collection.get(where=where_filter)
            ids_to_delete = records.get("ids", [])
            if ids_to_delete:
                self.collection.delete(ids=ids_to_delete)
                logger.info(f"Deleted {len(ids_to_delete)} stale chunks for file: {filename} (user_id={user_id})")
                return len(ids_to_delete)
            return 0
        except Exception as e:
            logger.error(f"Failed deleting chunks for {filename}: {e}")
            raise RuntimeError(f"ChromaDB delete failure: {e}") from e

    def delete_by_document_id(self, document_id: str, user_id: Optional[str] = None) -> int:
        """
        Delete all chunks associated with a specific document_id, scoped by user_id if provided.
        Returns the count of chunks removed.
        """
        if not document_id:
            return 0
        try:
            where_filter: Dict[str, Any] = {"document_id": document_id}
            if user_id:
                where_filter = {"$and": [{"document_id": document_id}, {"user_id": user_id}]}
            records = self.collection.get(where=where_filter)
            ids_to_delete = records.get("ids", [])
            if ids_to_delete:
                self.collection.delete(ids=ids_to_delete)
                logger.info(f"Deleted {len(ids_to_delete)} chunks for document_id: {document_id} (user_id={user_id})")
                return len(ids_to_delete)
            return 0
        except Exception as e:
            logger.error(f"Failed deleting chunks for document_id '{document_id}': {e}")
            raise RuntimeError(f"ChromaDB delete failure: {e}") from e

    def get_indexed_files(self, user_id: Optional[str] = None) -> Dict[str, str]:
        """
        Return a mapping of {filename: file_hash} for currently indexed documents.
        Scoped by user_id if provided.
        Uses metadata-only retrieval to avoid loading document texts into memory.
        """
        try:
            get_kwargs: Dict[str, Any] = {"include": ["metadatas"]}
            if user_id:
                get_kwargs["where"] = {"user_id": user_id}
            records = self.collection.get(**get_kwargs)
            metadatas = records.get("metadatas", []) or []
            file_map: Dict[str, str] = {}
            for meta in metadatas:
                fname = meta.get("filename")
                fhash = meta.get("file_hash")
                if fname and fhash and fname not in file_map:
                    file_map[fname] = fhash
            return file_map
        except Exception as e:
            logger.warning(f"Could not retrieve indexed files from ChromaDB: {e}")
            return {}


