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
        except Exception as e:
            logger.error(f"Failed to initialize ChromaDB at '{self.persist_path}': {e}")
            raise RuntimeError(f"ChromaDB initialization error: {e}") from e

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

    def delete_by_filename(self, filename: str) -> int:
        """
        Delete all chunks associated with a specific document filename.
        Returns the count of chunks removed.
        """
        if not filename:
            return 0
        try:
            records = self.collection.get(where={"filename": filename})
            ids_to_delete = records.get("ids", [])
            if ids_to_delete:
                self.collection.delete(ids=ids_to_delete)
                logger.info(f"Deleted {len(ids_to_delete)} stale chunks for file: {filename}")
                return len(ids_to_delete)
            return 0
        except Exception as e:
            logger.error(f"Failed deleting chunks for {filename}: {e}")
            raise RuntimeError(f"ChromaDB delete failure: {e}") from e

    def get_indexed_files(self) -> Dict[str, str]:
        """
        Return a mapping of {filename: file_hash} for all currently indexed documents.
        Uses metadata-only retrieval to avoid loading document texts into memory.
        """
        try:
            records = self.collection.get(include=["metadatas"])
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

