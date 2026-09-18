"""
Ingestion Pipeline Module.
Coordinates document loading, cleaning, chunking, embedding, and vector storage.

Now supports multi-format documents (PDF, DOCX, PPTX, XLSX, CSV, TXT, MD, Images)
via the ParserRegistry abstraction. All existing retrieval, generation, and
citation behavior is fully preserved.
"""

import hashlib
from pathlib import Path
from typing import Any, Dict, List, Optional

from app.config import settings
from app.embeddings.service import EmbeddingService
from app.ingestion.chunker import DocumentChunker, Chunk
from app.ingestion.loader import PDFLoader, Document, FileTypeDetector
from app.ingestion.parsers import ParserRegistry, UnsupportedFileTypeError
from app.utils.logger import setup_logger
from app.vectorstore.store import VectorStore

logger = setup_logger("ingestion.pipeline")

# Singleton registry shared across all pipeline instances (thread-safe reads)
_parser_registry = ParserRegistry()


class IngestionPipeline:
    """End-to-end multi-format document ingestion pipeline with idempotency checks."""

    def __init__(
        self,
        vector_store: Optional[VectorStore] = None,
        embedding_service: Optional[EmbeddingService] = None,
        chunker: Optional[DocumentChunker] = None,
        parser_registry: Optional[ParserRegistry] = None,
    ):
        self.vector_store = vector_store or VectorStore()
        self.embedding_service = embedding_service or EmbeddingService()
        self.chunker = chunker or DocumentChunker()
        self.parser_registry = parser_registry or _parser_registry

    def ingest_file(self, file_path: str | Path, force: bool = False) -> Dict[str, int]:
        """
        Ingest a single document file into the vector database.

        Supports all formats registered in ParserRegistry (PDF, DOCX, PPTX, XLSX,
        CSV, TXT, MD, Images). Detects if the document has changed, purging old
        chunks and re-indexing if modified.
        """
        path = Path(file_path).resolve()
        logger.info(f"Starting ingestion pipeline for file: {path.name}")

        # Pre-compute file hash for idempotency checks (parsers store it in metadata)
        try:
            hasher = hashlib.sha256()
            with open(path, "rb") as fh:
                for block in iter(lambda: fh.read(65536), b""):
                    hasher.update(block)
            computed_hash = hasher.hexdigest()
        except OSError as e:
            logger.error(f"Cannot read file for hashing: {path.name}: {e}")
            return {"documents": 0, "chunks": 0}

        try:
            documents = self.parser_registry.parse(
                file_path=path,
                original_filename=path.name,
                file_hash=computed_hash,
            )
        except UnsupportedFileTypeError as e:
            logger.warning(f"Unsupported file type for {path.name}: {e}")
            return {"documents": 0, "chunks": 0, "status": "unsupported"}

        if not documents:
            logger.warning(f"No text extracted from {path.name}. Nothing to ingest.")
            return {"documents": 0, "chunks": 0}

        doc_hash = str(documents[0].metadata.get("file_hash", ""))
        indexed_files = self.vector_store.get_indexed_files()

        # Check if already ingested with identical content hash
        if not force and path.name in indexed_files:
            if indexed_files[path.name] == doc_hash:
                logger.info(f"Document {path.name} (hash={doc_hash[:8]}) is up to date. Skipping re-embedding.")
                return {"documents": len(documents), "chunks": 0, "status": "already_indexed"}
            else:
                logger.info(f"Document {path.name} has changed on disk. Purging stale chunks before re-indexing...")
                self.vector_store.delete_by_filename(path.name)
        elif force and path.name in indexed_files:
            logger.info(f"Force reindexing: purging existing chunks for {path.name}...")
            self.vector_store.delete_by_filename(path.name)

        chunks = self.chunker.split_documents(documents)
        if not chunks:
            logger.warning(f"No chunks generated from {path.name}.")
            return {"documents": len(documents), "chunks": 0}

        # Generate embeddings
        texts = [c.text for c in chunks]
        embeddings = self.embedding_service.embed_batch(texts)

        # Store in ChromaDB
        stored_count = self.vector_store.upsert_chunks(chunks, embeddings)

        logger.info(f"Successfully finished ingestion for {path.name}: {stored_count} chunks indexed.")
        return {"documents": len(documents), "chunks": stored_count, "status": "indexed"}

    def ingest_uploaded_document(
        self,
        file_path: str | Path,
        original_filename: str,
        document_id: str,
        file_hash: Optional[str] = None,
    ) -> Dict[str, Any]:
        """
        Ingest an uploaded document of any supported format, preserving original
        filename and document_id metadata.
        """
        path = Path(file_path).resolve()

        try:
            documents = self.parser_registry.parse(
                file_path=path,
                original_filename=original_filename,
                file_hash=file_hash or "",
            )
        except UnsupportedFileTypeError as e:
            logger.warning(f"Unsupported file type for '{original_filename}': {e}")
            raise

        if not documents:
            logger.warning(f"No text extracted from uploaded file {original_filename}.")
            return {"page_count": 0, "chunk_count": 0}

        resolved_hash = file_hash or str(documents[0].metadata.get("file_hash", ""))
        for doc in documents:
            doc.metadata["filename"] = original_filename
            doc.metadata["document_id"] = document_id
            doc.metadata["file_hash"] = resolved_hash

        chunks = self.chunker.split_documents(documents)
        if not chunks:
            logger.warning(f"No chunks generated from uploaded file {original_filename}.")
            return {"page_count": len(documents), "chunk_count": 0}

        # Ensure every chunk carries document_id and clean original_filename
        for c in chunks:
            c.metadata["document_id"] = document_id
            c.metadata["filename"] = original_filename
            c.metadata["file_hash"] = resolved_hash

        texts = [c.text for c in chunks]
        embeddings = self.embedding_service.embed_batch(texts)
        stored_count = self.vector_store.upsert_chunks(chunks, embeddings)

        logger.info(
            f"Successfully ingested uploaded document '{original_filename}' ({document_id}): "
            f"{len(documents)} page(s)/sheet(s)/slide(s), {stored_count} chunks indexed."
        )
        return {"page_count": len(documents), "chunk_count": stored_count}

    def ingest_directory(self, dir_path: Optional[str | Path] = None, force: bool = False) -> Dict[str, int]:
        """
        Synchronize all supported documents from the documents directory into ChromaDB.
        Automatically cleans up stale chunks from deleted files and re-indexes modified files.

        Supports all extensions registered in ParserRegistry, not just PDFs.
        """
        # Resolve relative paths against the backend directory
        path_obj = Path(dir_path or settings.documents_dir)
        if not path_obj.is_absolute():
            backend_dir = Path(__file__).resolve().parent.parent.parent
            path_obj = backend_dir / path_obj
        target_dir = path_obj.resolve()
        if not target_dir.exists():
            raise FileNotFoundError(f"Documents directory does not exist: {target_dir}")

        # Collect all files with supported extensions
        supported = self.parser_registry.supported_extensions
        all_files: List[Path] = []
        for ext in supported:
            all_files.extend(target_dir.glob(f"*{ext}"))

        # Deduplicate (a file could match multiple glob patterns in theory)
        all_files = sorted(set(all_files))
        logger.info(f"Found {len(all_files)} supported document(s) in {target_dir}")

        # Purge stale chunks from documents that were deleted from disk (protecting user uploads)
        current_filenames = {p.name for p in all_files}
        try:
            raw_meta = self.vector_store.collection.get(include=["metadatas"])
            uploaded_filenames = {
                m.get("filename")
                for m in (raw_meta.get("metadatas") or [])
                if m and m.get("document_id")
            }
        except Exception:
            uploaded_filenames = set()

        indexed_files = self.vector_store.get_indexed_files()
        for indexed_fname in indexed_files:
            if indexed_fname not in current_filenames and indexed_fname not in uploaded_filenames:
                logger.info(f"Purging stale chunks for deleted document: {indexed_fname}")
                self.vector_store.delete_by_filename(indexed_fname)

        if not all_files:
            logger.warning(f"No supported document files found in {target_dir}")
            return {"documents": 0, "chunks": 0}

        total_docs = 0
        total_chunks = 0

        for doc_path in all_files:
            res = self.ingest_file(doc_path, force=force)
            total_docs += res.get("documents", 0)
            total_chunks += res.get("chunks", 0)

        return {"documents": total_docs, "chunks": total_chunks}
