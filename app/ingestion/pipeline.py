"""
Ingestion Pipeline Module.
Coordinates document loading, cleaning, chunking, embedding, and vector storage.
"""

from pathlib import Path
from typing import Dict, List, Optional

from app.config import settings
from app.embeddings.service import EmbeddingService
from app.ingestion.chunker import DocumentChunker, Chunk
from app.ingestion.loader import PDFLoader, Document
from app.utils.logger import setup_logger
from app.vectorstore.store import VectorStore

logger = setup_logger("ingestion.pipeline")


class IngestionPipeline:
    """End-to-end document ingestion pipeline with idempotency checks."""

    def __init__(
        self,
        vector_store: Optional[VectorStore] = None,
        embedding_service: Optional[EmbeddingService] = None,
        chunker: Optional[DocumentChunker] = None,
    ):
        self.vector_store = vector_store or VectorStore()
        self.embedding_service = embedding_service or EmbeddingService()
        self.chunker = chunker or DocumentChunker()

    def ingest_file(self, file_path: str | Path, force: bool = False) -> Dict[str, int]:
        """
        Ingest a single PDF document file into the vector database.
        Detects if document has changed, purging old chunks and re-indexing if modified.
        """
        path = Path(file_path).resolve()
        logger.info(f"Starting ingestion pipeline for file: {path.name}")

        loader = PDFLoader(path)
        documents = loader.load()

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

    def ingest_directory(self, dir_path: Optional[str | Path] = None, force: bool = False) -> Dict[str, int]:
        """
        Synchronize all PDF files from the documents directory into ChromaDB.
        Automatically cleans up stale chunks from deleted files and re-indexes modified files.
        """
        target_dir = Path(dir_path or settings.documents_dir).resolve()
        if not target_dir.exists():
            raise FileNotFoundError(f"Documents directory does not exist: {target_dir}")

        pdf_files = list(target_dir.glob("*.pdf"))
        logger.info(f"Found {len(pdf_files)} PDF files in {target_dir}")

        # Purge stale chunks from documents that were deleted from disk
        current_filenames = {p.name for p in pdf_files}
        indexed_files = self.vector_store.get_indexed_files()
        for indexed_fname in indexed_files:
            if indexed_fname not in current_filenames:
                logger.info(f"Purging stale chunks for deleted document: {indexed_fname}")
                self.vector_store.delete_by_filename(indexed_fname)

        if not pdf_files:
            logger.warning(f"No PDF files found in {target_dir}")
            return {"documents": 0, "chunks": 0}

        total_docs = 0
        total_chunks = 0

        for pdf_path in pdf_files:
            res = self.ingest_file(pdf_path, force=force)
            total_docs += res.get("documents", 0)
            total_chunks += res.get("chunks", 0)

        return {"documents": total_docs, "chunks": total_chunks}
