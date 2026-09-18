"""
Base parser abstraction for the multi-format document ingestion system.
All format-specific parsers inherit from BaseParser.
"""

from abc import ABC, abstractmethod
from pathlib import Path
from typing import List

from app.models import Document


class BaseParser(ABC):
    """
    Abstract base class for all document format parsers.

    Each concrete parser handles one or more file extensions and
    returns a normalized list of Document objects that feed directly
    into the existing DocumentChunker → Embedding → ChromaDB pipeline.
    """

    #: File extensions this parser handles (lowercase, including the dot)
    SUPPORTED_EXTENSIONS: frozenset = frozenset()

    @abstractmethod
    def parse(
        self,
        file_path: Path,
        original_filename: str,
        file_hash: str,
    ) -> List[Document]:
        """
        Parse the file at *file_path* and return a list of Document objects.

        Args:
            file_path: Absolute path to the file on disk.
            original_filename: The user-facing filename (used in metadata).
            file_hash: Pre-computed SHA-256 hex digest of the file content.

        Returns:
            A list of Document objects, each with:
              - page_content: Extracted text for this logical unit.
              - metadata: At minimum {"source", "filename", "doc_id",
                          "file_hash", "page_number", "total_pages"}.
        """

    def _base_metadata(
        self,
        file_path: Path,
        original_filename: str,
        file_hash: str,
        page_number: int,
        total_pages: int,
        **extra,
    ) -> dict:
        """
        Build the standard metadata dict shared by all parsers.
        Extra keyword arguments are merged in.
        """
        stem = Path(original_filename).stem
        doc_id = f"{stem}_{file_hash[:10]}"
        meta = {
            "source": str(file_path),
            "filename": original_filename,
            "doc_id": doc_id,
            "file_hash": file_hash,
            "page_number": page_number,
            "total_pages": total_pages,
        }
        meta.update(extra)
        return meta
