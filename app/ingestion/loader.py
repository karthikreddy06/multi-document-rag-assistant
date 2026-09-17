"""
Document Loading Module.
Loads PDF documents safely, extracts text page by page, and attaches comprehensive metadata.
"""

from dataclasses import dataclass, field
from hashlib import sha256
from pathlib import Path
from typing import Dict, List, Optional
from pypdf import PdfReader

from app.models import Document
from app.utils.logger import setup_logger
from app.utils.text_cleaner import clean_extracted_text

logger = setup_logger("ingestion.loader")


class PDFLoader:
    """Production-ready PDF loader with metadata extraction and error handling."""

    def __init__(self, file_path: str | Path):
        self.file_path = Path(file_path).resolve()

    def _compute_file_hash(self) -> str:
        """Compute SHA256 hash of the document file."""
        hasher = sha256()
        with open(self.file_path, "rb") as f:
            for chunk in iter(lambda: f.read(65536), b""):
                hasher.update(chunk)
        return hasher.hexdigest()

    def load(self) -> List[Document]:
        """
        Load and extract text from the PDF document.
        Returns a list of Document objects, one for each non-empty page.
        """
        if not self.file_path.exists():
            raise FileNotFoundError(f"Document file not found at: {self.file_path}")

        if not self.file_path.is_file():
            raise ValueError(f"Path is not a valid file: {self.file_path}")

        logger.info(f"Loading PDF from: {self.file_path}")

        try:
            reader = PdfReader(str(self.file_path))
        except Exception as e:
            logger.error(f"Failed to open PDF file {self.file_path}: {e}")
            raise ValueError(f"Corrupt or invalid PDF file: {e}") from e

        total_pages = len(reader.pages)
        if total_pages == 0:
            logger.warning(f"PDF file has 0 pages: {self.file_path}")
            return []

        doc_hash = self._compute_file_hash()
        doc_id = f"{self.file_path.stem}_{doc_hash[:10]}"

        documents: List[Document] = []

        for page_idx, page in enumerate(reader.pages):
            try:
                raw_text = page.extract_text() or ""
            except Exception as e:
                logger.warning(f"Error extracting text from page {page_idx + 1} in {self.file_path}: {e}")
                raw_text = ""

            cleaned_text = clean_extracted_text(raw_text)

            if not cleaned_text:
                logger.warning(f"Page {page_idx + 1} of {self.file_path.name} is empty. Skipping.")
                continue

            metadata = {
                "source": str(self.file_path),
                "filename": self.file_path.name,
                "doc_id": doc_id,
                "file_hash": doc_hash,
                "page_number": page_idx + 1,
                "total_pages": total_pages,
            }

            documents.append(Document(page_content=cleaned_text, metadata=metadata))

        logger.info(f"Successfully loaded {len(documents)} non-empty pages from {self.file_path.name}")
        return documents
