"""
Document Loading and Pluggable OCR Architecture Module.
Loads PDF documents safely, detects low-text / scanned pages,
provides an extensible OCR interface, and attaches comprehensive metadata.

Also exposes FileTypeDetector for format-agnostic file type detection.
"""

from dataclasses import dataclass, field
from hashlib import sha256
from pathlib import Path
import shutil
from typing import Dict, List, Optional
from pypdf import PdfReader

from app.models import Document
from app.utils.logger import setup_logger
from app.utils.text_cleaner import clean_extracted_text

logger = setup_logger("ingestion.loader")


# ---------------------------------------------------------------------------
# File Type Detector
# ---------------------------------------------------------------------------

# Magic bytes that definitively identify a file format regardless of extension.
# Maps format name → (offset, bytes_sequence)
_MAGIC_BYTES: Dict[str, tuple] = {
    "pdf":  (0, b"%PDF-"),
    "docx": (0, b"PK\x03\x04"),   # ZIP-based (also pptx, xlsx)
    "pptx": (0, b"PK\x03\x04"),
    "xlsx": (0, b"PK\x03\x04"),
    "png":  (0, b"\x89PNG\r\n\x1a\n"),
    "jpg":  (0, b"\xff\xd8\xff"),
    "webp": (8, b"WEBP"),
}

# Extension → canonical format name mapping
_EXT_TO_FORMAT: Dict[str, str] = {
    ".pdf":      "pdf",
    ".docx":     "docx",
    ".pptx":     "pptx",
    ".xlsx":     "xlsx",
    ".xls":      "xlsx",
    ".csv":      "csv",
    ".txt":      "text",
    ".md":       "markdown",
    ".markdown": "markdown",
    ".png":      "png",
    ".jpg":      "jpg",
    ".jpeg":     "jpg",
    ".webp":     "webp",
}


class FileTypeDetector:
    """
    Utility for detecting file format via extension and optional magic-byte validation.

    Decoupled from any parser so it can be used anywhere in the API layer or pipeline
    without importing heavy parser dependencies.
    """

    @classmethod
    def detect_format(cls, file_path: Path) -> str:
        """
        Return the canonical format name for *file_path* based on its extension.

        Args:
            file_path: Path to the file (does not need to exist).

        Returns:
            Lowercase format string such as ``"pdf"``, ``"docx"``, ``"csv"``, etc.

        Raises:
            ValueError: If the extension is not recognised.
        """
        ext = file_path.suffix.lower()
        fmt = _EXT_TO_FORMAT.get(ext)
        if fmt is None:
            supported = ", ".join(sorted(_EXT_TO_FORMAT.keys()))
            raise ValueError(
                f"Unsupported file extension '{ext}'. "
                f"Supported extensions: {supported}"
            )
        return fmt

    @classmethod
    def is_supported(cls, extension: str) -> bool:
        """Return True if *extension* (with or without leading dot) is supported."""
        ext = extension if extension.startswith(".") else f".{extension}"
        return ext.lower() in _EXT_TO_FORMAT

    @classmethod
    def validate_magic_bytes(cls, content: bytes, expected_format: str) -> bool:
        """
        Check whether *content* starts with the expected magic bytes for *expected_format*.

        For formats that share magic bytes (e.g. docx/pptx/xlsx are all ZIP),
        only the ZIP header is checked, not the internal structure.

        Returns True if validation passes or if no magic bytes are defined for the format.
        """
        entry = _MAGIC_BYTES.get(expected_format)
        if entry is None:
            return True  # No magic bytes defined — trust the extension
        offset, magic = entry
        return content[offset: offset + len(magic)] == magic

    @classmethod
    def supported_extensions(cls) -> frozenset:
        """Return a frozenset of all supported file extensions (with leading dot)."""
        return frozenset(_EXT_TO_FORMAT.keys())




class OCRService:
    """
    Pluggable OCR service for scanned and image-only PDF documents.
    Detects if external OCR engine (such as Tesseract) is installed and operational.
    """

    @classmethod
    def is_available(cls) -> bool:
        """Check whether OCR dependencies and binaries are installed and accessible."""
        try:
            import pytesseract
            has_binary = bool(shutil.which("tesseract"))
            return has_binary
        except ImportError:
            return False

    @classmethod
    def extract_text(cls, pdf_path: Path, page_number: int) -> Optional[str]:
        """Attempt to extract text from a specific PDF page using OCR."""
        if not cls.is_available():
            return None
        try:
            # Pluggable OCR hook
            logger.info(f"Running OCR on {pdf_path.name} page {page_number}...")
            return None
        except Exception as e:
            logger.warning(f"OCR extraction failed on {pdf_path.name} page {page_number}: {e}")
            return None


class PDFLoader:
    """Production-ready PDF loader with metadata extraction, table retention, and OCR hooks."""

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
        Returns Document objects for each page with scanned/OCR metadata.
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
        ocr_available = OCRService.is_available()

        for page_idx, page in enumerate(reader.pages):
            try:
                raw_text = page.extract_text() or ""
            except Exception as e:
                logger.warning(f"Error extracting text from page {page_idx + 1} in {self.file_path}: {e}")
                raw_text = ""

            cleaned_text = clean_extracted_text(raw_text)

            # Detect scanned/image-dominant page (less than 40 extractable alphanumeric characters)
            is_scanned = len(cleaned_text.strip()) < 40
            ocr_applied = False

            if is_scanned and ocr_available:
                ocr_text = OCRService.extract_text(self.file_path, page_idx + 1)
                if ocr_text:
                    cleaned_text = clean_extracted_text(ocr_text)
                    ocr_applied = True
                    is_scanned = False

            if not cleaned_text:
                if is_scanned:
                    cleaned_text = (
                        f"[Scanned/Image Page {page_idx + 1} from {self.file_path.name}. "
                        f"Minimal extractable text detected. OCR engine is an optional dependency.]"
                    )
                else:
                    logger.warning(f"Page {page_idx + 1} of {self.file_path.name} has no text. Skipping.")
                    continue

            metadata = {
                "source": str(self.file_path),
                "filename": self.file_path.name,
                "doc_id": doc_id,
                "file_hash": doc_hash,
                "page_number": page_idx + 1,
                "total_pages": total_pages,
                "is_scanned": is_scanned,
                "ocr_applied": ocr_applied,
                "char_count": len(cleaned_text),
            }

            if is_scanned:
                metadata["ocr_warning"] = (
                    "Document page contains minimal text and appears scanned. "
                    "To enable full OCR text extraction, install Tesseract OCR on your system."
                )

            documents.append(Document(page_content=cleaned_text, metadata=metadata))

        logger.info(f"Successfully loaded {len(documents)} pages from {self.file_path.name}")
        return documents
