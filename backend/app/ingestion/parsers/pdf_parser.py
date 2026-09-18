"""
PDF Parser — wraps the existing PDFLoader logic into the BaseParser interface.
Preserves all existing PDF extraction, OCR detection, and metadata behavior.
"""

from pathlib import Path
from typing import List

from app.ingestion.parsers.base import BaseParser
from app.models import Document
from app.utils.logger import setup_logger
from app.utils.text_cleaner import clean_extracted_text

logger = setup_logger("ingestion.parsers.pdf")


class PDFParser(BaseParser):
    """Parse PDF documents page-by-page using pypdf with OCR fallback hooks."""

    SUPPORTED_EXTENSIONS: frozenset = frozenset({".pdf"})

    def parse(
        self,
        file_path: Path,
        original_filename: str,
        file_hash: str,
    ) -> List[Document]:
        from pypdf import PdfReader

        if not file_path.exists():
            raise FileNotFoundError(f"PDF file not found: {file_path}")

        logger.info(f"Parsing PDF: {original_filename}")

        try:
            reader = PdfReader(str(file_path))
        except Exception as e:
            raise ValueError(f"Corrupt or invalid PDF file '{original_filename}': {e}") from e

        total_pages = len(reader.pages)
        if total_pages == 0:
            logger.warning(f"PDF has 0 pages: {original_filename}")
            return []

        documents: List[Document] = []

        for page_idx, page in enumerate(reader.pages):
            try:
                raw_text = page.extract_text() or ""
            except Exception as e:
                logger.warning(f"Error extracting text from page {page_idx + 1} in {original_filename}: {e}")
                raw_text = ""

            cleaned_text = clean_extracted_text(raw_text)

            is_scanned = len(cleaned_text.strip()) < 40
            ocr_applied = False

            if not cleaned_text:
                if is_scanned:
                    cleaned_text = (
                        f"[Scanned/Image Page {page_idx + 1} from {original_filename}. "
                        f"Minimal extractable text detected. OCR engine is an optional dependency.]"
                    )
                else:
                    logger.warning(f"Page {page_idx + 1} of {original_filename} has no text. Skipping.")
                    continue

            meta = self._base_metadata(
                file_path=file_path,
                original_filename=original_filename,
                file_hash=file_hash,
                page_number=page_idx + 1,
                total_pages=total_pages,
                is_scanned=is_scanned,
                ocr_applied=ocr_applied,
                char_count=len(cleaned_text),
                format="pdf",
            )

            if is_scanned:
                meta["ocr_warning"] = (
                    "Document page contains minimal text and appears scanned. "
                    "To enable full OCR text extraction, install Tesseract OCR on your system."
                )

            documents.append(Document(page_content=cleaned_text, metadata=meta))

        logger.info(f"PDFParser: extracted {len(documents)} pages from '{original_filename}'")
        return documents
