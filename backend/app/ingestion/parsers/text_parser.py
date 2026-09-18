"""
Text and Markdown Parser — handles plain text (.txt) and Markdown (.md) files.
For .md files the raw markdown is preserved so the DocumentChunker's heading
detection can split the content into meaningful sections.
"""

from pathlib import Path
from typing import List

from app.ingestion.parsers.base import BaseParser
from app.models import Document
from app.utils.logger import setup_logger

logger = setup_logger("ingestion.parsers.text")


class TextParser(BaseParser):
    """Parse plain text and Markdown files into a single Document."""

    SUPPORTED_EXTENSIONS: frozenset = frozenset({".txt", ".md", ".markdown"})

    def parse(
        self,
        file_path: Path,
        original_filename: str,
        file_hash: str,
    ) -> List[Document]:
        if not file_path.exists():
            raise FileNotFoundError(f"Text file not found: {file_path}")

        logger.info(f"Parsing text/markdown: {original_filename}")

        # Try UTF-8 first (standard for .md), fall back to latin-1
        for encoding in ("utf-8-sig", "utf-8", "latin-1"):
            try:
                content = file_path.read_text(encoding=encoding)
                break
            except UnicodeDecodeError:
                continue
        else:
            raise ValueError(f"Could not decode text file '{original_filename}' with any supported encoding.")

        content = content.strip()
        if not content:
            logger.warning(f"Text file is empty: {original_filename}")
            return []

        ext = file_path.suffix.lower()
        fmt = "markdown" if ext in (".md", ".markdown") else "text"

        meta = self._base_metadata(
            file_path=file_path,
            original_filename=original_filename,
            file_hash=file_hash,
            page_number=1,
            total_pages=1,
            char_count=len(content),
            format=fmt,
        )

        logger.info(f"TextParser: extracted {len(content)} chars from '{original_filename}'")
        return [Document(page_content=content, metadata=meta)]
