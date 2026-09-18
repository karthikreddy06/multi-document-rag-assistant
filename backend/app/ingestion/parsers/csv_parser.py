"""
CSV Parser — extracts tabular data from CSV files using Python's built-in csv module.
The entire file is represented as a header-aware, row-per-line text document.
"""

import csv
import io
from pathlib import Path
from typing import List

from app.ingestion.parsers.base import BaseParser
from app.models import Document
from app.utils.logger import setup_logger

logger = setup_logger("ingestion.parsers.csv")


class CSVParser(BaseParser):
    """Parse .csv files into a single Document with a readable text representation."""

    SUPPORTED_EXTENSIONS: frozenset = frozenset({".csv"})

    def parse(
        self,
        file_path: Path,
        original_filename: str,
        file_hash: str,
    ) -> List[Document]:
        if not file_path.exists():
            raise FileNotFoundError(f"CSV file not found: {file_path}")

        logger.info(f"Parsing CSV: {original_filename}")

        # Try to read as UTF-8, fall back to latin-1 for legacy files
        for encoding in ("utf-8-sig", "utf-8", "latin-1"):
            try:
                raw_content = file_path.read_text(encoding=encoding)
                break
            except UnicodeDecodeError:
                continue
        else:
            raise ValueError(f"Could not decode CSV file '{original_filename}' with any supported encoding.")

        try:
            dialect = csv.Sniffer().sniff(raw_content[:4096], delimiters=",;\t|")
        except csv.Error:
            dialect = csv.excel  # default comma-delimited

        reader = csv.reader(io.StringIO(raw_content), dialect=dialect)
        rows = list(reader)

        if not rows:
            logger.warning(f"CSV file is empty: {original_filename}")
            return []

        # Build a readable representation: header row as section, data rows below
        lines: List[str] = []
        header = rows[0]
        lines.append("Columns: " + ", ".join(str(h) for h in header))
        lines.append("")

        for row_idx, row in enumerate(rows[1:], start=1):
            if not any(cell.strip() for cell in row):
                continue  # skip blank rows
            # Format as "Field: Value" pairs for better RAG retrieval
            pairs = []
            for col, val in zip(header, row):
                if val.strip():
                    pairs.append(f"{col}: {val}")
            if pairs:
                lines.append(f"Row {row_idx}: " + " | ".join(pairs))

        full_text = "\n".join(lines).strip()
        if not full_text:
            logger.warning(f"No data extracted from CSV: {original_filename}")
            return []

        total_rows = len(rows) - 1  # exclude header

        meta = self._base_metadata(
            file_path=file_path,
            original_filename=original_filename,
            file_hash=file_hash,
            page_number=1,
            total_pages=1,
            char_count=len(full_text),
            format="csv",
            row_count=total_rows,
            column_count=len(header),
        )

        logger.info(f"CSVParser: extracted {total_rows} rows from '{original_filename}'")
        return [Document(page_content=full_text, metadata=meta)]
