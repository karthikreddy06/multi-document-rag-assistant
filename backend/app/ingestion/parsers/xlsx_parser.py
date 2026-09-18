"""
XLSX Parser — extracts tabular data from Excel workbooks using openpyxl.
Each worksheet becomes one Document with a human-readable text representation.
"""

from pathlib import Path
from typing import List

from app.ingestion.parsers.base import BaseParser
from app.models import Document
from app.utils.logger import setup_logger

logger = setup_logger("ingestion.parsers.xlsx")


class XLSXParser(BaseParser):
    """Parse .xlsx files; each worksheet becomes a separate Document."""

    SUPPORTED_EXTENSIONS: frozenset = frozenset({".xlsx", ".xls"})

    def parse(
        self,
        file_path: Path,
        original_filename: str,
        file_hash: str,
    ) -> List[Document]:
        try:
            import openpyxl  # openpyxl
        except ImportError as e:
            raise ImportError(
                "openpyxl is required for XLSX parsing. "
                "Install it with: pip install openpyxl"
            ) from e

        if not file_path.exists():
            raise FileNotFoundError(f"XLSX file not found: {file_path}")

        logger.info(f"Parsing XLSX: {original_filename}")

        try:
            wb = openpyxl.load_workbook(str(file_path), read_only=True, data_only=True)
        except Exception as e:
            raise ValueError(f"Could not open XLSX file '{original_filename}': {e}") from e

        sheet_names = wb.sheetnames
        total_sheets = len(sheet_names)
        documents: List[Document] = []

        for sheet_idx, sheet_name in enumerate(sheet_names):
            ws = wb[sheet_name]
            rows_list = []
            for row in ws.iter_rows(values_only=True):
                cell_values = [str(cell).strip() if cell is not None else "" for cell in row]
                if any(cell_values):
                    # Trim trailing empty cells
                    while cell_values and not cell_values[-1]:
                        cell_values.pop()
                    if cell_values:
                        rows_list.append(cell_values)

            if not rows_list:
                logger.debug(f"Sheet '{sheet_name}' in '{original_filename}' is empty, skipping.")
                continue

            # Format sheet text with clear structural headers
            sheet_lines = [f"# Workbook: {original_filename}", f"## Sheet: {sheet_name}", ""]
            
            # Format rows as markdown table if headers exist
            if len(rows_list) >= 1:
                header_row = rows_list[0]
                sheet_lines.append("| " + " | ".join(header_row) + " |")
                sheet_lines.append("| " + " | ".join(["---"] * len(header_row)) + " |")
                for r in rows_list[1:]:
                    # Pad or slice to match header count
                    padded = r + [""] * (len(header_row) - len(r))
                    sheet_lines.append("| " + " | ".join(padded[:len(header_row)]) + " |")
            
            sheet_text = "\n".join(sheet_lines).strip()

            meta = self._base_metadata(
                file_path=file_path,
                original_filename=original_filename,
                file_hash=file_hash,
                page_number=sheet_idx + 1,
                total_pages=total_sheets,
                char_count=len(sheet_text),
                format="xlsx",
                sheet_name=sheet_name,
            )

            documents.append(Document(page_content=sheet_text, metadata=meta))

        wb.close()
        logger.info(f"XLSXParser: extracted {len(documents)} sheets from '{original_filename}'")
        return documents
