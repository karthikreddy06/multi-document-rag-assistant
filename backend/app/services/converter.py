"""
File Conversion Service Module.
Supports practical bi-directional and cross-format conversions:
1. CSV -> XLSX
2. XLSX -> CSV
3. TXT -> DOCX
4. Markdown -> DOCX
5. TXT -> PDF
6. Markdown -> PDF
7. PDF -> DOCX
8. DOCX -> PDF (gracefully degraded with LibreOffice headless detection)
"""

import csv
import io
import os
from pathlib import Path
import re
import shutil
import subprocess
import tempfile
from typing import Dict, List, Optional, Set, Tuple

import openpyxl
from docx import Document as DocxDocument
from reportlab.lib.pagesizes import letter
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Preformatted
from reportlab.lib import colors

from app.utils.logger import setup_logger

logger = setup_logger("services.converter")


class ConversionError(Exception):
    """Base error for document conversion failures."""
    pass


class UnsupportedConversionError(ConversionError):
    """Raised when the requested format pair is not supported or missing system tool."""
    pass


# Mapping of supported conversions: source_ext -> set of target_exts
SUPPORTED_CONVERSIONS: Dict[str, Set[str]] = {
    ".csv": {".xlsx"},
    ".xlsx": {".csv"},
    ".txt": {".docx", ".pdf"},
    ".md": {".docx", ".pdf"},
    ".markdown": {".docx", ".pdf"},
    ".pdf": {".docx"},
    ".docx": {".pdf"},
}

MIME_TYPES: Dict[str, str] = {
    ".xlsx": "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
    ".csv": "text/csv",
    ".docx": "application/vnd.openxmlformats-officedocument.wordprocessingml.document",
    ".pdf": "application/pdf",
    ".txt": "text/plain",
    ".md": "text/markdown",
}


def is_conversion_supported(source_ext: str, target_ext: str) -> bool:
    """Check if conversion from source_ext to target_ext is supported."""
    src = source_ext.lower()
    tgt = target_ext.lower()
    if not src.startswith("."):
        src = f".{src}"
    if not tgt.startswith("."):
        tgt = f".{tgt}"
    return tgt in SUPPORTED_CONVERSIONS.get(src, set())


def get_supported_targets(source_ext: str) -> List[str]:
    """Get list of supported target extensions for a given source extension."""
    src = source_ext.lower()
    if not src.startswith("."):
        src = f".{src}"
    return sorted(list(SUPPORTED_CONVERSIONS.get(src, set())))


def detect_libreoffice() -> Optional[str]:
    """Detect if LibreOffice / soffice is available in the system PATH."""
    for cmd in ("soffice", "libreoffice", "soffice.bin"):
        path = shutil.which(cmd)
        if path:
            return path
    return None


class DocumentConverter:
    """Production service converting documents across business and document formats."""

    @classmethod
    def convert(
        cls,
        source_bytes: bytes,
        source_filename: str,
        target_format: str,
    ) -> Tuple[bytes, str, str]:
        """
        Convert source_bytes into target_format.
        Returns: (converted_bytes, output_filename, mime_type)
        """
        src_path = Path(source_filename)
        src_ext = src_path.suffix.lower()
        tgt_ext = target_format.lower()
        if not tgt_ext.startswith("."):
            tgt_ext = f".{tgt_ext}"

        if not is_conversion_supported(src_ext, tgt_ext):
            supported = get_supported_targets(src_ext)
            raise UnsupportedConversionError(
                f"Conversion from '{src_ext}' to '{tgt_ext}' is not supported. "
                f"Supported targets for '{src_ext}': {supported}"
            )

        output_filename = f"{src_path.stem}{tgt_ext}"
        mime_type = MIME_TYPES.get(tgt_ext, "application/octet-stream")

        try:
            if src_ext == ".csv" and tgt_ext == ".xlsx":
                out_bytes = cls._csv_to_xlsx(source_bytes)
            elif src_ext == ".xlsx" and tgt_ext == ".csv":
                out_bytes = cls._xlsx_to_csv(source_bytes)
            elif src_ext in (".txt",) and tgt_ext == ".docx":
                out_bytes = cls._txt_to_docx(source_bytes)
            elif src_ext in (".md", ".markdown") and tgt_ext == ".docx":
                out_bytes = cls._md_to_docx(source_bytes)
            elif src_ext in (".txt",) and tgt_ext == ".pdf":
                out_bytes = cls._txt_to_pdf(source_bytes)
            elif src_ext in (".md", ".markdown") and tgt_ext == ".pdf":
                out_bytes = cls._md_to_pdf(source_bytes)
            elif src_ext == ".pdf" and tgt_ext == ".docx":
                out_bytes = cls._pdf_to_docx(source_bytes)
            elif src_ext == ".docx" and tgt_ext == ".pdf":
                out_bytes = cls._docx_to_pdf(source_bytes)
            else:
                raise UnsupportedConversionError(f"Handler not implemented for {src_ext} -> {tgt_ext}")

            return out_bytes, output_filename, mime_type

        except UnsupportedConversionError:
            raise
        except Exception as e:
            logger.error(f"Conversion error for {source_filename} to {target_format}: {e}")
            raise ConversionError(f"Failed to convert {source_filename} to {target_format}: {e}") from e

    # --------------------------------------------------------------------------
    # Handlers
    # --------------------------------------------------------------------------
    @staticmethod
    def _csv_to_xlsx(data: bytes) -> bytes:
        text = data.decode("utf-8", errors="replace")
        wb = openpyxl.Workbook()
        ws = wb.active
        ws.title = "Data"

        reader = csv.reader(io.StringIO(text))
        for row in reader:
            ws.append(row)

        out_buf = io.BytesIO()
        wb.save(out_buf)
        return out_buf.getvalue()

    @staticmethod
    def _xlsx_to_csv(data: bytes) -> bytes:
        wb = openpyxl.load_workbook(filename=io.BytesIO(data), data_only=True)
        ws = wb.active
        out_buf = io.StringIO()
        writer = csv.writer(out_buf)

        for row in ws.iter_rows(values_only=True):
            # Clean None values to empty strings
            cleaned_row = ["" if cell is None else str(cell) for cell in row]
            if any(cleaned_row):
                writer.writerow(cleaned_row)

        return out_buf.getvalue().encode("utf-8")

    @staticmethod
    def _txt_to_docx(data: bytes) -> bytes:
        text = data.decode("utf-8", errors="replace")
        doc = DocxDocument()
        for line in text.splitlines():
            doc.add_paragraph(line)
        out_buf = io.BytesIO()
        doc.save(out_buf)
        return out_buf.getvalue()

    @staticmethod
    def _md_to_docx(data: bytes) -> bytes:
        text = data.decode("utf-8", errors="replace")
        doc = DocxDocument()
        for line in text.splitlines():
            stripped = line.strip()
            if stripped.startswith("### "):
                doc.add_heading(stripped[4:], level=3)
            elif stripped.startswith("## "):
                doc.add_heading(stripped[3:], level=2)
            elif stripped.startswith("# "):
                doc.add_heading(stripped[2:], level=1)
            elif stripped.startswith(("- ", "* ", "• ")):
                doc.add_paragraph(stripped[2:], style="List Bullet")
            else:
                doc.add_paragraph(line)
        out_buf = io.BytesIO()
        doc.save(out_buf)
        return out_buf.getvalue()

    @staticmethod
    def _txt_to_pdf(data: bytes) -> bytes:
        text = data.decode("utf-8", errors="replace")
        out_buf = io.BytesIO()
        doc = SimpleDocTemplate(out_buf, pagesize=letter, rightMargin=54, leftMargin=54, topMargin=54, bottomMargin=54)
        styles = getSampleStyleSheet()
        normal = styles["Normal"]
        normal.fontSize = 10
        normal.leading = 14

        story = []
        for line in text.splitlines():
            clean = line.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")
            if not clean.strip():
                story.append(Spacer(1, 10))
            else:
                story.append(Paragraph(clean, normal))

        doc.build(story)
        return out_buf.getvalue()

    @staticmethod
    def _md_to_pdf(data: bytes) -> bytes:
        text = data.decode("utf-8", errors="replace")
        out_buf = io.BytesIO()
        doc = SimpleDocTemplate(out_buf, pagesize=letter, rightMargin=54, leftMargin=54, topMargin=54, bottomMargin=54)
        styles = getSampleStyleSheet()

        h1 = styles["Heading1"]
        h2 = styles["Heading2"]
        h3 = styles["Heading3"]
        body = styles["Normal"]
        bullet = styles["Bullet"]

        story = []
        for line in text.splitlines():
            s = line.strip()
            if not s:
                story.append(Spacer(1, 8))
                continue
            clean = s.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")
            # Simple bold **text** to <b>text</b>
            clean = re.sub(r"\*\*(.+?)\*\*", r"<b>\1</b>", clean)

            if clean.startswith("### "):
                story.append(Paragraph(clean[4:], h3))
            elif clean.startswith("## "):
                story.append(Paragraph(clean[3:], h2))
            elif clean.startswith("# "):
                story.append(Paragraph(clean[2:], h1))
            elif clean.startswith(("- ", "* ", "• ")):
                story.append(Paragraph(clean[2:], bullet))
            else:
                story.append(Paragraph(clean, body))

        doc.build(story)
        return out_buf.getvalue()

    @staticmethod
    def _pdf_to_docx(data: bytes) -> bytes:
        from pdf2docx import Converter

        with tempfile.TemporaryDirectory() as tmp_dir:
            pdf_path = os.path.join(tmp_dir, "input.pdf")
            docx_path = os.path.join(tmp_dir, "output.docx")
            with open(pdf_path, "wb") as f:
                f.write(data)

            cv = Converter(pdf_path)
            cv.convert(docx_path, start=0, end=None)
            cv.close()

            with open(docx_path, "rb") as f:
                return f.read()

    @classmethod
    def _docx_to_pdf(cls, data: bytes) -> bytes:
        soffice = detect_libreoffice()
        if not soffice:
            raise UnsupportedConversionError(
                "DOCX to PDF conversion requires LibreOffice, which is not available in the current environment."
            )

        with tempfile.TemporaryDirectory() as tmp_dir:
            docx_path = os.path.join(tmp_dir, "input.docx")
            with open(docx_path, "wb") as f:
                f.write(data)

            cmd = [soffice, "--headless", "--convert-to", "pdf", "--outdir", tmp_dir, docx_path]
            try:
                res = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, timeout=60, check=True)
            except Exception as e:
                raise ConversionError(f"LibreOffice conversion failed: {e}") from e

            pdf_path = os.path.join(tmp_dir, "input.pdf")
            if not os.path.exists(pdf_path):
                raise ConversionError("LibreOffice completed without producing output PDF.")

            with open(pdf_path, "rb") as f:
                return f.read()
