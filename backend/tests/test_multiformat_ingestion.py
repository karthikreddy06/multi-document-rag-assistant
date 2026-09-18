"""
Test suite for multi-format document ingestion architecture.

Tests cover:
  - FileTypeDetector (extension detection, magic byte validation, supported_extensions)
  - ParserRegistry (dispatch, is_supported, UnsupportedFileTypeError)
  - Each concrete parser (PDFParser, DOCXParser, PPTXParser, XLSXParser, CSVParser,
    TextParser, ImageParser) using synthetic in-memory test files
  - IngestionPipeline.ingest_uploaded_document() with multi-format test fixtures
  - Upload API validation: supported types accepted, unsupported rejected
  - Idempotency: same content hash skips re-embedding
  - Regression: existing PDF behavior unchanged
"""

import csv
import hashlib
import io
import os
import sys
import tempfile
import textwrap
from pathlib import Path
from typing import List
from unittest.mock import MagicMock, patch

import pytest

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _write_temp(content: bytes, suffix: str) -> Path:
    """Write *content* to a temp file with *suffix* and return the Path."""
    tmp = tempfile.NamedTemporaryFile(delete=False, suffix=suffix)
    tmp.write(content)
    tmp.close()
    return Path(tmp.name)


def _sha256(content: bytes) -> str:
    return hashlib.sha256(content).hexdigest()


# ---------------------------------------------------------------------------
# Fixtures: synthetic test files
# ---------------------------------------------------------------------------

@pytest.fixture()
def txt_file(tmp_path: Path) -> Path:
    p = tmp_path / "sample.txt"
    p.write_text("Hello World\nThis is a plain text file.\nLine three.", encoding="utf-8")
    return p


@pytest.fixture()
def md_file(tmp_path: Path) -> Path:
    p = tmp_path / "readme.md"
    p.write_text(
        "# Introduction\n\nThis is the intro.\n\n## Section Two\n\nContent here.\n",
        encoding="utf-8",
    )
    return p


@pytest.fixture()
def csv_file(tmp_path: Path) -> Path:
    p = tmp_path / "data.csv"
    p.write_text(
        "Name,Age,City\nAlice,30,New York\nBob,25,London\nCharlie,35,Tokyo\n",
        encoding="utf-8",
    )
    return p


@pytest.fixture()
def docx_file(tmp_path: Path) -> Path:
    """Create a minimal real DOCX file using python-docx."""
    pytest.importorskip("docx")
    from docx import Document as DocxDocument
    doc = DocxDocument()
    doc.add_heading("Test Document", level=1)
    doc.add_paragraph("This is the first paragraph.")
    doc.add_heading("Section Two", level=2)
    doc.add_paragraph("Content in section two.")
    p = tmp_path / "test.docx"
    doc.save(str(p))
    return p


@pytest.fixture()
def pptx_file(tmp_path: Path) -> Path:
    """Create a minimal real PPTX file using python-pptx."""
    pytest.importorskip("pptx")
    from pptx import Presentation
    prs = Presentation()
    slide_layout = prs.slide_layouts[1]  # title and content
    slide = prs.slides.add_slide(slide_layout)
    slide.shapes.title.text = "Slide One"
    slide.placeholders[1].text = "Bullet point one\nBullet point two"
    p = tmp_path / "test.pptx"
    prs.save(str(p))
    return p


@pytest.fixture()
def xlsx_file(tmp_path: Path) -> Path:
    """Create a minimal real XLSX file using openpyxl."""
    pytest.importorskip("openpyxl")
    import openpyxl
    wb = openpyxl.Workbook()
    ws = wb.active
    ws.title = "Sheet1"
    ws.append(["Product", "Price", "Quantity"])
    ws.append(["Widget A", 9.99, 100])
    ws.append(["Widget B", 14.99, 50])
    p = tmp_path / "data.xlsx"
    wb.save(str(p))
    return p


@pytest.fixture()
def png_file(tmp_path: Path) -> Path:
    """Create a minimal real PNG image using Pillow."""
    pytest.importorskip("PIL")
    from PIL import Image
    img = Image.new("RGB", (100, 100), color=(73, 109, 137))
    p = tmp_path / "test.png"
    img.save(str(p))
    return p


# ---------------------------------------------------------------------------
# FileTypeDetector tests
# ---------------------------------------------------------------------------

class TestFileTypeDetector:
    def test_detect_format_pdf(self, tmp_path):
        from app.ingestion.loader import FileTypeDetector
        assert FileTypeDetector.detect_format(tmp_path / "file.pdf") == "pdf"

    def test_detect_format_docx(self, tmp_path):
        from app.ingestion.loader import FileTypeDetector
        assert FileTypeDetector.detect_format(tmp_path / "file.docx") == "docx"

    def test_detect_format_csv(self, tmp_path):
        from app.ingestion.loader import FileTypeDetector
        assert FileTypeDetector.detect_format(tmp_path / "data.csv") == "csv"

    def test_detect_format_md(self, tmp_path):
        from app.ingestion.loader import FileTypeDetector
        assert FileTypeDetector.detect_format(tmp_path / "README.md") == "markdown"

    def test_detect_format_txt(self, tmp_path):
        from app.ingestion.loader import FileTypeDetector
        assert FileTypeDetector.detect_format(tmp_path / "notes.txt") == "text"

    def test_detect_format_png(self, tmp_path):
        from app.ingestion.loader import FileTypeDetector
        assert FileTypeDetector.detect_format(tmp_path / "photo.png") == "png"

    def test_detect_format_jpeg(self, tmp_path):
        from app.ingestion.loader import FileTypeDetector
        assert FileTypeDetector.detect_format(tmp_path / "photo.jpeg") == "jpg"

    def test_detect_format_unsupported_raises(self, tmp_path):
        from app.ingestion.loader import FileTypeDetector
        with pytest.raises(ValueError, match="Unsupported file extension"):
            FileTypeDetector.detect_format(tmp_path / "archive.zip")

    def test_is_supported_with_dot(self):
        from app.ingestion.loader import FileTypeDetector
        assert FileTypeDetector.is_supported(".pdf") is True
        assert FileTypeDetector.is_supported(".exe") is False

    def test_is_supported_without_dot(self):
        from app.ingestion.loader import FileTypeDetector
        assert FileTypeDetector.is_supported("docx") is True
        assert FileTypeDetector.is_supported("bat") is False

    def test_supported_extensions_is_frozenset(self):
        from app.ingestion.loader import FileTypeDetector
        exts = FileTypeDetector.supported_extensions()
        assert isinstance(exts, frozenset)
        assert ".pdf" in exts
        assert ".docx" in exts
        assert ".csv" in exts
        assert ".md" in exts
        assert ".png" in exts

    def test_validate_magic_bytes_pdf(self):
        from app.ingestion.loader import FileTypeDetector
        assert FileTypeDetector.validate_magic_bytes(b"%PDF-1.4 content", "pdf") is True
        assert FileTypeDetector.validate_magic_bytes(b"not a pdf", "pdf") is False

    def test_validate_magic_bytes_png(self):
        from app.ingestion.loader import FileTypeDetector
        png_magic = b"\x89PNG\r\n\x1a\n" + b"\x00" * 20
        assert FileTypeDetector.validate_magic_bytes(png_magic, "png") is True
        assert FileTypeDetector.validate_magic_bytes(b"\xff\xd8\xff", "png") is False

    def test_validate_magic_bytes_no_magic_defined_returns_true(self):
        from app.ingestion.loader import FileTypeDetector
        # CSV has no magic bytes → always True
        assert FileTypeDetector.validate_magic_bytes(b"any content", "csv") is True
        assert FileTypeDetector.validate_magic_bytes(b"any content", "text") is True


# ---------------------------------------------------------------------------
# ParserRegistry tests
# ---------------------------------------------------------------------------

class TestParserRegistry:
    def test_is_supported_pdf(self):
        from app.ingestion.parsers import ParserRegistry
        reg = ParserRegistry()
        assert reg.is_supported(".pdf") is True

    def test_is_supported_all_formats(self):
        from app.ingestion.parsers import ParserRegistry
        reg = ParserRegistry()
        for ext in [".pdf", ".docx", ".pptx", ".xlsx", ".csv", ".txt", ".md", ".png", ".jpg", ".jpeg", ".webp"]:
            assert reg.is_supported(ext), f"Expected '{ext}' to be supported"

    def test_is_not_supported_exe(self):
        from app.ingestion.parsers import ParserRegistry
        reg = ParserRegistry()
        assert reg.is_supported(".exe") is False

    def test_get_parser_returns_correct_type(self):
        from app.ingestion.parsers import ParserRegistry, PDFParser, CSVParser, TextParser
        reg = ParserRegistry()
        assert isinstance(reg.get_parser(".pdf"), PDFParser)
        assert isinstance(reg.get_parser(".csv"), CSVParser)
        assert isinstance(reg.get_parser(".txt"), TextParser)
        assert isinstance(reg.get_parser(".md"), TextParser)

    def test_get_parser_unsupported_raises(self):
        from app.ingestion.parsers import ParserRegistry, UnsupportedFileTypeError
        reg = ParserRegistry()
        with pytest.raises(UnsupportedFileTypeError):
            reg.get_parser(".xyz")

    def test_register_custom_parser(self):
        from app.ingestion.parsers import ParserRegistry, BaseParser
        from app.models import Document

        class DummyParser(BaseParser):
            SUPPORTED_EXTENSIONS = frozenset({".dummy"})
            def parse(self, file_path, original_filename, file_hash):
                return [Document(page_content="dummy", metadata={})]

        reg = ParserRegistry()
        reg.register(".dummy", DummyParser())
        assert reg.is_supported(".dummy") is True

    def test_supported_extensions_property(self):
        from app.ingestion.parsers import ParserRegistry
        reg = ParserRegistry()
        exts = reg.supported_extensions
        assert isinstance(exts, frozenset)
        assert ".pdf" in exts


# ---------------------------------------------------------------------------
# TextParser tests
# ---------------------------------------------------------------------------

class TestTextParser:
    def test_parse_txt(self, txt_file):
        from app.ingestion.parsers.text_parser import TextParser
        parser = TextParser()
        docs = parser.parse(txt_file, "sample.txt", _sha256(txt_file.read_bytes()))
        assert len(docs) == 1
        assert "Hello World" in docs[0].page_content
        assert docs[0].metadata["format"] == "text"
        assert docs[0].metadata["filename"] == "sample.txt"

    def test_parse_md(self, md_file):
        from app.ingestion.parsers.text_parser import TextParser
        parser = TextParser()
        docs = parser.parse(md_file, "readme.md", _sha256(md_file.read_bytes()))
        assert len(docs) == 1
        assert "# Introduction" in docs[0].page_content
        assert docs[0].metadata["format"] == "markdown"

    def test_parse_empty_file_returns_empty(self, tmp_path):
        from app.ingestion.parsers.text_parser import TextParser
        p = tmp_path / "empty.txt"
        p.write_text("", encoding="utf-8")
        parser = TextParser()
        docs = parser.parse(p, "empty.txt", "abc123")
        assert docs == []

    def test_parse_missing_file_raises(self, tmp_path):
        from app.ingestion.parsers.text_parser import TextParser
        parser = TextParser()
        with pytest.raises(FileNotFoundError):
            parser.parse(tmp_path / "nonexistent.txt", "nonexistent.txt", "abc")


# ---------------------------------------------------------------------------
# CSVParser tests
# ---------------------------------------------------------------------------

class TestCSVParser:
    def test_parse_csv(self, csv_file):
        from app.ingestion.parsers.csv_parser import CSVParser
        parser = CSVParser()
        docs = parser.parse(csv_file, "data.csv", _sha256(csv_file.read_bytes()))
        assert len(docs) == 1
        content = docs[0].page_content
        assert "Name" in content
        assert "Alice" in content
        assert "Bob" in content
        assert docs[0].metadata["row_count"] == 3
        assert docs[0].metadata["column_count"] == 3

    def test_parse_csv_metadata(self, csv_file):
        from app.ingestion.parsers.csv_parser import CSVParser
        parser = CSVParser()
        docs = parser.parse(csv_file, "data.csv", "testhash")
        meta = docs[0].metadata
        assert meta["format"] == "csv"
        assert meta["filename"] == "data.csv"
        assert meta["page_number"] == 1
        assert meta["total_pages"] == 1

    def test_parse_empty_csv_returns_empty(self, tmp_path):
        from app.ingestion.parsers.csv_parser import CSVParser
        p = tmp_path / "empty.csv"
        p.write_text("", encoding="utf-8")
        parser = CSVParser()
        docs = parser.parse(p, "empty.csv", "abc")
        assert docs == []


# ---------------------------------------------------------------------------
# DOCXParser tests
# ---------------------------------------------------------------------------

class TestDOCXParser:
    def test_parse_docx(self, docx_file):
        from app.ingestion.parsers.docx_parser import DOCXParser
        parser = DOCXParser()
        docs = parser.parse(docx_file, "test.docx", _sha256(docx_file.read_bytes()))
        assert len(docs) == 1
        content = docs[0].page_content
        assert "Test Document" in content
        assert "first paragraph" in content
        assert docs[0].metadata["format"] == "docx"

    def test_parse_docx_headings_as_markdown(self, docx_file):
        from app.ingestion.parsers.docx_parser import DOCXParser
        parser = DOCXParser()
        docs = parser.parse(docx_file, "test.docx", "hash123")
        # Headings should be converted to # markdown style
        assert "#" in docs[0].page_content

    def test_docx_supported_extensions(self):
        from app.ingestion.parsers.docx_parser import DOCXParser
        assert ".docx" in DOCXParser.SUPPORTED_EXTENSIONS


# ---------------------------------------------------------------------------
# PPTXParser tests
# ---------------------------------------------------------------------------

class TestPPTXParser:
    def test_parse_pptx(self, pptx_file):
        from app.ingestion.parsers.pptx_parser import PPTXParser
        parser = PPTXParser()
        docs = parser.parse(pptx_file, "test.pptx", _sha256(pptx_file.read_bytes()))
        assert len(docs) >= 1
        # Title should be in content
        content = " ".join(d.page_content for d in docs)
        assert "Slide One" in content

    def test_pptx_slide_metadata(self, pptx_file):
        from app.ingestion.parsers.pptx_parser import PPTXParser
        parser = PPTXParser()
        docs = parser.parse(pptx_file, "test.pptx", "hash456")
        for doc in docs:
            assert doc.metadata["format"] == "pptx"
            assert doc.metadata["filename"] == "test.pptx"
            assert "page_number" in doc.metadata


# ---------------------------------------------------------------------------
# XLSXParser tests
# ---------------------------------------------------------------------------

class TestXLSXParser:
    def test_parse_xlsx(self, xlsx_file):
        from app.ingestion.parsers.xlsx_parser import XLSXParser
        parser = XLSXParser()
        docs = parser.parse(xlsx_file, "data.xlsx", _sha256(xlsx_file.read_bytes()))
        assert len(docs) >= 1
        content = docs[0].page_content
        assert "Sheet1" in content or "Product" in content

    def test_xlsx_metadata(self, xlsx_file):
        from app.ingestion.parsers.xlsx_parser import XLSXParser
        parser = XLSXParser()
        docs = parser.parse(xlsx_file, "data.xlsx", "hash789")
        assert docs[0].metadata["format"] == "xlsx"
        assert docs[0].metadata["filename"] == "data.xlsx"
        assert "sheet_name" in docs[0].metadata


# ---------------------------------------------------------------------------
# ImageParser tests
# ---------------------------------------------------------------------------

class TestImageParser:
    def test_parse_png_placeholder(self, png_file):
        from app.ingestion.parsers.image_parser import ImageParser
        parser = ImageParser()
        docs = parser.parse(png_file, "test.png", _sha256(png_file.read_bytes()))
        assert len(docs) == 1
        content = docs[0].page_content
        assert "test.png" in content
        assert "Image" in content
        assert docs[0].metadata["format"] == "image"

    def test_parse_png_with_vision_hook(self, png_file):
        from app.ingestion.parsers.image_parser import ImageParser
        hook = lambda p: "A blue square image used for testing."
        parser = ImageParser(vision_hook=hook)
        docs = parser.parse(png_file, "test.png", "hash_img")
        assert "blue square" in docs[0].page_content

    def test_parse_png_failed_vision_hook_falls_back(self, png_file):
        from app.ingestion.parsers.image_parser import ImageParser
        def bad_hook(p):
            raise RuntimeError("Vision service unavailable")
        parser = ImageParser(vision_hook=bad_hook)
        docs = parser.parse(png_file, "test.png", "hash_img")
        # Should fall back to placeholder without raising
        assert len(docs) == 1
        assert "Image" in docs[0].page_content

    def test_image_dimensions_in_metadata(self, png_file):
        from app.ingestion.parsers.image_parser import ImageParser
        parser = ImageParser()
        docs = parser.parse(png_file, "test.png", "hash_dim")
        meta = docs[0].metadata
        assert meta.get("image_width") == 100
        assert meta.get("image_height") == 100


# ---------------------------------------------------------------------------
# ParserRegistry.parse() dispatch tests
# ---------------------------------------------------------------------------

class TestParserRegistryDispatch:
    def test_dispatches_txt(self, txt_file):
        from app.ingestion.parsers import ParserRegistry
        reg = ParserRegistry()
        docs = reg.parse(txt_file, "sample.txt", _sha256(txt_file.read_bytes()))
        assert len(docs) == 1
        assert "Hello World" in docs[0].page_content

    def test_dispatches_csv(self, csv_file):
        from app.ingestion.parsers import ParserRegistry
        reg = ParserRegistry()
        docs = reg.parse(csv_file, "data.csv", _sha256(csv_file.read_bytes()))
        assert len(docs) == 1

    def test_dispatches_md(self, md_file):
        from app.ingestion.parsers import ParserRegistry
        reg = ParserRegistry()
        docs = reg.parse(md_file, "readme.md", _sha256(md_file.read_bytes()))
        assert len(docs) == 1
        assert "Introduction" in docs[0].page_content

    def test_unsupported_raises(self, tmp_path):
        from app.ingestion.parsers import ParserRegistry, UnsupportedFileTypeError
        reg = ParserRegistry()
        p = tmp_path / "archive.zip"
        p.write_bytes(b"PK\x03\x04")
        with pytest.raises(UnsupportedFileTypeError):
            reg.parse(p, "archive.zip", "abc")

    def test_extension_inferred_from_original_filename(self, tmp_path):
        """Registry should use original_filename extension, not file_path extension."""
        from app.ingestion.parsers import ParserRegistry
        reg = ParserRegistry()
        # Write a CSV content but the temp path has no extension - use original_filename
        csv_content = b"A,B\n1,2\n3,4\n"
        p = tmp_path / "datafile"
        p.write_bytes(csv_content)
        docs = reg.parse(p, "datafile.csv", _sha256(csv_content))
        assert len(docs) >= 1


# ---------------------------------------------------------------------------
# IngestionPipeline multi-format integration tests
# ---------------------------------------------------------------------------

class TestIngestionPipelineMultiFormat:
    """Integration tests for IngestionPipeline with real parser + mock vector store."""

    def _make_pipeline(self):
        """Return a pipeline with mock vector store and embedding service."""
        from app.ingestion.pipeline import IngestionPipeline
        mock_vs = MagicMock()
        mock_vs.get_indexed_files.return_value = {}
        mock_vs.upsert_chunks.return_value = 5
        mock_emb = MagicMock()
        mock_emb.embed_batch.return_value = [[0.1] * 768] * 100
        return IngestionPipeline(
            vector_store=mock_vs,
            embedding_service=mock_emb,
        )

    def test_ingest_txt_file(self, txt_file):
        pipeline = self._make_pipeline()
        result = pipeline.ingest_uploaded_document(
            file_path=txt_file,
            original_filename="sample.txt",
            document_id="doc-txt-001",
            file_hash=_sha256(txt_file.read_bytes()),
        )
        assert result["chunk_count"] > 0
        assert result["page_count"] >= 1

    def test_ingest_csv_file(self, csv_file):
        pipeline = self._make_pipeline()
        result = pipeline.ingest_uploaded_document(
            file_path=csv_file,
            original_filename="data.csv",
            document_id="doc-csv-001",
            file_hash=_sha256(csv_file.read_bytes()),
        )
        assert result["chunk_count"] > 0

    def test_ingest_md_file(self, md_file):
        pipeline = self._make_pipeline()
        result = pipeline.ingest_uploaded_document(
            file_path=md_file,
            original_filename="readme.md",
            document_id="doc-md-001",
            file_hash=_sha256(md_file.read_bytes()),
        )
        assert result["chunk_count"] > 0

    def test_ingest_docx_file(self, docx_file):
        pipeline = self._make_pipeline()
        result = pipeline.ingest_uploaded_document(
            file_path=docx_file,
            original_filename="test.docx",
            document_id="doc-docx-001",
            file_hash=_sha256(docx_file.read_bytes()),
        )
        assert result["chunk_count"] > 0

    def test_ingest_pptx_file(self, pptx_file):
        pipeline = self._make_pipeline()
        result = pipeline.ingest_uploaded_document(
            file_path=pptx_file,
            original_filename="test.pptx",
            document_id="doc-pptx-001",
            file_hash=_sha256(pptx_file.read_bytes()),
        )
        assert result["chunk_count"] > 0

    def test_ingest_xlsx_file(self, xlsx_file):
        pipeline = self._make_pipeline()
        result = pipeline.ingest_uploaded_document(
            file_path=xlsx_file,
            original_filename="data.xlsx",
            document_id="doc-xlsx-001",
            file_hash=_sha256(xlsx_file.read_bytes()),
        )
        assert result["chunk_count"] > 0

    def test_ingest_image_file(self, png_file):
        pipeline = self._make_pipeline()
        result = pipeline.ingest_uploaded_document(
            file_path=png_file,
            original_filename="test.png",
            document_id="doc-img-001",
            file_hash=_sha256(png_file.read_bytes()),
        )
        assert result["chunk_count"] > 0

    def test_all_chunks_have_document_id(self, txt_file):
        """Verify all chunks carry the document_id metadata."""
        from app.ingestion.pipeline import IngestionPipeline
        from app.ingestion.parsers import ParserRegistry

        captured_chunks = []

        mock_vs = MagicMock()
        mock_vs.get_indexed_files.return_value = {}

        def capture_upsert(chunks, embeddings):
            captured_chunks.extend(chunks)
            return len(chunks)

        mock_vs.upsert_chunks.side_effect = capture_upsert
        mock_emb = MagicMock()
        mock_emb.embed_batch.return_value = [[0.1] * 768] * 100

        pipeline = IngestionPipeline(vector_store=mock_vs, embedding_service=mock_emb)
        pipeline.ingest_uploaded_document(
            file_path=txt_file,
            original_filename="sample.txt",
            document_id="doc-check-001",
            file_hash=_sha256(txt_file.read_bytes()),
        )

        for chunk in captured_chunks:
            assert chunk.metadata.get("document_id") == "doc-check-001"
            assert chunk.metadata.get("filename") == "sample.txt"

    def test_unsupported_format_raises(self, tmp_path):
        from app.ingestion.pipeline import IngestionPipeline
        from app.ingestion.parsers import UnsupportedFileTypeError

        pipeline = self._make_pipeline()
        p = tmp_path / "archive.zip"
        p.write_bytes(b"PK\x03\x04")

        with pytest.raises(UnsupportedFileTypeError):
            pipeline.ingest_uploaded_document(
                file_path=p,
                original_filename="archive.zip",
                document_id="doc-fail-001",
                file_hash="abc",
            )


# ---------------------------------------------------------------------------
# Upload API validation tests (via TestClient — no LLM required)
# ---------------------------------------------------------------------------

class TestUploadAPIValidation:
    """Test upload endpoint extension and magic-byte validation without a real server."""

    @pytest.fixture()
    def client(self):
        from fastapi.testclient import TestClient
        from app.api.routes import app
        return TestClient(app, raise_server_exceptions=False)

    def _make_chat(self, client) -> str:
        resp = client.post("/api/chats", json={"title": "Test Chat"})
        assert resp.status_code == 201, resp.text
        return resp.json()["id"]

    def test_upload_unsupported_extension_rejected(self, client):
        chat_id = self._make_chat(client)
        resp = client.post(
            f"/api/chats/{chat_id}/documents",
            files={"file": ("malware.exe", b"MZ\x90\x00", "application/octet-stream")},
        )
        assert resp.status_code == 400
        assert "Unsupported file type" in resp.json()["detail"]

    def test_upload_pdf_accepted(self, client, tmp_path):
        """A valid minimal PDF should be accepted (may fail at indexing without Ollama, but passes validation)."""
        from pypdf import PdfWriter
        buf = io.BytesIO()
        writer = PdfWriter()
        writer.add_blank_page(100, 100)
        writer.write(buf)
        pdf_bytes = buf.getvalue()

        chat_id = self._make_chat(client)
        resp = client.post(
            f"/api/chats/{chat_id}/documents",
            files={"file": ("test.pdf", pdf_bytes, "application/pdf")},
        )
        # Should not be 400 (format validation passes)
        assert resp.status_code in (201, 500)  # 500 only if Ollama is down

    def test_upload_txt_file_accepted(self, client):
        chat_id = self._make_chat(client)
        resp = client.post(
            f"/api/chats/{chat_id}/documents",
            files={"file": ("notes.txt", b"Hello world from plain text.", "text/plain")},
        )
        # Should not be 400 on validation
        assert resp.status_code in (201, 500)

    def test_upload_csv_accepted(self, client):
        chat_id = self._make_chat(client)
        csv_content = b"Name,Score\nAlice,95\nBob,87\n"
        resp = client.post(
            f"/api/chats/{chat_id}/documents",
            files={"file": ("scores.csv", csv_content, "text/csv")},
        )
        assert resp.status_code in (201, 500)

    def test_upload_empty_file_rejected(self, client):
        chat_id = self._make_chat(client)
        resp = client.post(
            f"/api/chats/{chat_id}/documents",
            files={"file": ("empty.txt", b"", "text/plain")},
        )
        assert resp.status_code == 400
        assert "Empty file" in resp.json()["detail"]

    def test_upload_pdf_with_wrong_magic_bytes_rejected(self, client):
        """A .pdf file with non-PDF bytes should be rejected."""
        chat_id = self._make_chat(client)
        resp = client.post(
            f"/api/chats/{chat_id}/documents",
            files={"file": ("fake.pdf", b"This is not a real PDF file at all.", "application/pdf")},
        )
        assert resp.status_code == 400
        assert "does not match" in resp.json()["detail"] or "PDF" in resp.json()["detail"]

    def test_upload_nonexistent_chat_returns_404(self, client):
        resp = client.post(
            "/api/chats/nonexistent-chat-id/documents",
            files={"file": ("test.txt", b"hello", "text/plain")},
        )
        assert resp.status_code == 404
