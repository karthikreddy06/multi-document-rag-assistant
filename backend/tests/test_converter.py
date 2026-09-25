"""
Tests for DocumentConverter service (Phase 8).
Validates bi-directional conversions, unsupported format rejection,
and graceful degradation without crashing.
"""

import pytest
from app.services.converter import (
    DocumentConverter,
    is_conversion_supported,
    get_supported_targets,
    UnsupportedConversionError,
    detect_libreoffice,
)


def test_conversion_support_matrix():
    assert is_conversion_supported(".csv", ".xlsx") is True
    assert is_conversion_supported("csv", "xlsx") is True
    assert is_conversion_supported(".xlsx", ".csv") is True
    assert is_conversion_supported(".txt", ".docx") is True
    assert is_conversion_supported(".txt", ".pdf") is True
    assert is_conversion_supported(".md", ".docx") is True
    assert is_conversion_supported(".md", ".pdf") is True
    assert is_conversion_supported(".pdf", ".docx") is True
    # Unsupported
    assert is_conversion_supported(".csv", ".pdf") is False
    assert is_conversion_supported(".pptx", ".csv") is False

    assert ".xlsx" in get_supported_targets(".csv")
    assert ".docx" in get_supported_targets(".txt")
    assert ".pdf" in get_supported_targets(".txt")


def test_csv_to_xlsx_and_back():
    raw_csv = "Name,Age,Role\nAlice,30,Engineer\nBob,25,Designer\n".encode("utf-8")
    xlsx_bytes, xlsx_name, xlsx_mime = DocumentConverter.convert(raw_csv, "team.csv", ".xlsx")
    assert len(xlsx_bytes) > 0
    assert xlsx_name == "team.xlsx"
    assert "spreadsheetml" in xlsx_mime

    # Convert back from XLSX to CSV
    csv_bytes, csv_name, csv_mime = DocumentConverter.convert(xlsx_bytes, "team.xlsx", ".csv")
    assert csv_name == "team.csv"
    assert csv_mime == "text/csv"
    csv_str = csv_bytes.decode("utf-8")
    assert "Alice" in csv_str
    assert "Engineer" in csv_str
    assert "Bob" in csv_str


def test_txt_to_docx():
    raw_txt = "Hello world\nThis is a test document.\nLine three.".encode("utf-8")
    docx_bytes, docx_name, docx_mime = DocumentConverter.convert(raw_txt, "notes.txt", ".docx")
    assert len(docx_bytes) > 0
    assert docx_name == "notes.docx"
    assert "wordprocessingml" in docx_mime


def test_markdown_to_docx():
    raw_md = "# Title\n## Section 1\n- Item 1\n- Item 2\nRegular text paragraph.".encode("utf-8")
    docx_bytes, docx_name, docx_mime = DocumentConverter.convert(raw_md, "doc.md", ".docx")
    assert len(docx_bytes) > 0
    assert docx_name == "doc.docx"
    assert "wordprocessingml" in docx_mime


def test_txt_to_pdf():
    raw_txt = "Report\n\nGenerated for testing purposes.\nLine 1\nLine 2.".encode("utf-8")
    pdf_bytes, pdf_name, pdf_mime = DocumentConverter.convert(raw_txt, "report.txt", ".pdf")
    assert len(pdf_bytes) > 0
    assert pdf_name == "report.pdf"
    assert pdf_mime == "application/pdf"
    assert pdf_bytes.startswith(b"%PDF")


def test_markdown_to_pdf():
    raw_md = "# Header 1\n## Header 2\nThis is **bold** text.\n- Bullet point 1\n- Bullet point 2".encode("utf-8")
    pdf_bytes, pdf_name, pdf_mime = DocumentConverter.convert(raw_md, "guide.md", ".pdf")
    assert len(pdf_bytes) > 0
    assert pdf_name == "guide.pdf"
    assert pdf_mime == "application/pdf"
    assert pdf_bytes.startswith(b"%PDF")


def test_unsupported_conversion_raises():
    with pytest.raises(UnsupportedConversionError) as exc_info:
        DocumentConverter.convert(b"dummy", "image.png", ".xlsx")
    assert "not supported" in str(exc_info.value)


def test_docx_to_pdf_graceful():
    # If LibreOffice is not installed, it should raise UnsupportedConversionError, not crash
    raw_docx = b"PK\x03\x04fake docx content"
    if not detect_libreoffice():
        with pytest.raises(UnsupportedConversionError) as exc_info:
            DocumentConverter.convert(raw_docx, "test.docx", ".pdf")
        assert "LibreOffice" in str(exc_info.value)
