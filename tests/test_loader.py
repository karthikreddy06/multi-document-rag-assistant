"""
Tests for PDFLoader and text cleaning.
"""

from pathlib import Path
import pytest

from app.ingestion.loader import PDFLoader, Document
from app.utils.text_cleaner import clean_extracted_text


def test_clean_extracted_text():
    raw_sample = "T ravelT rack \u2022 F ull-Stack \u2013 T ools: B. T ech CGP A"
    cleaned = clean_extracted_text(raw_sample)
    assert "TravelTrack" in cleaned
    assert "Full-Stack" in cleaned
    assert "Tools:" in cleaned
    assert "B.Tech" in cleaned
    assert "CGPA" in cleaned


def test_pdf_loader_success():
    pdf_path = Path("documents/company.pdf")
    if not pdf_path.exists():
        pytest.skip("company.pdf not found in documents/")

    loader = PDFLoader(pdf_path)
    docs = loader.load()

    assert len(docs) > 0
    first_doc = docs[0]
    assert isinstance(first_doc, Document)
    assert "KARTHIK" in first_doc.page_content.upper()
    assert first_doc.metadata["page_number"] == 1
    assert "doc_id" in first_doc.metadata
    assert "file_hash" in first_doc.metadata


def test_pdf_loader_missing_file():
    loader = PDFLoader("documents/non_existent_file.pdf")
    with pytest.raises(FileNotFoundError):
        loader.load()
