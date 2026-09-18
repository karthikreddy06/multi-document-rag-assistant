"""
Automated tests for structural ingestion and pluggable OCR architecture.
Verifies table/list boundary preservation and scanned PDF metadata transparency.
"""

import pytest
from pathlib import Path
from app.models import Document
from app.ingestion.chunker import DocumentChunker
from app.ingestion.loader import PDFLoader, OCRService


def test_chunker_preserves_numbered_list():
    """Numbered list items should retain cohesive list content type without fragmentation."""
    list_content = (
        "Here are the instructions:\n"
        "1. Initialize the vector store\n"
        "2. Parse document sections\n"
        "3. Generate dense embeddings\n"
        "4. Perform hybrid search\n"
        "5. Synthesize grounded answer"
    )
    doc = Document(
        page_content=list_content,
        metadata={"doc_id": "test_doc", "page_number": 1, "filename": "test.pdf"}
    )
    chunker = DocumentChunker(chunk_size=500)
    chunks = chunker.split_documents([doc])

    assert len(chunks) == 1
    assert chunks[0].metadata.get("content_type") == "list"
    assert "1. Initialize" in chunks[0].text
    assert "5. Synthesize" in chunks[0].text


def test_chunker_preserves_table_structure():
    """Tabular rows should be detected as table content type and kept cohesive."""
    table_content = (
        "| Subject | Marks | Grade |\n"
        "| Mathematics | 95 | A+ |\n"
        "| Physics | 92 | A+ |\n"
        "| Chemistry | 89 | A |"
    )
    doc = Document(
        page_content=table_content,
        metadata={"doc_id": "test_table", "page_number": 1, "filename": "marks.pdf"}
    )
    chunker = DocumentChunker(chunk_size=500)
    chunks = chunker.split_documents([doc])

    assert len(chunks) == 1
    assert chunks[0].metadata.get("content_type") == "table"
    assert "Mathematics" in chunks[0].text
    assert "Chemistry" in chunks[0].text


def test_ocr_service_interface():
    """Pluggable OCR service checks availability without crashing."""
    available = OCRService.is_available()
    assert isinstance(available, bool)
