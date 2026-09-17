"""
Tests for DocumentChunker section-aware splitting and metadata.
"""

import pytest
from app.models import Document, Chunk
from app.ingestion.chunker import DocumentChunker


def test_chunker_section_splitting():
    sample_text = """John Doe
Email: john@example.com

PROFESSIONAL SUMMARY
Experienced software engineer.

TECHNICAL SKILLS
Programming: Python, Java, Go
Databases: PostgreSQL, Redis

PROJECTS
TravelTrack - Travel app
Built with FastAPI.
GitHub: https://github.com/john/TripTrack

SkillMatch - Job portal
Built with Django.
GitHub: https://github.com/john/SkillMatch-Web

EDUCATION
B.S. in Computer Science
"""
    doc = Document(page_content=sample_text, metadata={"doc_id": "test_doc_1", "page_number": 1})
    chunker = DocumentChunker(chunk_size=800, chunk_overlap=50)
    chunks = chunker.split_documents([doc])

    assert len(chunks) >= 5

    sections = [c.metadata.get("section") for c in chunks]
    assert any("TECHNICAL SKILLS" in s for s in sections)
    assert any("TravelTrack" in s for s in sections)
    assert any("SkillMatch" in s for s in sections)
    assert any("EDUCATION" in s for s in sections)

    # Verify skills section kept all lines intact
    skills_chunk = next(c for c in chunks if c.metadata.get("section") == "TECHNICAL SKILLS")
    assert "Python" in skills_chunk.text
    assert "PostgreSQL" in skills_chunk.text


def test_chunker_fallback_on_plain_text():
    long_plain_text = "This is a sentence that will be repeated to test fallback splitting. " * 50
    doc = Document(page_content=long_plain_text, metadata={"doc_id": "plain_doc", "page_number": 1})
    chunker = DocumentChunker(chunk_size=200, chunk_overlap=20)
    chunks = chunker.split_documents([doc])

    assert len(chunks) > 1
    for c in chunks:
        assert len(c.text) <= 250
        assert "chunk_id" in c.metadata
