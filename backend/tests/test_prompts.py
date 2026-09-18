"""
Tests for Prompt Construction and Guardrails.
"""

from app.models import RetrievedChunk
from app.generation.prompts import build_rag_prompt, format_context


def test_format_context():
    chunks = [
        RetrievedChunk(text="Python, SQL, Java", metadata={"filename": "sample_doc.pdf", "page_number": 2, "section": "TECHNICAL SKILLS"}),
        RetrievedChunk(text="Document content details", metadata={"filename": "sample_doc.pdf", "page_number": 3, "section": "OVERVIEW"}),
    ]
    formatted = format_context(chunks)
    assert "DOCUMENT 1: sample_doc.pdf" in formatted
    assert "TECHNICAL SKILLS" in formatted
    assert "Document content details" in formatted


def test_build_rag_prompt_rules():
    chunks = [
        RetrievedChunk(text="Python and SQL", metadata={"filename": "doc.pdf", "section": "SKILLS"}),
    ]
    prompt = build_rag_prompt("What are the skills?", chunks)

    # Verify critical generic guardrail rules
    assert "STRICT GROUNDING" in prompt
    assert "MULTI-DOCUMENT ANSWERS" in prompt
    assert "IMAGE HANDLING" in prompt
    assert "ACCURACY & TERMINOLOGY" in prompt
    assert "Question: What are the skills?" in prompt
