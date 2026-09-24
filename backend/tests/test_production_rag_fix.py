"""
Tests for production RAG answer fix:
- Groq streaming token parsing & reasoning token handling
- SSE output formatting & empty-stream fallback
- Broad/full-document resume retrieval across multiple pages
- Strict user isolation during broad/document-wide retrieval
"""

import json
import uuid
from typing import Any, Dict, List
from unittest.mock import MagicMock, patch

import pytest
from fastapi.testclient import TestClient

from app.api.routes import app
from app.db import repository
from app.generation.generator import LLMGenerator
from app.generation.prompts import build_rag_prompt
from app.models import RetrievedChunk
from app.retrieval.document_resolver import DocumentResolver, ResolvedDocument
from app.retrieval.planner import GenerationBudget, RetrievalPlan, RetrievalPlanner, RetrievalStrategy
from app.retrieval.query_understanding import QueryAnalyzer, QueryIntent
from app.retrieval.retriever import HybridRetriever


# ------------------------------------------------------------------------------
# 1. Groq Streaming Token Parsing & Reasoning Token Handling
# ------------------------------------------------------------------------------

def test_groq_streaming_reasoning_and_token_parsing():
    """Verify Groq stream parsing ignores delta.reasoning, sets hidden format, and yields content tokens."""
    generator = LLMGenerator(
        provider="groq",
        model="openai/gpt-oss-20b",
        api_url="https://api.groq.com/openai/v1",
        api_key="gsk_test_key",
        num_predict=120,
    )

    # Mock SSE stream lines from Groq
    mock_lines = [
        b'data: {"id":"1","choices":[{"delta":{"reasoning":"Thinking about resume..."},"index":0}]}',
        b'data: {"id":"2","choices":[{"delta":{"content":"Here "},"index":0}]}',
        b'data:{"id":"3","choices":[{"delta":{"content":"is the summary."},"index":0}]}',
        b'data: [DONE]',
    ]

    mock_resp = MagicMock()
    mock_resp.iter_lines.return_value = mock_lines
    mock_resp.raise_for_status.return_value = None

    captured_payload = {}

    class MockStreamContext:
        def __enter__(self):
            return mock_resp
        def __exit__(self, *args):
            pass

    def mock_stream(method, url, headers=None, json=None):
        nonlocal captured_payload
        captured_payload = json
        return MockStreamContext()

    with patch("httpx.Client.stream", side_effect=mock_stream):
        tokens = list(generator._cloud_generate_stream(prompt="Summarize", num_predict=120))

    # Verify content tokens were yielded and reasoning tokens were suppressed
    assert tokens == ["Here ", "is the summary."]

    # Verify request payload has reasoning_format="hidden" and max_tokens >= 1024
    assert captured_payload.get("reasoning_format") == "hidden"
    assert captured_payload.get("max_tokens") >= 1024


def test_groq_payload_max_tokens_floor():
    """Ensure cloud/Groq endpoints enforce a minimum token headroom of 1024 even if small num_predict is requested."""
    generator = LLMGenerator(
        provider="groq",
        model="openai/gpt-oss-20b",
        api_url="https://api.groq.com/openai/v1",
        api_key="gsk_test_key",
        num_predict=120,
    )

    captured_payload = {}

    mock_resp = MagicMock()
    mock_resp.raise_for_status.return_value = None
    mock_resp.json.return_value = {
        "choices": [{"message": {"content": "Summary text", "role": "assistant"}}]
    }

    def mock_post(url, headers=None, json=None):
        nonlocal captured_payload
        captured_payload = json
        return mock_resp

    with patch("httpx.Client.post", side_effect=mock_post):
        ans = generator._cloud_generate(prompt="Summarize", num_predict=120)

    assert ans == "Summary text"
    assert captured_payload.get("max_tokens") == 1024
    assert captured_payload.get("reasoning_format") == "hidden"


# ------------------------------------------------------------------------------
# 2. SSE Output Formatting & Empty-Stream Fallback
# ------------------------------------------------------------------------------

def test_sse_output_and_fallback_on_zero_stream_tokens():
    """Verify /api/chats/{chat_id}/stream outputs standard SSE and uses fallback if stream yields 0 tokens."""
    # Use legacy_user consistent with conftest auto-auth
    user_id = "legacy_user"
    user = repository.get_user_by_id(user_id)
    if not user:
        user = repository.create_user(email="legacy@local.dev", password_hash="hashed_pw", user_id=user_id)

    chat = repository.create_chat(user_id=user_id, title="SSE Stream Test")
    chat_id = chat["id"]

    # Attach a ready document to chat
    doc = repository.create_document(
        filename="resume.pdf",
        file_hash=f"hash_sse_{uuid.uuid4().hex[:12]}",
        file_size=1024,
        storage_path="/tmp/resume.pdf",
        status="ready",
        page_count=3,
        user_id=user_id,
    )
    repository.attach_document_to_chat(chat_id=chat_id, document_id=doc["id"], user_id=user_id)

    client = TestClient(app)

    from app.auth.security import create_access_token
    token = create_access_token(data={"sub": user_id, "email": "legacy@local.dev"})
    headers = {"Authorization": f"Bearer {token}"}

    # Mock generator stream returning 0 tokens, and non-streaming fallback returning actual text
    def mock_stream(*args, **kwargs):
        return iter([])  # 0 tokens

    def mock_generate(*args, **kwargs):
        return "Complete resume summary fallback generated."

    with patch("app.generation.generator.LLMGenerator.generate_answer_stream", side_effect=mock_stream), \
         patch("app.generation.generator.LLMGenerator.generate_answer", side_effect=mock_generate):

        resp = client.post(
            f"/api/chats/{chat_id}/stream",
            json={"query": "complete summary of the uploaded resume", "show_context": False},
            headers=headers,
        )

        assert resp.status_code == 200
        assert resp.headers["content-type"].startswith("text/event-stream")

        events = []
        for line in resp.text.split("\n"):
            line = line.strip()
            if line.startswith("data:"):
                data_str = line[5:].strip()
                if data_str != "[DONE]":
                    events.append(json.loads(data_str))

        # Ensure fallback token was emitted so frontend does NOT get an empty assistant response
        token_events = [e for e in events if e.get("type") == "token"]
        assert len(token_events) >= 1
        assert token_events[0]["token"] == "Complete resume summary fallback generated."

        # Verify sources event was emitted
        sources_events = [e for e in events if e.get("type") == "sources"]
        assert len(sources_events) == 1


# ------------------------------------------------------------------------------
# 3. Broad / Full-Document Resume Retrieval
# ------------------------------------------------------------------------------

def test_query_understanding_for_complete_summary_of_resume():
    """Verify 'complete summary of the uploaded resume' is classified as EXHAUSTIVE/SUMMARIZATION."""
    queries = [
        "complete summary of the uploaded resume",
        "full summary of the resume",
        "give me a complete summary of the uploaded document",
        "summary of the uploaded resume",
        "summarize the resume",
    ]

    for q in queries:
        analysis = QueryAnalyzer.analyze(q)
        assert analysis.intent in (QueryIntent.SUMMARIZATION, QueryIntent.EXHAUSTIVE), (
            f"Query '{q}' was classified as {analysis.intent}, expected SUMMARIZATION or EXHAUSTIVE"
        )


def test_document_resolver_for_uploaded_resume():
    """Verify DocumentResolver marks 'the uploaded resume' as strictly targeted."""
    docs = [{"id": "doc_res_1", "filename": "Karthik_Resume.pdf", "file_hash": "h_res_1"}]
    where_base = {"$and": [{"user_id": "user_123"}, {"document_id": "doc_res_1"}]}

    res = DocumentResolver.resolve(
        query="complete summary of the uploaded resume",
        available_documents=docs,
        base_where_filter=where_base,
    )

    assert res.is_strictly_targeted is True
    assert len(res.resolved_documents) == 1
    assert res.resolved_documents[0].filename == "Karthik_Resume.pdf"


def test_planner_selects_document_wide_for_resume_summary():
    """Verify RetrievalPlanner selects DOCUMENT_WIDE strategy with high top_k for resume summary."""
    q = "complete summary of the uploaded resume"
    analysis = QueryAnalyzer.analyze(q)

    docs = [{"id": "doc_res_1", "filename": "Karthik_Resume.pdf", "file_hash": "h_res_1"}]
    where_base = {"$and": [{"user_id": "user_123"}, {"document_id": "doc_res_1"}]}

    resolution = DocumentResolver.resolve(
        query=analysis.cleaned_query,
        available_documents=docs,
        base_where_filter=where_base,
    )

    plan = RetrievalPlanner.create_plan(analysis=analysis, doc_resolution=resolution, default_top_k=3)

    assert plan.strategy == RetrievalStrategy.DOCUMENT_WIDE
    assert plan.final_top_k == 50
    assert plan.generation_budget.num_predict == 768
    assert plan.generation_budget.num_ctx == 3584


def test_document_wide_retrieval_covers_multiple_pages():
    """Verify _retrieve_document_wide retrieves chunks from all pages in sequential order."""
    mock_vector_store = MagicMock()
    mock_embedding = MagicMock()

    # Document with 6 chunks across 3 pages
    mock_chunks_data = {
        "documents": [
            "Experience chunk p2",
            "Header chunk p1",
            "Education chunk p3",
            "Skills chunk p1",
            "Projects chunk p2",
            "Certifications chunk p3",
        ],
        "metadatas": [
            {"filename": "resume.pdf", "page_number": 2, "chunk_index": 0},
            {"filename": "resume.pdf", "page_number": 1, "chunk_index": 0},
            {"filename": "resume.pdf", "page_number": 3, "chunk_index": 0},
            {"filename": "resume.pdf", "page_number": 1, "chunk_index": 1},
            {"filename": "resume.pdf", "page_number": 2, "chunk_index": 1},
            {"filename": "resume.pdf", "page_number": 3, "chunk_index": 1},
        ],
    }

    mock_vector_store.collection.get.return_value = mock_chunks_data

    retriever = HybridRetriever(
        vector_store=mock_vector_store,
        embedding_service=mock_embedding,
        top_k=3,
    )

    plan = RetrievalPlan(
        strategy=RetrievalStrategy.DOCUMENT_WIDE,
        query="complete summary of the uploaded resume",
        target_documents=[ResolvedDocument(filename="resume.pdf", doc_id="d1", file_hash="h1")],
        candidate_k=100,
        final_top_k=50,
        where_filter={"$and": [{"user_id": "u1"}, {"filename": "resume.pdf"}]},
    )

    chunks = retriever._retrieve_document_wide(plan)

    # Must retrieve all 6 chunks, not just 3
    assert len(chunks) == 6

    # Verify sorted sequentially by page_number, then chunk_index
    page_numbers = [c.metadata["page_number"] for c in chunks]
    chunk_indices = [c.metadata["chunk_index"] for c in chunks]

    assert page_numbers == [1, 1, 2, 2, 3, 3]
    assert chunk_indices == [0, 1, 0, 1, 0, 1]


# ------------------------------------------------------------------------------
# 4. Strict User Isolation During Broad Retrieval
# ------------------------------------------------------------------------------

def test_user_isolation_during_broad_document_wide_retrieval():
    """Verify that broad/document-wide retrieval strictly enforces user_id filtering in Chroma."""
    mock_vector_store = MagicMock()
    mock_embedding = MagicMock()

    captured_where = None

    def mock_collection_get(where=None, include=None):
        nonlocal captured_where
        captured_where = where
        # Simulate that only User A's chunks are returned because where was filtered by user_id
        return {
            "documents": ["User A Page 1", "User A Page 2"],
            "metadatas": [
                {"filename": "resume_a.pdf", "user_id": "user_a", "page_number": 1, "chunk_index": 0},
                {"filename": "resume_a.pdf", "user_id": "user_a", "page_number": 2, "chunk_index": 0},
            ],
        }

    mock_vector_store.collection.get = mock_collection_get

    retriever = HybridRetriever(
        vector_store=mock_vector_store,
        embedding_service=mock_embedding,
        top_k=3,
    )

    user_a_where = {"user_id": "user_a"}
    plan = RetrievalPlan(
        strategy=RetrievalStrategy.DOCUMENT_WIDE,
        query="complete summary of the uploaded resume",
        target_documents=[ResolvedDocument(filename="resume_a.pdf", doc_id="doc_a", file_hash="ha")],
        candidate_k=100,
        final_top_k=50,
        where_filter=user_a_where,
    )

    chunks = retriever._retrieve_document_wide(plan)

    # Verify where filter sent to Chroma contains user_a
    assert captured_where is not None
    where_str = json.dumps(captured_where)
    assert "user_a" in where_str
    assert "user_b" not in where_str

    # Verify all returned chunks strictly belong to user_a
    for c in chunks:
        assert c.metadata["user_id"] == "user_a"


# ------------------------------------------------------------------------------
# 5. Markdown Section Formatting & Grounding Rules
# ------------------------------------------------------------------------------

def test_markdown_section_formatting_prompt_directive():
    """Verify that broad/summary queries include the Markdown section formatting directive."""
    chunks = [
        RetrievedChunk(text="Karthik Reddy\nPhone: +1-555-0199", metadata={"filename": "resume.pdf", "section": "Contact"}),
        RetrievedChunk(text="Education: B.Tech CS", metadata={"filename": "resume.pdf", "section": "Education"}),
    ]

    prompt = build_rag_prompt("complete summary of the uploaded resume", chunks)

    # Assert Markdown structure directives are present
    assert "STRUCTURED MARKDOWN FORMATTING" in prompt
    assert "DOCUMENT SUMMARY FORMATTING DIRECTIVE" in prompt
    assert "## Heading" in prompt or "## Section Name" in prompt
    assert "- **Field**: Value" in prompt
    assert "### Item" in prompt
    assert "Do NOT output a single compressed paragraph" in prompt


def test_grounding_and_missing_information_instructions():
    """Verify strict source-grounding instructions and exact missing-information wording."""
    chunks = [
        RetrievedChunk(text="Sample content", metadata={"filename": "doc.pdf"}),
    ]

    prompt = build_rag_prompt("What are the publications?", chunks)

    # Assert exact grounding requirements
    assert "Use ONLY information contained in the retrieved document context" in prompt
    assert "Do not invent missing information" in prompt
    assert "Not found in the provided document." in prompt
    assert "Do not fabricate or hallucinate plausible details" in prompt
    assert "Never expose or assume information from documents outside the provided context" in prompt


def test_streaming_generation_preserves_markdown_sections():
    """Verify streaming generation preserves Markdown headings, bullet points, and bold tags without corruption."""
    generator = LLMGenerator(
        provider="groq",
        model="openai/gpt-oss-20b",
        api_url="https://api.groq.com/openai/v1",
        api_key="gsk_test_key",
        num_predict=768,
    )

    markdown_chunks = [
        "## Contact Information\n",
        "- **Phone**: +1-555-0199\n",
        "- **Email**: karthik@example.com\n\n",
        "## Experience\n",
        "### Backend Developer Intern\n",
        "- Built async REST APIs using FastAPI\n\n",
        "## Projects\n",
        "### TravelTrack\n",
        "- Real-time itinerary planning tool\n",
    ]

    def mock_cloud_stream(prompt, num_predict=None):
        for piece in markdown_chunks:
            yield piece

    with patch.object(generator, "_cloud_generate_stream", side_effect=mock_cloud_stream):
        tokens = list(generator.generate_answer_stream(
            question="complete summary of the uploaded resume",
            chunks=[RetrievedChunk(text="Karthik resume", metadata={"filename": "resume.pdf"})],
        ))

    full_output = "".join(tokens)

    # Verify Markdown elements are preserved in the stream
    assert "## Contact Information" in full_output
    assert "- **Phone**: +1-555-0199" in full_output
    assert "### Backend Developer Intern" in full_output
    assert "### TravelTrack" in full_output


def test_factual_query_prompt_is_clean_without_redundant_directives():
    """Verify specific factual queries do not force full-document summary directive."""
    chunks = [
        RetrievedChunk(text="Graduation year: 2026", metadata={"filename": "resume.pdf"}),
    ]

    prompt = build_rag_prompt("What is the graduation year?", chunks)

    # Core grounding is present, but document summary directive is not triggered
    assert "STRICT GROUNDING" in prompt
    assert "DOCUMENT SUMMARY FORMATTING DIRECTIVE" not in prompt

