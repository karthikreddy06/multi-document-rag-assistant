"""
Automated tests for Phase 4.4 and 4.5: File Upload API and Document Processing.
Tests validation, security, deduplication, and chat document attachments.
"""

import io
from pathlib import Path
import pytest
from fastapi.testclient import TestClient

from app.api.routes import app, get_rag_app
from app.db import repository, init_db
from app.db.database import get_db_path

MINIMAL_VALID_PDF = (
    b"%PDF-1.4\n"
    b"1 0 obj<</Type/Catalog/Pages 2 0 R>>endobj\n"
    b"2 0 obj<</Type/Pages/Count 1/Kids[3 0 R]>>endobj\n"
    b"3 0 obj<</Type/Page/MediaBox[0 0 612 792]/Parent 2 0 R/Resources<<>>>>endobj\n"
    b"xref\n"
    b"0 4\n"
    b"0000000000 65535 f \n"
    b"0000000010 00000 n \n"
    b"0000000060 00000 n \n"
    b"0000000117 00000 n \n"
    b"trailer<</Size 4/Root 1 0 R>>\n"
    b"startxref\n"
    b"193\n"
    b"%%EOF\n"
)


@pytest.fixture(scope="module")
def client():
    """Test client for FastAPI app with initialized SQLite database."""
    init_db()
    with TestClient(app) as test_client:
        yield test_client


@pytest.fixture
def test_chat(client):
    """Creates a temporary chat for testing."""
    res = client.post("/api/chats", json={"title": "Upload Test Chat"})
    assert res.status_code == 201
    data = res.json()
    chat_id = data["id"]
    yield chat_id
    # Teardown chat
    client.delete(f"/api/chats/{chat_id}")


def test_upload_nonexistent_chat(client):
    """Uploading to a nonexistent chat returns 404."""
    files = {"file": ("test.pdf", io.BytesIO(MINIMAL_VALID_PDF), "application/pdf")}
    res = client.post("/api/chats/chat_nonexistent_123/documents", files=files)
    assert res.status_code == 404
    assert "not found" in res.json()["detail"].lower()


def test_upload_invalid_extension(client, test_chat):
    """Uploading an unsupported extension returns 400."""
    files = {"file": ("malware.exe", io.BytesIO(b"MZ\x90\x00"), "application/octet-stream")}
    res = client.post(f"/api/chats/{test_chat}/documents", files=files)
    assert res.status_code == 400
    assert "unsupported file type" in res.json()["detail"].lower()


def test_upload_empty_file(client, test_chat):
    """Uploading an empty (0 bytes) file returns 400."""
    files = {"file": ("empty.pdf", io.BytesIO(b""), "application/pdf")}
    res = client.post(f"/api/chats/{test_chat}/documents", files=files)
    assert res.status_code == 400
    assert "empty file" in res.json()["detail"].lower()


def test_upload_invalid_magic_bytes(client, test_chat):
    """Uploading a .pdf file with non-PDF content is rejected by magic byte check."""
    fake_content = b"NOT_A_PDF_DOCUMENT_CONTENT"
    files = {"file": ("fake.pdf", io.BytesIO(fake_content), "application/pdf")}
    res = client.post(f"/api/chats/{test_chat}/documents", files=files)
    assert res.status_code == 400
    assert "does not match" in res.json()["detail"].lower()


def test_upload_file_exceeds_max_size(client, test_chat, monkeypatch):
    """Uploading a file exceeding 50 MB is rejected."""
    # Monkeypatch MAX_UPLOAD_SIZE to 50 bytes for fast test execution
    import app.api.routes as routes_module
    monkeypatch.setattr(routes_module, "MAX_UPLOAD_SIZE", 50)

    large_content = b"%PDF-" + b"A" * 60
    files = {"file": ("large.pdf", io.BytesIO(large_content), "application/pdf")}
    res = client.post(f"/api/chats/{test_chat}/documents", files=files)
    assert res.status_code == 400
    assert "exceeds maximum permitted limit" in res.json()["detail"].lower()


def test_upload_path_traversal_sanitization(client, test_chat):
    """Path traversal sequences in filename are sanitized safely."""
    traversal_name = "../../etc/passwd.pdf"
    files = {"file": (traversal_name, io.BytesIO(MINIMAL_VALID_PDF), "application/pdf")}
    res = client.post(f"/api/chats/{test_chat}/documents", files=files)
    assert res.status_code == 201
    data = res.json()["document"]
    # Path traversal stripped, filename sanitized
    assert "/" not in data["filename"]
    assert "\\" not in data["filename"]
    assert ".." not in data["filename"]
    assert data["filename"].endswith(".pdf")


def test_upload_valid_pdf_and_deduplication(client, test_chat):
    """Uploading a valid PDF creates/indexes it, and re-uploading reuses it without reprocessing."""
    files = {"file": ("first_upload.pdf", io.BytesIO(MINIMAL_VALID_PDF), "application/pdf")}
    res1 = client.post(f"/api/chats/{test_chat}/documents", files=files)
    assert res1.status_code == 201
    doc1 = res1.json()["document"]
    assert doc1["status"] == "ready"
    assert doc1["file_size"] == len(MINIMAL_VALID_PDF)
    doc_id = doc1["id"]

    # Re-upload the exact same file to the same chat (idempotency check)
    files2 = {"file": ("first_upload_renamed.pdf", io.BytesIO(MINIMAL_VALID_PDF), "application/pdf")}
    res2 = client.post(f"/api/chats/{test_chat}/documents", files=files2)
    assert res2.status_code == 201
    doc2 = res2.json()["document"]
    assert doc2["id"] == doc_id
    assert "without reprocessing" in res2.json()["message"].lower()

    # Create a second chat and attach the same file
    res_chat2 = client.post("/api/chats", json={"title": "Second Chat"})
    chat2_id = res_chat2.json()["id"]

    try:
        files3 = {"file": ("shared.pdf", io.BytesIO(MINIMAL_VALID_PDF), "application/pdf")}
        res3 = client.post(f"/api/chats/{chat2_id}/documents", files=files3)
        assert res3.status_code == 201
        doc3 = res3.json()["document"]
        assert doc3["id"] == doc_id
        assert "without reprocessing" in res3.json()["message"].lower()

        # Check list of documents in chat2
        list_res = client.get(f"/api/chats/{chat2_id}/documents")
        assert list_res.status_code == 200
        docs_list = list_res.json()
        assert len(docs_list) == 1
        assert docs_list[0]["id"] == doc_id

        # Detach from chat2
        del_res = client.delete(f"/api/chats/{chat2_id}/documents/{doc_id}")
        assert del_res.status_code == 204

        # Document is removed from chat2
        list_res2 = client.get(f"/api/chats/{chat2_id}/documents")
        assert len(list_res2.json()) == 0

        # But document is STILL attached to test_chat
        list_res_orig = client.get(f"/api/chats/{test_chat}/documents")
        assert len(list_res_orig.json()) == 1
        assert list_res_orig.json()[0]["id"] == doc_id
    finally:
        client.delete(f"/api/chats/{chat2_id}")
