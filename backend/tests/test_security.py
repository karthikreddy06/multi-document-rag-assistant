"""
Automated tests for Phase 4 Security Requirements.
Tests malicious inputs, boundary conditions, path traversal, injection, and isolation.
"""

import io
from pathlib import Path
import pytest
from fastapi.testclient import TestClient

from app.api.routes import app
from app.db import repository, init_db

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
    res = client.post("/api/chats", json={"title": "Security Test Chat"})
    cid = res.json()["id"]
    yield cid
    client.delete(f"/api/chats/{cid}")


def test_fake_pdf_extension_rejection(client, test_chat):
    """Files with truly unsupported extensions (executables, scripts) must be rejected."""
    # Note: .png, .docx, .json are now valid formats in the multi-format system
    # Only truly dangerous/unsupported extensions should be blocked
    bad_extensions = ["payload.exe", "script.sh", "archive.zip", "library.dll", "binary.bin"]
    for bad_name in bad_extensions:
        files = {"file": (bad_name, io.BytesIO(b"content"), "application/octet-stream")}
        res = client.post(f"/api/chats/{test_chat}/documents", files=files)
        assert res.status_code == 400
        assert "unsupported file type" in res.json()["detail"].lower()


def test_magic_byte_validation_rejects_disguised_file(client, test_chat):
    """File disguised as .pdf with non-PDF contents must be rejected by magic bytes."""
    disguised_content = b"GIF89a\x01\x00\x01\x00\x80\x00\x00"
    files = {"file": ("malicious.pdf", io.BytesIO(disguised_content), "application/pdf")}
    res = client.post(f"/api/chats/{test_chat}/documents", files=files)
    assert res.status_code == 400
    assert "does not match" in res.json()["detail"].lower()


def test_empty_file_rejection(client, test_chat):
    """0-byte file must be rejected."""
    files = {"file": ("zero.pdf", io.BytesIO(b""), "application/pdf")}
    res = client.post(f"/api/chats/{test_chat}/documents", files=files)
    assert res.status_code == 400
    assert "empty file" in res.json()["detail"].lower()


def test_path_traversal_filenames(client, test_chat):
    """Ensure directory traversal characters (../ and ..\\) cannot escape storage directory."""
    traversal_attacks = [
        "../../../../etc/passwd.pdf",
        "..\\..\\..\\windows\\system32\\calc.pdf",
        "....//....//nested.pdf",
        "/etc/shadow.pdf",
        "C:\\Windows\\System32\\cmd.pdf",
    ]
    for attack in traversal_attacks:
        files = {"file": (attack, io.BytesIO(MINIMAL_VALID_PDF), "application/pdf")}
        res = client.post(f"/api/chats/{test_chat}/documents", files=files)
        assert res.status_code == 201
        doc = res.json()["document"]
        assert "/" not in doc["filename"]
        assert "\\" not in doc["filename"]
        assert ".." not in doc["filename"]
        # Storage path is safely scoped and does not reveal absolute filesystem path
        assert not Path(doc["storage_path"]).is_absolute()
        assert "uploads/" in doc["storage_path"]


def test_sql_injection_defense_on_chat_id(client):
    """Attempted SQL injection strings in chat_id must safely return 404, not corrupt DB."""
    sqli_payloads = [
        "' OR '1'='1",
        "chat_1'; DROP TABLE chats; --",
        "\" OR 1=1 --",
        "chat_1' UNION SELECT 1,2,3,4 --",
    ]
    for payload in sqli_payloads:
        res = client.get(f"/api/chats/{payload}")
        assert res.status_code == 404
        # Verify table still exists and functions
        chats_res = client.get("/api/chats")
        assert chats_res.status_code == 200


def test_no_stack_traces_exposed(client):
    """Malformed or invalid requests should not leak internal stack traces or Python line numbers."""
    res = client.get("/api/chats/chat_invalid_id")
    assert res.status_code == 404
    body_text = res.text
    assert "Traceback (most recent call last)" not in body_text
    assert "File \"" not in body_text
    assert "line " not in body_text


def test_cross_chat_document_removal_protection(client, test_chat):
    """A chat cannot remove a document that belongs to another chat."""
    # Create second chat and attach a document
    res_b = client.post("/api/chats", json={"title": "Chat B"})
    chat_b_id = res_b.json()["id"]

    try:
        files = {"file": ("chat_b_doc.pdf", io.BytesIO(MINIMAL_VALID_PDF), "application/pdf")}
        upload_res = client.post(f"/api/chats/{chat_b_id}/documents", files=files)
        assert upload_res.status_code == 201
        doc_id = upload_res.json()["document"]["id"]

        # Attempt to delete Chat B's document from test_chat (which does not have it attached)
        del_res = client.delete(f"/api/chats/{test_chat}/documents/{doc_id}")
        assert del_res.status_code == 404
        assert "not attached" in del_res.json()["detail"].lower()
    finally:
        client.delete(f"/api/chats/{chat_b_id}")


def test_chat_rename_validation(client, test_chat):
    """Whitespace-only titles or empty titles are rejected with 422."""
    for bad_title in ["", "   ", "\t\n  "]:
        res = client.patch(f"/api/chats/{test_chat}", json={"title": bad_title})
        assert res.status_code == 422
