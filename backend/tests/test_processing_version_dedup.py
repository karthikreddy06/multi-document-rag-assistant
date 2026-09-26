"""
Tests for Processing-Version Aware Ingestion and Deduplication.

Requirements tested:
1. Same user + same file hash + same processing version + ready -> reuse existing document.
2. Same user + same file hash + older processing version -> reprocess document using current pipeline.
3. Different users with same file hash -> isolated documents and vector store entries.
4. Existing image documents with older versions reprocess through vision hook.
5. Processing version is stored on documents and reflected in API responses.
"""

import io
import uuid
import hashlib
from unittest.mock import patch, MagicMock
from PIL import Image
import pytest
from fastapi.testclient import TestClient

from app.api.routes import app
from app.config import settings
from app.db import repository

client = TestClient(app)


@pytest.fixture
def test_users():
    """Create two distinct test users with unique emails for isolation testing."""
    uid = uuid.uuid4().hex[:8]
    email_a = f"user_pv_a_{uid}@example.com"
    email_b = f"user_pv_b_{uid}@example.com"
    pwd = "Password123!"

    client.post("/api/auth/register", json={"email": email_a, "password": pwd})
    login_a = client.post("/api/auth/login", json={"email": email_a, "password": pwd})
    token_a = login_a.json()["access_token"]
    user_a = login_a.json()["user"]

    client.post("/api/auth/register", json={"email": email_b, "password": pwd})
    login_b = client.post("/api/auth/login", json={"email": email_b, "password": pwd})
    token_b = login_b.json()["access_token"]
    user_b = login_b.json()["user"]

    return {
        "user_a": user_a,
        "headers_a": {"Authorization": f"Bearer {token_a}"},
        "user_b": user_b,
        "headers_b": {"Authorization": f"Bearer {token_b}"},
    }


def _create_sample_png() -> bytes:
    """Create a sample valid PNG image in bytes."""
    buf = io.BytesIO()
    img = Image.new("RGB", (64, 64), color="blue")
    img.save(buf, format="PNG")
    return buf.getvalue()


def test_same_hash_same_version_deduplicates_and_reuses(test_users):
    """
    When the same file is uploaded twice by the same user with the same processing version,
    the second upload must reuse the existing ready document without reprocessing.
    """
    headers = test_users["headers_a"]
    run_id = uuid.uuid4().hex[:8]

    # 1. Create chat session
    chat_res = client.post("/api/chats", json={"title": "Test Chat 1"}, headers=headers)
    assert chat_res.status_code in (200, 201)
    chat_id = chat_res.json()["id"]

    # 2. Upload text document
    content = f"Content for testing deduplication versioning.\nUnique text string {run_id}.".encode("utf-8")
    with patch("app.api.routes.get_storage_service") as mock_storage_factory:
        mock_storage = MagicMock()
        mock_storage.upload_file.return_value = f"uploads/mock_{run_id}.txt"
        mock_storage_factory.return_value = mock_storage

        upload_1 = client.post(
            f"/api/chats/{chat_id}/documents",
            files={"file": (f"notes_{run_id}.txt", content, "text/plain")},
            headers=headers,
        )
        assert upload_1.status_code in (200, 201)
        doc_1 = upload_1.json()["document"]
        assert upload_1.json()["message"] == "Document uploaded, parsed, and indexed successfully."
        assert doc_1["processing_version"] == settings.processing_version

        # 3. Create a second chat session for the same user
        chat_res_2 = client.post("/api/chats", json={"title": "Test Chat 2"}, headers=headers)
        chat_id_2 = chat_res_2.json()["id"]

        # 4. Upload the exact same content to the second chat
        upload_2 = client.post(
            f"/api/chats/{chat_id_2}/documents",
            files={"file": (f"notes_{run_id}.txt", content, "text/plain")},
            headers=headers,
        )
        assert upload_2.status_code in (200, 201)
        doc_2 = upload_2.json()["document"]

        # Must reuse existing document without reprocessing
        assert upload_2.json()["message"] == "Existing document attached to chat without reprocessing."
        assert doc_2["id"] == doc_1["id"]
        assert doc_2["file_hash"] == doc_1["file_hash"]
        assert doc_2["processing_version"] == settings.processing_version


def test_same_hash_older_version_triggers_reprocessing(test_users):
    """
    When a file was previously indexed under an older processing version (e.g. version 1),
    uploading the same file under current pipeline version (e.g. version 2) must trigger reprocessing.
    """
    headers = test_users["headers_a"]
    user_id = test_users["user_a"]["id"]
    run_id = uuid.uuid4().hex[:8]

    # 1. Create chat session
    chat_res = client.post("/api/chats", json={"title": "Test Upgrade Chat"}, headers=headers)
    chat_id = chat_res.json()["id"]

    # 2. Upload file initially
    content = f"Document to simulate pipeline upgrade.\nOriginal text block {run_id}.".encode("utf-8")

    with patch("app.api.routes.get_storage_service") as mock_storage_factory:
        mock_storage = MagicMock()
        mock_storage.upload_file.return_value = f"uploads/mock_{run_id}.txt"
        mock_storage_factory.return_value = mock_storage

        upload_init = client.post(
            f"/api/chats/{chat_id}/documents",
            files={"file": (f"upgrade_doc_{run_id}.txt", content, "text/plain")},
            headers=headers,
        )
        assert upload_init.status_code in (200, 201)
        doc_id = upload_init.json()["document"]["id"]

        # 3. Simulate an older pipeline version in the database (e.g. version 1)
        repository.update_document_status(
            document_id=doc_id,
            status="ready",
            processing_version=1,
            user_id=user_id,
        )
        check_doc = repository.get_document_by_id(doc_id, user_id=user_id)
        assert check_doc["processing_version"] == 1

        # 4. Re-upload the exact same file content
        upload_upgraded = client.post(
            f"/api/chats/{chat_id}/documents",
            files={"file": (f"upgrade_doc_{run_id}.txt", content, "text/plain")},
            headers=headers,
        )
        assert upload_upgraded.status_code in (200, 201)

        # Must NOT say "without reprocessing"
        assert upload_upgraded.json()["message"] == "Document uploaded, parsed, and indexed successfully."
        upgraded_doc = upload_upgraded.json()["document"]
        assert upgraded_doc["id"] == doc_id
        assert upgraded_doc["processing_version"] == settings.processing_version
        assert upgraded_doc["processing_version"] > 1


def test_different_users_same_hash_isolated(test_users):
    """
    When User A and User B upload the exact same file content, they must get
    isolated documents with distinct records and no cross-user leakage.
    """
    headers_a = test_users["headers_a"]
    headers_b = test_users["headers_b"]
    run_id = uuid.uuid4().hex[:8]

    # Create chats
    chat_a = client.post("/api/chats", json={"title": "Chat User A"}, headers=headers_a).json()["id"]
    chat_b = client.post("/api/chats", json={"title": "Chat User B"}, headers=headers_b).json()["id"]

    shared_content = f"Shared knowledge base document content across different users {run_id}.".encode("utf-8")

    with patch("app.api.routes.get_storage_service") as mock_storage_factory:
        mock_storage = MagicMock()
        mock_storage.upload_file.return_value = f"uploads/mock_{run_id}.txt"
        mock_storage_factory.return_value = mock_storage

        # User A uploads
        up_a = client.post(
            f"/api/chats/{chat_a}/documents",
            files={"file": ("shared.txt", shared_content, "text/plain")},
            headers=headers_a,
        )
        assert up_a.status_code in (200, 201)
        doc_a = up_a.json()["document"]

        # User B uploads exact same file
        up_b = client.post(
            f"/api/chats/{chat_b}/documents",
            files={"file": ("shared.txt", shared_content, "text/plain")},
            headers=headers_b,
        )
        assert up_b.status_code in (200, 201)
        doc_b = up_b.json()["document"]

        # Different document IDs based on user-deterministic hashing
        assert doc_a["id"] != doc_b["id"]
        assert doc_a["file_hash"] == doc_b["file_hash"]

        # User A cannot view User B's document
        res_b_from_a = client.get(f"/api/documents/{doc_b['id']}", headers=headers_a)
        assert res_b_from_a.status_code == 404

        # User B cannot view User A's document
        res_a_from_b = client.get(f"/api/documents/{doc_a['id']}", headers=headers_b)
        assert res_a_from_b.status_code == 404


def test_existing_image_documents_reprocessed_with_vision(test_users):
    """
    Verify that an image originally uploaded with older processing version 1 (placeholder)
    is reprocessed when re-uploaded under version 2 with the vision pipeline.
    """
    headers = test_users["headers_a"]
    user_id = test_users["user_a"]["id"]
    run_id = uuid.uuid4().hex[:8]

    chat_id = client.post("/api/chats", json={"title": "Image Test Chat"}, headers=headers).json()["id"]
    img_bytes = _create_sample_png()

    from app.api.routes import get_rag_app
    img_parser = get_rag_app().pipeline.parser_registry._parsers[".png"]

    with patch("app.api.routes.get_storage_service") as mock_storage_factory:
        mock_storage = MagicMock()
        mock_storage.upload_file.return_value = f"uploads/mock_{run_id}.png"
        mock_storage_factory.return_value = mock_storage

        # Initial upload
        mock_vision_markdown = "### Extracted Text\nNo text.\n\n### Visual Description\nA solid blue square image."
        with patch.object(img_parser, "_vision_hook", return_value=mock_vision_markdown):
            up_img = client.post(
                f"/api/chats/{chat_id}/documents",
                files={"file": (f"diagram_{run_id}.png", img_bytes, "image/png")},
                headers=headers,
            )
            assert up_img.status_code in (200, 201)
            doc_id = up_img.json()["document"]["id"]

        # Simulate that this was indexed under older pipeline version 1
        repository.update_document_status(
            document_id=doc_id,
            status="ready",
            processing_version=1,
            user_id=user_id,
        )

        # Re-upload triggers vision re-parsing because version 1 < version 2
        mock_upgraded_vision = "### Extracted Text\nNo text.\n\n### Visual Description\nHigh-resolution blue geometric test patch."
        with patch.object(img_parser, "_vision_hook", return_value=mock_upgraded_vision) as mock_vision:
            up_re = client.post(
                f"/api/chats/{chat_id}/documents",
                files={"file": (f"diagram_{run_id}.png", img_bytes, "image/png")},
                headers=headers,
            )
            assert up_re.status_code in (200, 201)
            assert up_re.json()["message"] == "Document uploaded, parsed, and indexed successfully."
            # Confirm vision hook was invoked for re-parsing
            mock_vision.assert_called_once()

        # Verify document in library has updated processing_version
        lib_doc = repository.get_document_by_id(doc_id, user_id=user_id)
        assert lib_doc["processing_version"] == settings.processing_version
        assert lib_doc["processing_version"] >= 2
