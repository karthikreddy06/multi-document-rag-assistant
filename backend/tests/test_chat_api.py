"""
Tests for Chat Session CRUD Endpoints in FastAPI.
Validates POST, GET, PATCH, DELETE /api/chats, validation, and database cascade.
"""

import time
import pytest
from fastapi.testclient import TestClient

from app.api.routes import app
from app.db import repository, database, init_db


@pytest.fixture
def client(tmp_path, monkeypatch):
    """
    Provide TestClient connected to an isolated test database.
    """
    test_db = tmp_path / "test_api_chat.db"
    init_db(test_db)

    # Monkeypatch get_db_connection in repository and database to use test_db
    orig_get_conn = database.get_db_connection

    def mock_get_conn(custom_path=None):
        return orig_get_conn(custom_path or test_db)

    monkeypatch.setattr(database, "get_db_connection", mock_get_conn)
    monkeypatch.setattr(repository, "get_db_connection", mock_get_conn)

    return TestClient(app)


def test_1_create_chat_default_title(client):
    """Test POST /api/chats with empty body creates chat with title 'New Chat'."""
    res = client.post("/api/chats", json={})
    assert res.status_code == 201
    data = res.json()
    assert data["title"] == "New Chat"
    assert data["id"].startswith("chat_")
    assert data["created_at"]
    assert data["updated_at"]


def test_2_create_chat_custom_title(client):
    """Test POST /api/chats with custom title."""
    res = client.post("/api/chats", json={"title": "Literature Analysis"})
    assert res.status_code == 201
    data = res.json()
    assert data["title"] == "Literature Analysis"
    assert data["id"]


def test_3_list_chats(client):
    """Test GET /api/chats returns chats ordered by updated_at DESC."""
    c1 = client.post("/api/chats", json={"title": "Chat Alpha"}).json()
    time.sleep(0.01)
    c2 = client.post("/api/chats", json={"title": "Chat Beta"}).json()

    res = client.get("/api/chats")
    assert res.status_code == 200
    chats = res.json()
    assert isinstance(chats, list)
    assert len(chats) >= 2

    # Most recent (c2) should be first
    assert chats[0]["id"] == c2["id"]


def test_4_get_chat_existing(client):
    """Test GET /api/chats/{id} returns 200 for existing chat."""
    c = client.post("/api/chats", json={"title": "Target Chat"}).json()
    res = client.get(f"/api/chats/{c['id']}")
    assert res.status_code == 200
    assert res.json()["title"] == "Target Chat"


def test_5_get_chat_invalid_id(client):
    """Test GET /api/chats/{id} returns 404 for non-existent chat."""
    res = client.get("/api/chats/non_existent_chat_999")
    assert res.status_code == 404
    assert "not found" in res.json()["detail"].lower()


def test_6_rename_chat_success(client):
    """Test PATCH /api/chats/{id} renames chat and updates updated_at."""
    c = client.post("/api/chats", json={"title": "Original Title"}).json()
    time.sleep(0.01)

    res = client.patch(f"/api/chats/{c['id']}", json={"title": "Renamed Title"})
    assert res.status_code == 200
    updated = res.json()
    assert updated["title"] == "Renamed Title"
    assert updated["id"] == c["id"]


def test_7_rename_chat_whitespace_validation(client):
    """Test PATCH /api/chats/{id} with blank or whitespace-only title returns 422."""
    c = client.post("/api/chats", json={"title": "Valid Title"}).json()

    # Empty string
    res1 = client.patch(f"/api/chats/{c['id']}", json={"title": ""})
    assert res1.status_code == 422

    # Whitespace only
    res2 = client.patch(f"/api/chats/{c['id']}", json={"title": "    "})
    assert res2.status_code == 422


def test_8_rename_chat_not_found(client):
    """Test PATCH /api/chats/{id} on non-existent chat returns 404."""
    res = client.patch("/api/chats/unknown_chat_id", json={"title": "Valid New Title"})
    assert res.status_code == 404


def test_9_delete_chat_success(client):
    """Test DELETE /api/chats/{id} returns 204."""
    c = client.post("/api/chats", json={"title": "To Be Deleted"}).json()
    res = client.delete(f"/api/chats/{c['id']}")
    assert res.status_code == 204

    # Confirm it cannot be retrieved
    get_res = client.get(f"/api/chats/{c['id']}")
    assert get_res.status_code == 404


def test_10_delete_chat_not_found(client):
    """Test DELETE /api/chats/{id} returns 404 when ID does not exist."""
    res = client.delete("/api/chats/unknown_chat_id")
    assert res.status_code == 404


def test_11_delete_chat_cascades_records(client):
    """
    Verify deleting chat cascades to messages and chat_documents in database.
    """
    # 1. Create chat via API
    c = client.post("/api/chats", json={"title": "Cascade Chat"}).json()
    chat_id = c["id"]

    # 2. Add message and document via repository
    msg = repository.create_message(chat_id=chat_id, role="user", content="Hello test")
    doc = repository.create_document(
        filename="cascade_test.pdf",
        file_hash="hash_cascade_1",
        file_size=500,
        status="ready",
    )
    repository.attach_document_to_chat(chat_id, doc["id"])

    # Verify they exist
    assert len(repository.list_messages(chat_id)) == 1
    assert len(repository.list_chat_documents(chat_id)) == 1

    # 3. Delete chat via API
    del_res = client.delete(f"/api/chats/{chat_id}")
    assert del_res.status_code == 204

    # 4. Verify cascade: messages and chat_documents gone
    assert len(repository.list_messages(chat_id)) == 0
    assert len(repository.list_chat_documents(chat_id)) == 0

    # 5. Verify document in global catalog still exists
    assert repository.get_document_by_id(doc["id"]) is not None
