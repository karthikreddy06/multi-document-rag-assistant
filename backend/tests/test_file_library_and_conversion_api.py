"""
Tests for File Library, View, Download, Conversion, and Auto-Titling Endpoints.
Verifies JWT user ownership, supported conversions, storage cleanup, and isolation.
"""

import pytest
from fastapi.testclient import TestClient
from app.api.routes import app
from app.api.auth import create_access_token
from app.db import repository

client = TestClient(app)


@pytest.fixture
def auth_users():
    # Register and login User A
    client.post("/api/auth/register", json={"email": "user_lib_a@example.com", "password": "Password123!"})
    login_a = client.post("/api/auth/login", json={"email": "user_lib_a@example.com", "password": "Password123!"})
    token_a = login_a.json()["access_token"]
    user_a = login_a.json()["user"]

    # Register and login User B
    client.post("/api/auth/register", json={"email": "user_lib_b@example.com", "password": "Password123!"})
    login_b = client.post("/api/auth/login", json={"email": "user_lib_b@example.com", "password": "Password123!"})
    token_b = login_b.json()["access_token"]
    user_b = login_b.json()["user"]

    return {
        "user_a": user_a,
        "token_a": token_a,
        "headers_a": {"Authorization": f"Bearer {token_a}"},
        "user_b": user_b,
        "token_b": token_b,
        "headers_b": {"Authorization": f"Bearer {token_b}"},
    }


def test_file_library_crud_view_download_and_convert(auth_users):
    headers_a = auth_users["headers_a"]
    headers_b = auth_users["headers_b"]
    user_a_id = auth_users["user_a"]["id"]

    # 1. Create a chat session for user A
    chat_res = client.post("/api/chats", json={"title": "New Chat"}, headers=headers_a)
    assert chat_res.status_code in (200, 201)
    chat_id = chat_res.json()["id"]

    # 2. Upload a CSV file to chat
    csv_content = b"Name,Age,Role\nAlice,30,Developer\nBob,28,Designer\n"
    upload_res = client.post(
        f"/api/chats/{chat_id}/documents",
        files={"file": ("staff.csv", csv_content, "text/csv")},
        headers=headers_a,
    )
    assert upload_res.status_code in (200, 201)
    doc_id = upload_res.json()["document"]["id"]

    # 3. View document library - User A sees document with file_type, size, status
    docs_res = client.get("/api/documents", headers=headers_a)
    assert docs_res.status_code == 200
    docs_data = docs_res.json()
    assert docs_data["total_documents"] >= 1
    found_doc = next((d for d in docs_data["documents"] if d["id"] == doc_id), None)
    assert found_doc is not None
    assert found_doc["filename"] == "staff.csv"
    assert found_doc["file_type"] == "CSV"
    assert found_doc["file_size"] == len(csv_content)

    # 4. Isolation: User B cannot see User A's document in library
    docs_b_res = client.get("/api/documents", headers=headers_b)
    assert docs_b_res.status_code == 200
    assert not any(d["id"] == doc_id for d in docs_b_res.json()["documents"])

    # 5. Download document - User A receives attachment
    dl_res = client.get(f"/api/documents/{doc_id}/download", headers=headers_a)
    assert dl_res.status_code == 200
    assert dl_res.content == csv_content
    assert 'attachment; filename="staff.csv"' in dl_res.headers.get("Content-Disposition", "")

    # 6. Isolation: User B receives 404 attempting to download
    dl_b_res = client.get(f"/api/documents/{doc_id}/download", headers=headers_b)
    assert dl_b_res.status_code == 404

    # 7. View document - User A receives inline view
    view_res = client.get(f"/api/documents/{doc_id}/view", headers=headers_a)
    assert view_res.status_code == 200
    assert view_res.content == csv_content
    assert "inline" in view_res.headers.get("Content-Disposition", "")

    # 8. Conversion targets
    targets_res = client.get(f"/api/documents/{doc_id}/convert/targets", headers=headers_a)
    assert targets_res.status_code == 200
    targets = targets_res.json()["supported_targets"]
    assert ".xlsx" in targets

    # 9. Convert CSV to XLSX
    convert_res = client.post(
        f"/api/documents/{doc_id}/convert",
        json={"target_format": "xlsx"},
        headers=headers_a,
    )
    assert convert_res.status_code == 200
    assert len(convert_res.content) > 0
    assert 'attachment; filename="staff.xlsx"' in convert_res.headers.get("Content-Disposition", "")
    assert "spreadsheetml" in convert_res.headers.get("Content-Type", "")

    # 10. Unsupported conversion returns 400
    unsupp_res = client.post(
        f"/api/documents/{doc_id}/convert",
        json={"target_format": "pdf"},
        headers=headers_a,
    )
    assert unsupp_res.status_code == 400

    # 11. Attach existing document to another chat
    chat2_res = client.post("/api/chats", json={"title": "Second Chat"}, headers=headers_a)
    assert chat2_res.status_code in (200, 201)
    chat2_id = chat2_res.json()["id"]
    attach_res = client.post(f"/api/chats/{chat2_id}/documents/{doc_id}", headers=headers_a)
    assert attach_res.status_code == 200

    # 12. Delete document
    del_res = client.delete(f"/api/documents/{doc_id}", headers=headers_a)
    assert del_res.status_code == 204

    # Document now 404
    get_after_del = client.get(f"/api/documents/{doc_id}", headers=headers_a)
    assert get_after_del.status_code == 404


def test_auto_titling_on_first_message(auth_users):
    headers = auth_users["headers_a"]

    # Create chat with default New Chat title
    chat_res = client.post("/api/chats", json={"title": "New Chat"}, headers=headers)
    assert chat_res.status_code in (200, 201)
    chat_id = chat_res.json()["id"]

    # Post message to chat
    client.post(
        f"/api/chats/{chat_id}/chat",
        json={"query": "Explain the project architecture and database schema in detail"},
        headers=headers,
    )

    # Re-fetch chat session
    fetched = client.get(f"/api/chats/{chat_id}", headers=headers).json()
    assert fetched["title"] != "New Chat"
    assert "Explain the project architecture" in fetched["title"]


def test_conversion_query_param_token_and_isolation(auth_users):
    token_a = auth_users["token_a"]
    token_b = auth_users["token_b"]
    headers_a = auth_users["headers_a"]

    # 1. Create chat and upload a document for User A
    chat_res = client.post("/api/chats", json={"title": "Test Chat"}, headers=headers_a)
    chat_id = chat_res.json()["id"]
    csv_content = b"Col1,Col2\nVal1,Val2\n"
    up_res = client.post(
        f"/api/chats/{chat_id}/documents",
        files={"file": ("test_token.csv", csv_content, "text/csv")},
        headers=headers_a,
    )
    doc_id = up_res.json()["document"]["id"]

    # 2. User A downloads converted XLSX using ?token= query parameter without Authorization header
    res_a = client.get(f"/api/documents/{doc_id}/convert?target_format=xlsx&token={token_a}")
    assert res_a.status_code == 200
    assert len(res_a.content) > 0
    assert 'attachment; filename="test_token.xlsx"' in res_a.headers.get("Content-Disposition", "")

    # 3. User B attempts to download User A's file using User B's token -> 404 Not Found (User Isolation)
    res_b = client.get(f"/api/documents/{doc_id}/convert?target_format=xlsx&token={token_b}")
    assert res_b.status_code == 404

    # 4. Request with no token or header -> 401 Unauthorized
    res_anon = client.get(f"/api/documents/{doc_id}/convert?target_format=xlsx")
    assert res_anon.status_code == 401


def test_natural_language_conversion_in_chat(auth_users):
    headers_a = auth_users["headers_a"]

    # 1. Create chat and upload a CSV
    chat_res = client.post("/api/chats", json={"title": "Conversion Chat"}, headers=headers_a)
    chat_id = chat_res.json()["id"]
    csv_content = b"Department,Budget\nEngineering,50000\nMarketing,30000\n"
    client.post(
        f"/api/chats/{chat_id}/documents",
        files={"file": ("department_budget.csv", csv_content, "text/csv")},
        headers=headers_a,
    )

    # 2. User asks natural language conversion query in chat
    chat_query_res = client.post(
        f"/api/chats/{chat_id}/chat",
        json={"query": "Convert this document to Excel please"},
        headers=headers_a,
    )
    assert chat_query_res.status_code == 200
    ans = chat_query_res.json()["answer"]
    assert "Successfully converted" in ans or "download" in ans.lower()
    assert "department_budget.xlsx" in ans
    assert "/api/documents/" in ans
    assert "token=" in ans
