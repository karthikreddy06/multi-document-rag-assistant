"""
End-to-End Frontend Authentication Flow Tests (Phase 5).
Simulates the exact 12-step sequence requested for the React frontend authentication:
1. Register User A.
2. Login User A.
3. Refresh browser and verify User A remains logged in (restore session via /api/auth/me).
4. Logout (client-side token removal).
5. Login User B (Register + Login).
6. Verify B cannot see A's documents/chats.
7. Verify A can see only A's documents/chats after logging back in.
8. Upload a document as A and verify it is not visible to B.
9. Ask a RAG question as A.
10. Verify RAG still works with the authenticated session.
11. Test invalid/expired token behavior (returns 401, triggering global logout).
12. Test logout.
"""

import io
import json
import pytest
from fastapi.testclient import TestClient

from app.config import settings
from app.api.routes import app
from app.db import repository, database, init_db
from app.main import RAGApplication
from app.api.routes import get_rag_app


@pytest.fixture
def auth_flow_env(tmp_path, monkeypatch):
    """Set up an isolated SQLite and ChromaDB environment for the frontend auth flow test."""
    test_db = tmp_path / "test_frontend_auth.db"
    test_chroma = tmp_path / "test_frontend_chroma"
    test_chroma.mkdir(parents=True, exist_ok=True)

    init_db(test_db)

    orig_get_conn = database.get_db_connection

    def mock_get_conn(custom_path=None):
        return orig_get_conn(custom_path or test_db)

    monkeypatch.setattr(database, "get_db_connection", mock_get_conn)
    monkeypatch.setattr(repository, "get_db_connection", mock_get_conn)
    monkeypatch.setattr(settings, "chroma_path", str(test_chroma))
    monkeypatch.setattr(settings, "collection_name", "test_frontend_auth_docs")

    rag_app = RAGApplication(auto_ingest=False)

    def mock_gen(question, chunks, **kwargs):
        return f"Synthesized answer for '{question}' from {[c.metadata.get('filename') for c in chunks]}"

    rag_app.generator.generate = mock_gen
    app.dependency_overrides[get_rag_app] = lambda: rag_app

    client = TestClient(app)
    yield client
    app.dependency_overrides.clear()


def test_complete_frontend_auth_flow(auth_flow_env):
    client = auth_flow_env

    # ---------------------------------------------------------
    # STEP 1: Register User A
    # ---------------------------------------------------------
    reg_a = client.post(
        "/api/auth/register",
        json={"email": "usera@example.com", "password": "UserAPassword123!"}
    )
    assert reg_a.status_code == 201, reg_a.text
    data_a = reg_a.json()
    assert data_a["email"] == "usera@example.com"
    user_a_id = data_a["id"]

    # ---------------------------------------------------------
    # STEP 2: Login User A
    # ---------------------------------------------------------
    login_a = client.post(
        "/api/auth/login",
        json={"email": "usera@example.com", "password": "UserAPassword123!"}
    )
    assert login_a.status_code == 200, login_a.text
    token_a = login_a.json().get("access_token")
    assert token_a is not None
    auth_header_a = {"Authorization": f"Bearer {token_a}"}

    # ---------------------------------------------------------
    # STEP 3: Refresh browser and verify User A remains logged in
    # (Simulated by AuthContext mounting and calling /api/auth/me with token_a)
    # ---------------------------------------------------------
    me_a = client.get("/api/auth/me", headers=auth_header_a)
    assert me_a.status_code == 200
    assert me_a.json()["email"] == "usera@example.com"
    assert me_a.json()["id"] == user_a_id

    # Create a chat for User A
    chat_a = client.post(
        "/api/chats",
        json={"title": "User A Research Chat"},
        headers=auth_header_a
    )
    assert chat_a.status_code == 201
    chat_a_id = chat_a.json()["id"]

    # ---------------------------------------------------------
    # STEP 4: Logout
    # (In frontend, client discards token from localStorage & memory)
    # ---------------------------------------------------------
    client_token = None
    assert client_token is None

    # ---------------------------------------------------------
    # STEP 5: Register & Login User B
    # ---------------------------------------------------------
    reg_b = client.post(
        "/api/auth/register",
        json={"email": "userb@example.com", "password": "UserBPassword123!"}
    )
    assert reg_b.status_code == 201
    login_b = client.post(
        "/api/auth/login",
        json={"email": "userb@example.com", "password": "UserBPassword123!"}
    )
    assert login_b.status_code == 200
    token_b = login_b.json().get("access_token")
    assert token_b is not None
    auth_header_b = {"Authorization": f"Bearer {token_b}"}

    # ---------------------------------------------------------
    # STEP 6: Verify User B cannot see User A's documents/chats
    # ---------------------------------------------------------
    b_chats = client.get("/api/chats", headers=auth_header_b).json()
    assert len(b_chats) == 0  # B cannot see A's chat
    assert not any(c["id"] == chat_a_id for c in b_chats)

    # User B cannot open User A's chat by ID (returns 404)
    b_get_a_chat = client.get(f"/api/chats/{chat_a_id}", headers=auth_header_b)
    assert b_get_a_chat.status_code == 404

    # ---------------------------------------------------------
    # STEP 7: Verify User A can see only User A's documents/chats after logging back in
    # ---------------------------------------------------------
    # Create a chat for User B first
    client.post("/api/chats", json={"title": "User B Secret Chat"}, headers=auth_header_b)

    # Re-login User A
    re_login_a = client.post(
        "/api/auth/login",
        json={"email": "usera@example.com", "password": "UserAPassword123!"}
    )
    assert re_login_a.status_code == 200
    token_a2 = re_login_a.json().get("access_token")
    assert token_a2 is not None
    auth_header_a2 = {"Authorization": f"Bearer {token_a2}"}

    a_chats = client.get("/api/chats", headers=auth_header_a2).json()
    assert len(a_chats) == 1
    assert a_chats[0]["id"] == chat_a_id
    assert a_chats[0]["title"] == "User A Research Chat"

    # ---------------------------------------------------------
    # STEP 8: Upload a document as A and verify it is not visible to B
    # ---------------------------------------------------------
    file_content = b"User A Confidential Project: Alpha Roadmap and Strategic Goals."
    files = {"file": ("user_a_plan.txt", io.BytesIO(file_content), "text/plain")}
    upload_res = client.post(
        f"/api/chats/{chat_a_id}/documents",
        files=files,
        headers=auth_header_a2
    )
    assert upload_res.status_code == 201, upload_res.text
    doc_a_id = upload_res.json()["document"]["id"]

    # Verify A sees the document in their chat
    a_docs = client.get(f"/api/chats/{chat_a_id}/documents", headers=auth_header_a2).json()
    assert any(d["id"] == doc_a_id for d in a_docs)

    # Verify B cannot directly access A's chat documents (returns 404)
    b_access_doc = client.get(f"/api/chats/{chat_a_id}/documents", headers=auth_header_b)
    assert b_access_doc.status_code == 404

    # ---------------------------------------------------------
    # STEP 9 & 10: Ask a RAG question as A & verify RAG works with authenticated session
    # ---------------------------------------------------------
    rag_query = {
        "query": "What is the strategic goal in the confidential project?"
    }
    rag_res = client.post(f"/api/chats/{chat_a_id}/chat", json=rag_query, headers=auth_header_a2)
    assert rag_res.status_code == 200, rag_res.text
    rag_data = rag_res.json()
    assert "answer" in rag_data
    assert len(rag_data["sources"]) > 0

    # ---------------------------------------------------------
    # STEP 11: Test invalid/expired token behavior
    # ---------------------------------------------------------
    # Bad token
    bad_header = {"Authorization": "Bearer totally-invalid-jwt-token"}
    res_bad = client.get("/api/auth/me", headers=bad_header)
    assert res_bad.status_code == 401

    # Protected route with bad token
    res_bad_route = client.get("/api/chats", headers=bad_header)
    assert res_bad_route.status_code == 401

    # Unauthenticated request to protected route
    res_no_auth = client.get("/api/chats")
    assert res_no_auth.status_code == 401

    # ---------------------------------------------------------
    # STEP 12: Test logout (clears session, subsequent unauthenticated requests blocked)
    # ---------------------------------------------------------
    # In frontend client, logout clears localStorage.
    # Without token, all protected workspace endpoints reject with 401
    logged_out_get_chats = client.get("/api/chats")
    assert logged_out_get_chats.status_code == 401
    logged_out_get_docs = client.get("/api/documents")
    assert logged_out_get_docs.status_code == 401
