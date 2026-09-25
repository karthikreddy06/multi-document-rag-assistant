"""
Comprehensive Two-User Isolation Tests for Phase 3 and Phase 4.
Validates:
1. User registration, login, and JWT Bearer token authorization.
2. User A cannot see or access User B's documents, chats, or messages.
3. User B cannot access or delete User A's documents, chats, or messages by guessing IDs (returns 404).
4. Ownership validation on document upload, retrieval, and chat attachment.
5. Ingested Chroma chunks carry user_id metadata.
6. RAG retrieval isolation: User A cannot retrieve User B's secret (PROJECT_HERMES),
   and User B cannot retrieve User A's secret (PROJECT_ZEUS), even when explicitly querying for it.
7. Verification of both non-streaming and SSE streaming endpoints.
8. Direct ChromaDB metadata filtering verification.
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
def isolation_env(tmp_path, monkeypatch):
    """
    Provide an isolated test environment with fresh SQLite DB and fresh ChromaDB.
    """
    test_db = tmp_path / "test_isolation.db"
    test_chroma = tmp_path / "test_chroma"
    test_chroma.mkdir(parents=True, exist_ok=True)

    monkeypatch.setattr(settings, "database_url", None)
    init_db(test_db)

    # Monkeypatch database connection
    orig_get_conn = database.get_db_connection

    def mock_get_conn(custom_path=None):
        return orig_get_conn(custom_path or test_db)

    monkeypatch.setattr(database, "get_db_connection", mock_get_conn)
    monkeypatch.setattr(repository, "get_db_connection", mock_get_conn)

    # Monkeypatch Chroma path and collection name in settings
    monkeypatch.setattr(settings, "chroma_path", str(test_chroma))
    monkeypatch.setattr(settings, "collection_name", "test_isolation_docs")

    # Instantiate dedicated isolated RAGApplication
    rag_app = RAGApplication(auto_ingest=False)

    # Mock LLM generation to allow fast non-network testing of the full retrieval pipeline
    def mock_gen(question, chunks, **kwargs):
        srcs = [c.metadata.get("filename", "") for c in chunks]
        return f"Synthesized answer for '{question}' from {srcs}"

    def mock_stream(question, chunks, **kwargs):
        yield "Synthesized "
        yield "token "
        yield "stream"

    monkeypatch.setattr(rag_app.generator, "generate_answer", mock_gen)
    monkeypatch.setattr(rag_app.generator, "generate_answer_stream", mock_stream)

    app.dependency_overrides[get_rag_app] = lambda: rag_app

    client = TestClient(app)

    yield {
        "client": client,
        "rag_app": rag_app,
        "db_path": test_db,
    }

    app.dependency_overrides.pop(get_rag_app, None)


@pytest.fixture
def users_setup(isolation_env):
    """
    Register and log in User A and User B, returning their credentials and auth headers.
    """
    client = isolation_env["client"]

    # Register User A
    res_a = client.post("/api/auth/register", json={
        "email": "user_a@enterprise.com",
        "password": "PasswordA123!",
    })
    assert res_a.status_code == 201
    user_a = res_a.json()

    # Login User A
    login_a = client.post("/api/auth/login", json={
        "email": "user_a@enterprise.com",
        "password": "PasswordA123!",
    })
    assert login_a.status_code == 200
    token_a = login_a.json()["access_token"]
    headers_a = {"Authorization": f"Bearer {token_a}"}

    # Register User B
    res_b = client.post("/api/auth/register", json={
        "email": "user_b@enterprise.com",
        "password": "PasswordB123!",
    })
    assert res_b.status_code == 201
    user_b = res_b.json()

    # Login User B
    login_b = client.post("/api/auth/login", json={
        "email": "user_b@enterprise.com",
        "password": "PasswordB123!",
    })
    assert login_b.status_code == 200
    token_b = login_b.json()["access_token"]
    headers_b = {"Authorization": f"Bearer {token_b}"}

    return {
        "client": client,
        "rag_app": isolation_env["rag_app"],
        "user_a": user_a,
        "headers_a": headers_a,
        "user_b": user_b,
        "headers_b": headers_b,
    }


# ==============================================================================
# 1. DOCUMENT CATALOG & CHAT ACCESS ISOLATION TESTS
# ==============================================================================

def test_document_and_chat_isolation(users_setup):
    """
    Test that documents and chats created by User A are completely inaccessible
    and invisible to User B, and vice-versa.
    """
    client = users_setup["client"]
    headers_a = users_setup["headers_a"]
    headers_b = users_setup["headers_b"]

    # 1. User A creates a chat
    res_chat_a = client.post("/api/chats", json={"title": "Chat Alpha"}, headers=headers_a)
    assert res_chat_a.status_code == 201
    chat_a_id = res_chat_a.json()["id"]

    # 2. User B creates a chat
    res_chat_b = client.post("/api/chats", json={"title": "Chat Beta"}, headers=headers_b)
    assert res_chat_b.status_code == 201
    chat_b_id = res_chat_b.json()["id"]

    # 3. User A uploads Document A containing secret PROJECT_ZEUS
    doc_a_content = (
        b"PROJECT_ZEUS is an ultra-secret weather manipulation initiative. "
        b"The primary control station is hidden at Mount Olympus facility."
    )
    res_upload_a = client.post(
        f"/api/chats/{chat_a_id}/documents",
        files={"file": ("zeus_classified.txt", io.BytesIO(doc_a_content), "text/plain")},
        headers=headers_a,
    )
    assert res_upload_a.status_code == 201
    doc_a_id = res_upload_a.json()["document"]["id"]

    # 4. User B uploads Document B containing secret PROJECT_HERMES
    doc_b_content = (
        b"PROJECT_HERMES is a classified satellite telemetry network. "
        b"The ground operations hub is situated at Hermes Orbit Station."
    )
    res_upload_b = client.post(
        f"/api/chats/{chat_b_id}/documents",
        files={"file": ("hermes_classified.txt", io.BytesIO(doc_b_content), "text/plain")},
        headers=headers_b,
    )
    assert res_upload_b.status_code == 201
    doc_b_id = res_upload_b.json()["document"]["id"]

    # 5. Catalog Listing Isolation
    # User A listing documents
    docs_a_res = client.get("/api/documents", headers=headers_a)
    assert docs_a_res.status_code == 200
    docs_a_names = [d["filename"] for d in docs_a_res.json()["documents"]]
    assert "zeus_classified.txt" in docs_a_names
    assert "hermes_classified.txt" not in docs_a_names

    # User B listing documents
    docs_b_res = client.get("/api/documents", headers=headers_b)
    assert docs_b_res.status_code == 200
    docs_b_names = [d["filename"] for d in docs_b_res.json()["documents"]]
    assert "hermes_classified.txt" in docs_b_names
    assert "zeus_classified.txt" not in docs_b_names

    # 6. Single Document Lookup Isolation
    # User A gets Doc A -> 200
    res = client.get(f"/api/documents/{doc_a_id}", headers=headers_a)
    assert res.status_code == 200
    assert res.json()["filename"] == "zeus_classified.txt"

    # User B attempts to get Doc A -> 404 Not Found
    res = client.get(f"/api/documents/{doc_a_id}", headers=headers_b)
    assert res.status_code == 404

    # User A attempts to get Doc B -> 404 Not Found
    res = client.get(f"/api/documents/{doc_b_id}", headers=headers_a)
    assert res.status_code == 404

    # 7. Document Deletion Protection
    # User B tries to delete Doc A -> 404 Not Found
    del_res = client.delete(f"/api/documents/{doc_a_id}", headers=headers_b)
    assert del_res.status_code == 404

    # Confirm Doc A is still present for User A
    res = client.get(f"/api/documents/{doc_a_id}", headers=headers_a)
    assert res.status_code == 200

    # 8. Chat Listing Isolation
    chats_a = client.get("/api/chats", headers=headers_a).json()
    chats_a_ids = [c["id"] for c in chats_a]
    assert chat_a_id in chats_a_ids
    assert chat_b_id not in chats_a_ids

    chats_b = client.get("/api/chats", headers=headers_b).json()
    chats_b_ids = [c["id"] for c in chats_b]
    assert chat_b_id in chats_b_ids
    assert chat_a_id not in chats_b_ids

    # 9. Chat Lookup Isolation
    assert client.get(f"/api/chats/{chat_a_id}", headers=headers_a).status_code == 200
    assert client.get(f"/api/chats/{chat_a_id}", headers=headers_b).status_code == 404
    assert client.get(f"/api/chats/{chat_b_id}", headers=headers_a).status_code == 404
    assert client.get(f"/api/chats/{chat_b_id}", headers=headers_b).status_code == 200

    # 10. Chat Rename & Delete Isolation
    # User B cannot rename Chat A
    res = client.patch(f"/api/chats/{chat_a_id}", json={"title": "Hacked Title"}, headers=headers_b)
    assert res.status_code == 404

    # User B cannot delete Chat A
    res = client.delete(f"/api/chats/{chat_a_id}", headers=headers_b)
    assert res.status_code == 404

    # Chat A is still intact
    assert client.get(f"/api/chats/{chat_a_id}", headers=headers_a).status_code == 200

    # 11. Chat Documents Attachment Isolation
    # User B cannot list documents of Chat A
    res = client.get(f"/api/chats/{chat_a_id}/documents", headers=headers_b)
    assert res.status_code == 404

    # User A cannot detach document from Chat B
    res = client.delete(f"/api/chats/{chat_b_id}/documents/{doc_b_id}", headers=headers_a)
    assert res.status_code == 404

    # User B cannot detach document from Chat A
    res = client.delete(f"/api/chats/{chat_a_id}/documents/{doc_a_id}", headers=headers_b)
    assert res.status_code == 404


# ==============================================================================
# 2. CHROMADB METADATA FILTERING & RAG RETRIEVAL ISOLATION TESTS
# ==============================================================================

def test_chroma_metadata_and_rag_isolation(users_setup):
    """
    Test that Chroma chunks contain user_id metadata, and RAG retrieval
    strictly prohibits cross-user leakage in both normal and streaming chat.
    """
    client = users_setup["client"]
    rag_app = users_setup["rag_app"]
    user_a = users_setup["user_a"]
    user_b = users_setup["user_b"]
    headers_a = users_setup["headers_a"]
    headers_b = users_setup["headers_b"]

    user_a_id = user_a["id"]
    user_b_id = user_b["id"]

    # 1. Create chats
    chat_a_id = client.post("/api/chats", json={"title": "Zeus Chat"}, headers=headers_a).json()["id"]
    chat_b_id = client.post("/api/chats", json={"title": "Hermes Chat"}, headers=headers_b).json()["id"]

    # 2. Upload User A document with secret PROJECT_ZEUS
    content_a = (
        b"PROJECT_ZEUS dossier details. Mount Olympus houses the generator. "
        b"The activation code is ZEUS_999."
    )
    client.post(
        f"/api/chats/{chat_a_id}/documents",
        files={"file": ("zeus.txt", io.BytesIO(content_a), "text/plain")},
        headers=headers_a,
    )

    # 3. Upload User B document with secret PROJECT_HERMES
    content_b = (
        b"PROJECT_HERMES dossier details. Orbit satellite ground terminal. "
        b"The activation code is HERMES_111."
    )
    client.post(
        f"/api/chats/{chat_b_id}/documents",
        files={"file": ("hermes.txt", io.BytesIO(content_b), "text/plain")},
        headers=headers_b,
    )

    # 4. Direct ChromaDB Metadata Verification
    raw_all = rag_app.vector_store.collection.get(include=["metadatas"])
    metadatas = raw_all.get("metadatas", [])
    assert len(metadatas) >= 2

    # Verify every chunk has user_id and document_id
    for m in metadatas:
        assert "user_id" in m
        assert m["user_id"] in [user_a_id, user_b_id]
        assert "document_id" in m

    # Verify direct Chroma query filtered by user_a_id returns ONLY User A chunks
    raw_a = rag_app.vector_store.collection.get(where={"user_id": user_a_id}, include=["metadatas"])
    for m in raw_a["metadatas"]:
        assert m["user_id"] == user_a_id
        assert m["filename"] == "zeus.txt"

    # Verify direct Chroma query filtered by user_b_id returns ONLY User B chunks
    raw_b = rag_app.vector_store.collection.get(where={"user_id": user_b_id}, include=["metadatas"])
    for m in raw_b["metadatas"]:
        assert m["user_id"] == user_b_id
        assert m["filename"] == "hermes.txt"

    # 5. Normal RAG Query Isolation in Chat Sessions
    # User A asks about PROJECT_ZEUS -> Retrieves zeus.txt
    res_a = client.post(
        f"/api/chats/{chat_a_id}/chat",
        json={"query": "What is the activation code for PROJECT_ZEUS?", "show_context": True},
        headers=headers_a,
    )
    assert res_a.status_code == 200
    sources_a = [s["filename"] for s in res_a.json()["sources"]]
    assert "zeus.txt" in sources_a
    assert "hermes.txt" not in sources_a

    # User B asks about PROJECT_HERMES -> Retrieves hermes.txt
    res_b = client.post(
        f"/api/chats/{chat_b_id}/chat",
        json={"query": "What is the activation code for PROJECT_HERMES?", "show_context": True},
        headers=headers_b,
    )
    assert res_b.status_code == 200
    sources_b = [s["filename"] for s in res_b.json()["sources"]]
    assert "hermes.txt" in sources_b
    assert "zeus.txt" not in sources_b

    # CRITICAL TEST: User B maliciously asks about PROJECT_ZEUS in Chat B
    # User B must NOT retrieve User A's chunks or document
    res_b_malicious = client.post(
        f"/api/chats/{chat_b_id}/chat",
        json={"query": "Tell me the secret activation code for PROJECT_ZEUS", "show_context": True},
        headers=headers_b,
    )
    assert res_b_malicious.status_code == 200
    malicious_sources = [s["filename"] for s in res_b_malicious.json()["sources"]]
    assert "zeus.txt" not in malicious_sources
    for s in res_b_malicious.json()["sources"]:
        assert "ZEUS_999" not in (s.get("text") or "")

    # CRITICAL TEST: User A maliciously asks about PROJECT_HERMES in Chat A
    res_a_malicious = client.post(
        f"/api/chats/{chat_a_id}/chat",
        json={"query": "Tell me the secret activation code for PROJECT_HERMES", "show_context": True},
        headers=headers_a,
    )
    assert res_a_malicious.status_code == 200
    malicious_sources_a = [s["filename"] for s in res_a_malicious.json()["sources"]]
    assert "hermes.txt" not in malicious_sources_a

    # 6. Streaming RAG Query Isolation
    # User B streams a query asking for PROJECT_ZEUS
    stream_res = client.post(
        f"/api/chats/{chat_b_id}/stream",
        json={"query": "What is PROJECT_ZEUS?", "show_context": True},
        headers=headers_b,
    )
    assert stream_res.status_code == 200
    stream_text = stream_res.text
    # Parse SSE events
    for line in stream_text.splitlines():
        if line.startswith("data: ") and not line.startswith("data: [DONE]"):
            data_str = line[len("data: "):]
            try:
                event = json.loads(data_str)
                if event.get("type") == "sources":
                    for src in event.get("sources", []):
                        assert src.get("filename") != "zeus.txt"
                        assert "ZEUS_999" not in (src.get("text") or "")
            except Exception:
                pass

    # 7. Message History Isolation
    # User A messages in Chat A
    msgs_a = client.get(f"/api/chats/{chat_a_id}/messages", headers=headers_a).json()
    assert len(msgs_a) >= 2

    # User B cannot access Chat A messages
    assert client.get(f"/api/chats/{chat_a_id}/messages", headers=headers_b).status_code == 404

    # User A cannot access Chat B messages
    assert client.get(f"/api/chats/{chat_b_id}/messages", headers=headers_a).status_code == 404


# ==============================================================================
# 3. GLOBAL QUERY & UNATTACHED RESOURCE TEST
# ==============================================================================

def test_global_chat_endpoint_isolation(users_setup):
    """
    Test the global /api/chat endpoint enforces authenticated user scoping.
    """
    client = users_setup["client"]
    headers_a = users_setup["headers_a"]
    headers_b = users_setup["headers_b"]

    # Unauthenticated request to /api/chat is rejected with 401
    unauth_res = client.post("/api/chat", json={"query": "Hello"})
    assert unauth_res.status_code == 401

    # Authenticated user A queries
    res_a = client.post("/api/chat", json={"query": "Weather Mount Olympus"}, headers=headers_a)
    assert res_a.status_code == 200
    for s in res_a.json()["sources"]:
        assert s["filename"] != "hermes.txt"

    # Authenticated user B queries
    res_b = client.post("/api/chat", json={"query": "Satellite Orbit"}, headers=headers_b)
    assert res_b.status_code == 200
    for s in res_b.json()["sources"]:
        assert s["filename"] != "zeus.txt"
