"""
Automated tests for Phase 4.6 and Phase 4.7: Chat-Scoped Retrieval and Message Persistence.
Ensures strict retrieval isolation between chats and reliable message persistence.
"""

from pathlib import Path
import pytest
from fastapi.testclient import TestClient

from app.api.routes import app
from app.db import repository, init_db
from app.vectorstore.store import VectorStore


@pytest.fixture(scope="module")
def client():
    """Test client for FastAPI app with initialized SQLite database."""
    init_db()
    with TestClient(app) as test_client:
        yield test_client


@pytest.fixture(scope="module")
def vs():
    return VectorStore()


def test_chat_no_ready_documents(client):
    """Asking a question in a chat with no attached documents returns clean guidance and persists messages."""
    res = client.post("/api/chats", json={"title": "Empty Scope Chat"})
    chat_id = res.json()["id"]

    try:
        # Ask question without attaching documents
        q_res = client.post(
            f"/api/chats/{chat_id}/chat",
            json={"query": "What is the meaning of life?", "show_context": False},
        )
        assert q_res.status_code == 200
        data = q_res.json()
        assert "no ready documents" in data["answer"].lower()
        assert len(data["sources"]) == 0

        # Verify messages persisted
        msg_res = client.get(f"/api/chats/{chat_id}/messages")
        assert msg_res.status_code == 200
        msgs = msg_res.json()
        assert len(msgs) == 2
        assert msgs[0]["role"] == "user"
        assert msgs[0]["content"] == "What is the meaning of life?"
        assert msgs[1]["role"] == "assistant"
        assert "no ready documents" in msgs[1]["content"].lower()
    finally:
        client.delete(f"/api/chats/{chat_id}")


@pytest.mark.llm
def test_chat_document_isolation(client, vs):
    """
    CRITICAL ISOLATION TEST:
    Chat A is attached to oldmansea.pdf.
    Chat B is attached to Alice_in_Wonderland.pdf.
    Queries in Chat A cannot retrieve Alice_in_Wonderland.
    Queries in Chat B cannot retrieve oldmansea.
    """
    indexed = vs.get_indexed_files()
    alice_hash = indexed.get("Alice_in_Wonderland.pdf")
    oldman_hash = indexed.get("oldmansea.pdf")

    assert alice_hash is not None, "Alice_in_Wonderland.pdf must be indexed."
    assert oldman_hash is not None, "oldmansea.pdf must be indexed."

    # Register/retrieve docs in SQLite repository
    doc_alice = repository.get_document_by_hash(alice_hash)
    if not doc_alice:
        doc_alice = repository.create_document(
            filename="Alice_in_Wonderland.pdf",
            file_hash=alice_hash,
            file_size=1000,
            status="ready",
        )

    doc_oldman = repository.get_document_by_hash(oldman_hash)
    if not doc_oldman:
        doc_oldman = repository.create_document(
            filename="oldmansea.pdf",
            file_hash=oldman_hash,
            file_size=1000,
            status="ready",
        )

    # Create Chat A and attach oldmansea.pdf
    res_a = client.post("/api/chats", json={"title": "Chat A (Old Man)"})
    chat_a_id = res_a.json()["id"]
    repository.attach_document_to_chat(chat_a_id, doc_oldman["id"])

    # Create Chat B and attach Alice_in_Wonderland.pdf
    res_b = client.post("/api/chats", json={"title": "Chat B (Alice)"})
    chat_b_id = res_b.json()["id"]
    repository.attach_document_to_chat(chat_b_id, doc_alice["id"])

    try:
        # Question in Chat A about Santiago (from Old Man and the Sea)
        res1 = client.post(
            f"/api/chats/{chat_a_id}/chat",
            json={"query": "Who is Santiago?", "show_context": True},
        )
        assert res1.status_code == 200
        data1 = res1.json()
        assert len(data1["sources"]) > 0
        # All sources MUST be oldmansea.pdf
        for s in data1["sources"]:
            assert s["filename"] == "oldmansea.pdf", f"Expected oldmansea.pdf, got {s['filename']}"

        # Question in Chat A about Alice in Wonderland (Cross-Chat Isolation check)
        res2 = client.post(
            f"/api/chats/{chat_a_id}/chat",
            json={"query": "Tell me about the White Rabbit and Queen of Hearts in wonderland", "show_context": True},
        )
        assert res2.status_code == 200
        data2 = res2.json()
        # Chat A CANNOT retrieve any chunk from Alice_in_Wonderland.pdf
        for s in data2["sources"]:
            assert s["filename"] != "Alice_in_Wonderland.pdf", "LEAKAGE: Chat A retrieved Alice_in_Wonderland.pdf!"

        # Question in Chat B about Alice in Wonderland
        res3 = client.post(
            f"/api/chats/{chat_b_id}/chat",
            json={"query": "Who falls down the rabbit hole in wonderland?", "show_context": True},
        )
        assert res3.status_code == 200
        data3 = res3.json()
        assert len(data3["sources"]) > 0
        # All sources MUST be Alice_in_Wonderland.pdf
        for s in data3["sources"]:
            assert s["filename"] == "Alice_in_Wonderland.pdf", f"Expected Alice_in_Wonderland.pdf, got {s['filename']}"

        # Question in Chat B about Santiago (Cross-Chat Isolation check)
        res4 = client.post(
            f"/api/chats/{chat_b_id}/chat",
            json={"query": "How many days did the old man go without taking a fish?", "show_context": True},
        )
        assert res4.status_code == 200
        data4 = res4.json()
        # Chat B CANNOT retrieve any chunk from oldmansea.pdf
        for s in data4["sources"]:
            assert s["filename"] != "oldmansea.pdf", "LEAKAGE: Chat B retrieved oldmansea.pdf!"

        # Verify message history on Chat A
        msgs_a = client.get(f"/api/chats/{chat_a_id}/messages").json()
        assert len(msgs_a) == 4  # 2 queries -> 4 messages (user, assistant, user, assistant)
        assert msgs_a[0]["role"] == "user"
        assert msgs_a[1]["role"] == "assistant"
        assert len(msgs_a[1]["sources"]) > 0
        assert msgs_a[2]["role"] == "user"
        assert msgs_a[3]["role"] == "assistant"
    finally:
        client.delete(f"/api/chats/{chat_a_id}")
        client.delete(f"/api/chats/{chat_b_id}")


@pytest.mark.llm
def test_shared_document_retrieval(client, vs):
    """
    SHARED DOCUMENT TEST:
    If a document is attached to both Chat A and Chat B, both chats can retrieve it.
    """
    indexed = vs.get_indexed_files()
    sample_hash = indexed.get("sample.pdf")
    assert sample_hash is not None

    doc_sample = repository.get_document_by_hash(sample_hash)
    if not doc_sample:
        doc_sample = repository.create_document(
            filename="sample.pdf",
            file_hash=sample_hash,
            file_size=500,
            status="ready",
        )

    res_a = client.post("/api/chats", json={"title": "Shared Chat A"})
    chat_a_id = res_a.json()["id"]
    res_b = client.post("/api/chats", json={"title": "Shared Chat B"})
    chat_b_id = res_b.json()["id"]

    try:
        repository.attach_document_to_chat(chat_a_id, doc_sample["id"])
        repository.attach_document_to_chat(chat_b_id, doc_sample["id"])

        q = "Sample PDF Document Robert Maron"
        res_a_q = client.post(f"/api/chats/{chat_a_id}/chat", json={"query": q, "show_context": True})
        res_b_q = client.post(f"/api/chats/{chat_b_id}/chat", json={"query": q, "show_context": True})

        assert res_a_q.status_code == 200
        assert res_b_q.status_code == 200

        assert any(s["filename"] == "sample.pdf" for s in res_a_q.json()["sources"])
        assert any(s["filename"] == "sample.pdf" for s in res_b_q.json()["sources"])
    finally:
        client.delete(f"/api/chats/{chat_a_id}")
        client.delete(f"/api/chats/{chat_b_id}")
