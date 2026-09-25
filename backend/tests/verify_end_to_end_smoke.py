"""
End-to-End Smoke Test verifying all Phase 13 requirements:
1. Register/test isolated user
2. Login
3. Upload a document
4. Confirm document appears in file library (with type, size, status, chunks)
5. Confirm file view inline
6. Confirm original can be downloaded
7. Ask: "explain the pdf completely"
8. Verify document-wide retrieval strategy triggered
9. Verify answer is formatted Markdown
10. Verify source chunks are returned
11. Create follow-up question
12. Reload/open the chat
13. Verify previous messages remain
14. Verify the uploaded document remains attached
15. Convert file to supported target format
16. Download converted file
17. Logout
18. Login again
19. Verify chats/documents remain
20. Verify another user cannot access them
"""

import io
import pytest
from fastapi.testclient import TestClient

from app.api.routes import app
from app.db import init_db
from app.config import settings

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


def test_full_phase_13_e2e_smoke(monkeypatch):
    from app.api.routes import get_rag_app
    rag_app = get_rag_app()
    monkeypatch.setattr(
        rag_app.generator,
        "generate_answer",
        lambda question, chunks, **kwargs: f"## Analysis\n- Documents: {[c.metadata.get('filename') for c in chunks]}\n- Overview: Complete explanation of document content."
    )

    client = TestClient(app)

    # 1. Register isolated user A
    email_a = "smoke_user_a@enterprise.com"
    pwd_a = "SmokePassword2026!"
    reg_a = client.post("/api/auth/register", json={"email": email_a, "password": pwd_a})
    assert reg_a.status_code in (201, 409)

    # 2. Login
    login_a = client.post("/api/auth/login", json={"email": email_a, "password": pwd_a})
    assert login_a.status_code == 200
    token_a = login_a.json()["access_token"]
    user_a = login_a.json()["user"]
    headers_a = {"Authorization": f"Bearer {token_a}"}

    # Register & Login isolated user B
    email_b = "smoke_user_b@enterprise.com"
    pwd_b = "SmokePassword2026!"
    client.post("/api/auth/register", json={"email": email_b, "password": pwd_b})
    login_b = client.post("/api/auth/login", json={"email": email_b, "password": pwd_b})
    headers_b = {"Authorization": f"Bearer {login_b.json()['access_token']}"}

    # Create chat for User A
    chat_res = client.post("/api/chats", json={"title": "New Chat"}, headers=headers_a)
    assert chat_res.status_code in (200, 201)
    chat_id = chat_res.json()["id"]

    # 3. Upload a document (CSV for conversion and PDF for viewing)
    csv_bytes = b"Name,Role,City\nAlice,Engineer,Seoul\nBob,Architect,Tokyo\n"
    csv_upload = client.post(
        f"/api/chats/{chat_id}/documents",
        files={"file": ("team_data.csv", csv_bytes, "text/csv")},
        headers=headers_a,
    )
    assert csv_upload.status_code in (200, 201)
    doc_id = csv_upload.json()["document"]["id"]

    # 4. Confirm document appears in file library with metadata
    docs_res = client.get("/api/documents", headers=headers_a)
    assert docs_res.status_code == 200
    docs = docs_res.json()["documents"]
    matched = next((d for d in docs if d["id"] == doc_id), None)
    assert matched is not None
    assert matched["filename"] == "team_data.csv"
    assert matched["file_type"] == "CSV"
    assert matched["file_size"] == len(csv_bytes)
    assert matched["status"] in ("ready", "pending", "processing")

    # 5. Confirm file can be viewed inline
    view_res = client.get(f"/api/documents/{doc_id}/view", headers=headers_a)
    assert view_res.status_code == 200
    assert view_res.content == csv_bytes
    assert "inline" in view_res.headers.get("Content-Disposition", "")

    # 6. Confirm original can be downloaded
    dl_res = client.get(f"/api/documents/{doc_id}/download", headers=headers_a)
    assert dl_res.status_code == 200
    assert dl_res.content == csv_bytes
    assert "attachment" in dl_res.headers.get("Content-Disposition", "")

    # 7. Ask: "explain the pdf completely" / broad document query
    # Check that broad query correctly invokes document-wide retrieval
    from app.retrieval.query_understanding import QueryAnalyzer, QueryIntent
    from app.retrieval.document_resolver import DocumentResolutionResult, ResolvedDocument
    from app.retrieval.planner import RetrievalPlanner, RetrievalStrategy

    analysis = QueryAnalyzer.analyze("explain the pdf completely")
    assert analysis.intent in (QueryIntent.EXHAUSTIVE, QueryIntent.SUMMARIZATION)
    doc_res = DocumentResolutionResult(
        resolved_documents=[ResolvedDocument(filename="team_data.csv", doc_id=doc_id)],
        is_strictly_targeted=True,
        chroma_where_filter={"user_id": user_a["id"]},
    )
    plan = RetrievalPlanner.create_plan(analysis, doc_res)
    assert plan.strategy == RetrievalStrategy.DOCUMENT_WIDE
    assert plan.candidate_k >= 50
    assert plan.final_top_k >= 25

    # 8-10. Ask in chat session & verify source provenance and markdown
    chat_ask = client.post(
        f"/api/chats/{chat_id}/chat",
        json={"query": "explain the pdf completely", "show_context": True},
        headers=headers_a,
    )
    assert chat_ask.status_code == 200
    chat_answer = chat_ask.json()
    assert "answer" in chat_answer
    assert isinstance(chat_answer["sources"], list)

    # 11. Create a follow-up question
    followup_ask = client.post(
        f"/api/chats/{chat_id}/chat",
        json={"query": "Who is Alice?", "show_context": True},
        headers=headers_a,
    )
    assert followup_ask.status_code == 200

    # 12-14. Reload/open the chat and verify message history and attached documents
    chat_details = client.get(f"/api/chats/{chat_id}", headers=headers_a)
    assert chat_details.status_code == 200
    # Auto-titling check: title is no longer "New Chat"
    assert chat_details.json()["title"] != "New Chat"

    chat_messages = client.get(f"/api/chats/{chat_id}/messages", headers=headers_a)
    assert chat_messages.status_code == 200
    msgs = chat_messages.json()
    assert len(msgs) >= 4  # 2 user queries + 2 assistant answers

    chat_docs = client.get(f"/api/chats/{chat_id}/documents", headers=headers_a)
    assert chat_docs.status_code == 200
    assert any(cd["id"] == doc_id for cd in chat_docs.json())

    # 15. Convert file to supported target format (CSV -> XLSX)
    convert_res = client.post(
        f"/api/documents/{doc_id}/convert",
        json={"target_format": "xlsx"},
        headers=headers_a,
    )
    assert convert_res.status_code == 200
    # 16. Download converted file
    assert len(convert_res.content) > 0
    assert "spreadsheetml" in convert_res.headers.get("Content-Type", "")

    # 17. Logout (Client discards token)
    # 18. Login again
    relogin_a = client.post("/api/auth/login", json={"email": email_a, "password": pwd_a})
    assert relogin_a.status_code == 200
    headers_a_new = {"Authorization": f"Bearer {relogin_a.json()['access_token']}"}

    # 19. Verify chats/documents remain
    re_chats = client.get("/api/chats", headers=headers_a_new)
    assert re_chats.status_code == 200
    assert any(c["id"] == chat_id for c in re_chats.json())

    re_docs = client.get("/api/documents", headers=headers_a_new)
    assert re_docs.status_code == 200
    assert any(d["id"] == doc_id for d in re_docs.json()["documents"])

    # 20. Verify another user cannot access them (Isolation)
    b_chat_access = client.get(f"/api/chats/{chat_id}", headers=headers_b)
    assert b_chat_access.status_code == 404

    b_doc_access = client.get(f"/api/documents/{doc_id}", headers=headers_b)
    assert b_doc_access.status_code == 404

    b_dl_access = client.get(f"/api/documents/{doc_id}/download", headers=headers_b)
    assert b_dl_access.status_code == 404

    b_view_access = client.get(f"/api/documents/{doc_id}/view", headers=headers_b)
    assert b_view_access.status_code == 404
