"""
Live API Verification Script testing all endpoints required for Phase 4.12.
"""

import io
import json
from fastapi.testclient import TestClient
from app.api.routes import app

MINIMAL_PDF = (
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

def run_live_api_checks():
    client = TestClient(app)
    endpoint_results = []

    def record(name, method, url, status_code, expected_status):
        success = status_code == expected_status
        endpoint_results.append({
            "name": name,
            "method": method,
            "url": url,
            "status_code": status_code,
            "expected": expected_status,
            "pass": success,
        })
        print(f"[{method}] {url} -> {status_code} ({'PASS' if success else 'FAIL'})")

    print("\n--- RUNNING LIVE API VERIFICATION ---")

    # 1. GET /api/health
    r = client.get("/api/health")
    record("Health check", "GET", "/api/health", r.status_code, 200)

    # 2. GET /api/documents
    r = client.get("/api/documents")
    record("Get documents catalog", "GET", "/api/documents", r.status_code, 200)

    # 3. GET /api/chats
    r = client.get("/api/chats")
    record("List chats", "GET", "/api/chats", r.status_code, 200)

    # 4. POST /api/chats
    r = client.post("/api/chats", json={"title": "Verification Chat"})
    record("Create chat", "POST", "/api/chats", r.status_code, 201)
    chat_id = r.json()["id"]

    # 5. GET /api/chats/{id}
    r = client.get(f"/api/chats/{chat_id}")
    record("Get chat by ID", "GET", f"/api/chats/{{id}}", r.status_code, 200)

    # 6. PATCH /api/chats/{id}
    r = client.patch(f"/api/chats/{chat_id}", json={"title": "Renamed Verification Chat"})
    record("Rename chat", "PATCH", f"/api/chats/{{id}}", r.status_code, 200)

    # 7. POST /api/chats/{id}/documents
    files = {"file": ("test_doc.pdf", io.BytesIO(MINIMAL_PDF), "application/pdf")}
    r = client.post(f"/api/chats/{chat_id}/documents", files=files)
    record("Upload document to chat", "POST", f"/api/chats/{{id}}/documents", r.status_code, 201)
    doc_id = r.json()["document"]["id"]

    # 8. GET /api/chats/{id}/documents
    r = client.get(f"/api/chats/{chat_id}/documents")
    record("List chat documents", "GET", f"/api/chats/{{id}}/documents", r.status_code, 200)

    # 9. POST /api/chats/{id}/chat (empty scope / test prompt)
    r = client.post(f"/api/chats/{chat_id}/chat", json={"query": "Hello?", "show_context": False})
    record("Session chat query", "POST", f"/api/chats/{{id}}/chat", r.status_code, 200)

    # 10. GET /api/chats/{id}/messages
    r = client.get(f"/api/chats/{chat_id}/messages")
    record("Get chat messages", "GET", f"/api/chats/{{id}}/messages", r.status_code, 200)

    # 11. DELETE /api/chats/{id}/documents/{document_id}
    r = client.delete(f"/api/chats/{chat_id}/documents/{doc_id}")
    record("Remove document from chat", "DELETE", f"/api/chats/{{id}}/documents/{{doc_id}}", r.status_code, 204)

    # 12. DELETE /api/chats/{id}
    r = client.delete(f"/api/chats/{chat_id}")
    record("Delete chat", "DELETE", f"/api/chats/{{id}}", r.status_code, 204)

    # 13. POST /api/chat (Legacy)
    # Using simple mock query or show_context
    r = client.post("/api/chat", json={"query": "", "show_context": False})
    record("Legacy chat validation", "POST", "/api/chat", r.status_code, 400)

    return endpoint_results

if __name__ == "__main__":
    run_live_api_checks()
