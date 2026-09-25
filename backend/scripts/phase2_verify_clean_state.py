# -*- coding: utf-8 -*-
"""
Phase 2: Comprehensive verification of clean production state.
Checks:
- documents == 9
- document_chunks == 1208
- Supabase Storage objects == 9
- orphaned pgvector chunks == 0
- orphaned Storage objects == 0
- no documents reference missing Storage objects
- no active documents missing pgvector chunks
- no broken chat_documents references
- all 9 production documents intact
- sample query test against production pgvector corpus
"""
import sys, os
sys.stdout.reconfigure(encoding='utf-8')
from pathlib import Path

BACKEND_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(BACKEND_DIR))
os.chdir(str(BACKEND_DIR))

from dotenv import load_dotenv
load_dotenv(".env")

from app.db.database import get_db_connection
from app.config import settings
from app.vectorstore.store import VectorStore
from app.embeddings.service import EmbeddingService
from app.retrieval.retriever import HybridRetriever
import httpx

EXPECTED_DOC_IDS = {
    "doc_7434cd3761f241f5": ("marine_protected_area_clean.csv", 1146),
    "doc_3e6e0177abc44b7e": ("Astha_Bisoi_Resume1.pdf", 14),
    "doc_762a3344ec8a41b0": ("Run NVIDIA Nemotron 3 Ultra FREE in Your Terminal.docx", 12),
    "doc_0a3fb82e08424130": ("Karthik_Reddy_DataScientist_Resume.pdf", 9),
    "doc_adbbb71c1346476c": ("KarthikReddy_Resume.pdf", 8),
    "doc_524cdc0f973e4626": ("Apocolypse Food Prep.xlsx", 7),
    "doc_3db6004113ce41a9": ("python_number_programs_20.pdf", 6),
    "doc_b55d07906c484096": ("AI_Driven_Collections_Strategy_Presentation.pptx", 5),
    "doc_d180e03b45a34ad5": ("image.jpg", 1),
}

def main():
    print("=" * 70)
    print("PHASE 2: VERIFY CLEAN STATE & PRODUCTION QUERY TEST")
    print("=" * 70)

    # 1. Supabase Storage Audit
    bucket = settings.supabase_storage_bucket
    key = settings.effective_supabase_secret_key
    url = f"{settings.supabase_url.rstrip('/')}/storage/v1/object/list/{bucket}"
    headers = {
        "apikey": key,
        "Authorization": f"Bearer {key}",
        "Content-Type": "application/json",
    }

    def list_prefix(prefix=""):
        payload = {"prefix": prefix, "limit": 100}
        r = httpx.post(url, headers=headers, json=payload, timeout=15)
        if r.status_code == 200:
            items = r.json()
            results = []
            for item in items:
                name = item.get("name")
                item_id = item.get("id")
                if item_id is None:
                    results.extend(list_prefix(f"{prefix}/{name}".strip("/")))
                else:
                    results.append(f"{prefix}/{name}".strip("/"))
            return results
        return []

    storage_objects = set(list_prefix())

    # 2. Database Audit
    with get_db_connection() as conn:
        with conn.cursor() as cur:
            # All documents
            cur.execute("""
                SELECT d.id, d.filename, d.storage_path, COUNT(c.id) as chunk_count
                FROM documents d
                LEFT JOIN document_chunks c ON d.id = c.document_id
                GROUP BY d.id, d.filename, d.storage_path
                ORDER BY d.filename;
            """)
            db_docs = cur.fetchall()

            # Total chunks
            cur.execute("SELECT COUNT(*) as cnt FROM document_chunks;")
            total_chunks = cur.fetchone()["cnt"]

            # Orphaned chunks
            cur.execute("""
                SELECT COUNT(*) as cnt FROM document_chunks c
                WHERE NOT EXISTS (SELECT 1 FROM documents d WHERE d.id = c.document_id);
            """)
            orphaned_chunks = cur.fetchone()["cnt"]

            # Broken chat_documents links
            cur.execute("""
                SELECT COUNT(*) as cnt FROM chat_documents cd
                WHERE NOT EXISTS (SELECT 1 FROM documents d WHERE d.id = cd.document_id)
                   OR NOT EXISTS (SELECT 1 FROM chats ch WHERE ch.id = cd.chat_id);
            """)
            broken_chat_docs = cur.fetchone()["cnt"]

    # 3. Check consistency
    doc_id_map = {d["id"]: d for d in db_docs}
    docs_missing_storage = [d for d in db_docs if d["storage_path"] not in storage_objects]
    docs_missing_chunks = [d for d in db_docs if d["chunk_count"] == 0]

    orphaned_storage = [
        obj for obj in storage_objects
        if not any(d["storage_path"] == obj for d in db_docs)
    ]

    print("\n--- AUDIT RESULTS ---")
    print(f"Total documents in database:     {len(db_docs)}  (Expected: 9)")
    print(f"Total pgvector chunks:           {total_chunks}  (Expected: 1208)")
    print(f"Total Supabase Storage objects:  {len(storage_objects)}  (Expected: 9)")
    print(f"Orphaned pgvector chunks:        {orphaned_chunks}  (Expected: 0)")
    print(f"Orphaned Storage objects:        {len(orphaned_storage)}  (Expected: 0)")
    print(f"Documents missing Storage:       {len(docs_missing_storage)}  (Expected: 0)")
    print(f"Active documents missing chunks: {len(docs_missing_chunks)}  (Expected: 0)")
    print(f"Broken chat_documents links:     {broken_chat_docs}  (Expected: 0)")

    assert len(db_docs) == 9, f"Failed: expected 9 documents, found {len(db_docs)}"
    assert total_chunks == 1208, f"Failed: expected 1208 chunks, found {total_chunks}"
    assert len(storage_objects) == 9, f"Failed: expected 9 storage objects, found {len(storage_objects)}"
    assert orphaned_chunks == 0, f"Failed: found {orphaned_chunks} orphaned chunks"
    assert len(orphaned_storage) == 0, f"Failed: found {len(orphaned_storage)} orphaned storage objects"
    assert len(docs_missing_storage) == 0, f"Failed: {len(docs_missing_storage)} docs missing storage"
    assert len(docs_missing_chunks) == 0, f"Failed: {len(docs_missing_chunks)} docs missing chunks"
    assert broken_chat_docs == 0, f"Failed: found {broken_chat_docs} broken chat_documents links"

    print("\n--- VERIFYING THE 9 PRODUCTION DOCUMENTS ---")
    for doc_id, (expected_fn, expected_chunks) in EXPECTED_DOC_IDS.items():
        doc = doc_id_map.get(doc_id)
        assert doc is not None, f"Missing expected document: {doc_id} ({expected_fn})"
        assert doc["filename"] == expected_fn, f"Filename mismatch for {doc_id}: {doc['filename']} vs {expected_fn}"
        assert doc["chunk_count"] == expected_chunks, f"Chunk count mismatch for {expected_fn}: {doc['chunk_count']} vs {expected_chunks}"
        print(f"  ✓ [{doc_id}] {doc['filename']} | chunks: {doc['chunk_count']} | storage: OK")

    # 4. Query Test
    print("\n--- TESTING APPLICATION RETRIEVAL QUERY ---")
    vector_store = VectorStore()
    embedding_service = EmbeddingService(provider="local", model="all-MiniLM-L6-v2", dimensions=384)
    print(f"Vector Store provider: {vector_store.__class__.__name__}")
    print(f"Embedding Service:     {embedding_service.provider} / {embedding_service.model} ({embedding_service.dimensions}d)")

    retriever = HybridRetriever(vector_store=vector_store, embedding_service=embedding_service, top_k=3)
    test_query = "What data science and machine learning skills does Karthik have?"
    results = retriever.retrieve(query=test_query, where={"user_id": "legacy_user"}, top_k=3)

    print(f"Query: '{test_query}'")
    print(f"Results returned: {len(results)}")
    for i, r in enumerate(results, 1):
        fn = r.metadata.get("filename", "unknown")
        doc_id = r.metadata.get("document_id", "unknown")
        print(f"  Result {i}: [{fn}] doc_id={doc_id} (score={r.score:.4f})")
        print(f"    Excerpt: {r.text[:120].replace(chr(10), ' ')}...")

    assert len(results) > 0, "Query test failed to return results!"
    print("\n✓ SUCCESS: Production RAG retrieval query executed and returned relevant chunks!")
    print("=" * 70)

if __name__ == "__main__":
    main()
