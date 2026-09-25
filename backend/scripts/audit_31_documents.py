# -*- coding: utf-8 -*-
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
import httpx

def main():
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

    with get_db_connection() as conn:
        with conn.cursor() as cur:
            cur.execute("""
                SELECT d.id, d.filename, d.user_id, d.storage_path, d.file_hash,
                       d.status, d.created_at,
                       COUNT(c.id) as chunk_count
                FROM documents d
                LEFT JOIN document_chunks c ON d.id = c.document_id
                GROUP BY d.id, d.filename, d.user_id, d.storage_path, d.file_hash, d.status, d.created_at
                ORDER BY d.created_at ASC;
            """)
            docs = cur.fetchall()

    active_docs = []
    zero_chunk_docs = []
    missing_storage_docs = []
    other_inconsistent = []

    for d in docs:
        in_storage = d["storage_path"] in storage_objects if d["storage_path"] else False
        d["in_storage"] = in_storage
        if d["chunk_count"] > 0 and in_storage:
            active_docs.append(d)
        elif d["chunk_count"] > 0 and not in_storage:
            missing_storage_docs.append(d)
        elif d["chunk_count"] == 0 and in_storage:
            other_inconsistent.append(d)
        else: # chunk_count == 0 and not in_storage
            zero_chunk_docs.append(d)

    print("================================================================================")
    print(f"AUDIT SUMMARY: {len(docs)} TOTAL DOCUMENTS")
    print(f"Active in Corpus (chunks > 0 AND in storage): {len(active_docs)}")
    print(f"Zero Chunks & No Storage:                     {len(zero_chunk_docs)}")
    print(f"Chunks > 0 but Missing Storage:               {len(missing_storage_docs)}")
    print(f"Zero Chunks but In Storage:                   {len(other_inconsistent)}")
    print("================================================================================\n")

    print(f"--- 1. ACTIVE CORPUS DOCUMENTS ({len(active_docs)}) ---")
    for d in active_docs:
        print(f"  [{d['id']}] {d['filename']}")
        print(f"      user_id: {d['user_id']} | chunks: {d['chunk_count']} | in_storage: {d['in_storage']}")
        print(f"      storage_path: {d['storage_path']}")
        print(f"      file_hash:    {d['file_hash'][:20]}...")
        print()

    print(f"--- 2. THE 22 NON-CORPUS DOCUMENTS ({len(zero_chunk_docs) + len(missing_storage_docs) + len(other_inconsistent)}) ---")
    non_corpus = zero_chunk_docs + missing_storage_docs + other_inconsistent
    for i, d in enumerate(non_corpus, 1):
        print(f"  {i:2d}. [{d['id']}] {d['filename']}")
        print(f"      user_id: {d['user_id']} | chunks: {d['chunk_count']} | in_storage: {d['in_storage']}")
        print(f"      storage_path: {d['storage_path']}")
        print(f"      file_hash:    {d['file_hash'][:20]}... | status: {d['status']}")
        print(f"      created_at:   {d['created_at']}")
        print()

if __name__ == "__main__":
    main()
