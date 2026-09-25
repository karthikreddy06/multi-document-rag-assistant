# -*- coding: utf-8 -*-
"""
Audit specifically targeted documents for removal:
- 10thmarksheet (1).pdf
- 12thmarksheet (1).pdf
- oracle java se17 - ScoreReport(M.Karthik reddy).pdf
- passwd.pdf
- hermes_classified.txt
- zeus_classified.txt

Prints ONLY safe metadata (filename, document_id, chunk_count, storage_path).
NO document contents or secrets exposed.
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
import httpx

TARGET_FILENAMES = [
    "10thmarksheet (1).pdf",
    "12thmarksheet (1).pdf",
    "oracle java se17 - ScoreReport(M.Karthik reddy).pdf",
    "passwd.pdf",
    "hermes_classified.txt",
    "zeus_classified.txt"
]

def main():
    print("=" * 70)
    print("PRE-REMOVAL SAFE SUMMARY OF TARGETED DOCUMENTS")
    print("=" * 70)

    with get_db_connection() as conn:
        with conn.cursor() as cur:
            # 1. Fetch matching documents
            cur.execute("""
                SELECT d.id, d.filename, d.user_id, d.storage_path, d.file_hash,
                       COUNT(c.id) as chunk_count
                FROM documents d
                LEFT JOIN document_chunks c ON d.id = c.document_id
                WHERE d.filename = ANY(%s)
                GROUP BY d.id, d.filename, d.user_id, d.storage_path, d.file_hash
                ORDER BY d.filename;
            """, (TARGET_FILENAMES,))
            docs = cur.fetchall()

            # 2. Check chat_documents associations
            cur.execute("""
                SELECT cd.document_id, COUNT(*) as chat_assoc_count
                FROM chat_documents cd
                JOIN documents d ON cd.document_id = d.id
                WHERE d.filename = ANY(%s)
                GROUP BY cd.document_id;
            """, (TARGET_FILENAMES,))
            chat_assocs = {r["document_id"]: r["chat_assoc_count"] for r in cur.fetchall()}

            # 3. Check for any duplicate or orphaned chunks with these filenames or document_ids
            cur.execute("""
                SELECT filename, document_id, COUNT(*) as cnt
                FROM document_chunks
                WHERE filename = ANY(%s)
                GROUP BY filename, document_id;
            """, (TARGET_FILENAMES,))
            chunk_summary = cur.fetchall()

    print(f"\nFound {len(docs)} matching document records in PostgreSQL:")
    print("-" * 70)
    for d in docs:
        doc_id = d["id"]
        fn = d["filename"]
        chunks = d["chunk_count"]
        spath = d["storage_path"] or "(None)"
        chats = chat_assocs.get(doc_id, 0)
        print(f"filename:       {fn}")
        print(f"document_id:    {doc_id}")
        print(f"chunk_count:    {chunks}")
        print(f"storage_path:   {spath}")
        print(f"chat_links:     {chats}")
        print("-" * 70)

    print("\nChunks breakdown in document_chunks:")
    for cs in chunk_summary:
        print(f"  filename: {cs['filename']} | doc_id: {cs['document_id']} | chunks: {cs['cnt']}")

    # 4. Check Supabase Storage objects
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

    all_objects = list_prefix()
    matching_storage = [
        obj for obj in all_objects
        if any(obj.endswith(fn) or f"/{fn}" in obj for fn in TARGET_FILENAMES)
    ]

    print(f"\nMatching objects in Supabase Storage ({len(matching_storage)} found):")
    for obj in matching_storage:
        print(f"  storage_object: {obj}")

    print("=" * 70)

if __name__ == "__main__":
    main()
