# -*- coding: utf-8 -*-
"""
Targeted Production Cleanup Script.
Removes ONLY the specified sensitive/personal/classified documents:
- 10thmarksheet (1).pdf
- 12thmarksheet (1).pdf
- oracle java se17 - ScoreReport(M.Karthik reddy).pdf
- passwd.pdf
- hermes_classified.txt
- zeus_classified.txt

Safe summary printed beforehand.
Performs deletion from:
1. document_chunks
2. chat_documents
3. documents
4. Supabase Storage

Verifies zero residual records across all tables and storage.
Runs comprehensive post-cleanup audit.
"""
import sys, os, json
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

TARGET_DOC_IDS = [
    "doc_b20f7a93133f424d", # 10thmarksheet (1).pdf
    "doc_d1928cdb40734c24", # 12thmarksheet (1).pdf
    "doc_3e96b08b69bc7690", # hermes_classified.txt
    "doc_29d5957f740a4ecf", # oracle java se17 - ScoreReport(M.Karthik reddy).pdf
    "doc_2371a21aa1034434", # passwd.pdf
    "doc_7c64d8310bd0248b", # zeus_classified.txt
]

def main():
    print("=" * 75)
    print("EXECUTE TARGETED PRODUCTION CLEANUP")
    print("=" * 75)

    # -------------------------------------------------------------------------
    # STEP 0: SAFE PRE-DELETION SUMMARY
    # -------------------------------------------------------------------------
    print("\n--- SAFE PRE-DELETION SUMMARY ---")
    with get_db_connection() as conn:
        with conn.cursor() as cur:
            cur.execute("""
                SELECT d.id, d.filename, d.storage_path, COUNT(c.id) as chunk_count
                FROM documents d
                LEFT JOIN document_chunks c ON d.id = c.document_id
                WHERE d.id = ANY(%s) OR d.filename = ANY(%s)
                GROUP BY d.id, d.filename, d.storage_path
                ORDER BY d.filename;
            """, (TARGET_DOC_IDS, TARGET_FILENAMES))
            to_remove = cur.fetchall()

    for item in to_remove:
        print(f"filename:     {item['filename']}")
        print(f"document_id:  {item['id']}")
        print(f"chunk_count:  {item['chunk_count']}")
        print(f"storage_path: {item['storage_path'] or '(None)'}")
        print()

    # -------------------------------------------------------------------------
    # STEP 1, 2, 4: DATABASE DELETIONS
    # -------------------------------------------------------------------------
    print("--- PERFORMING DATABASE DELETIONS ---")
    with get_db_connection() as conn:
        with conn.cursor() as cur:
            # 1. Delete rows from document_chunks
            cur.execute("""
                DELETE FROM document_chunks
                WHERE document_id = ANY(%s) OR filename = ANY(%s);
            """, (TARGET_DOC_IDS, TARGET_FILENAMES))
            chunks_deleted = cur.rowcount
            print(f"  ✓ Deleted {chunks_deleted} rows from document_chunks.")

            # 4. Delete rows from chat_documents
            cur.execute("""
                DELETE FROM chat_documents
                WHERE document_id = ANY(%s);
            """, (TARGET_DOC_IDS,))
            chat_docs_deleted = cur.rowcount
            print(f"  ✓ Deleted {chat_docs_deleted} rows from chat_documents.")

            # 2. Delete rows from documents
            cur.execute("""
                DELETE FROM documents
                WHERE id = ANY(%s) OR filename = ANY(%s);
            """, (TARGET_DOC_IDS, TARGET_FILENAMES))
            docs_deleted = cur.rowcount
            print(f"  ✓ Deleted {docs_deleted} rows from documents.")

            conn.commit()

    # -------------------------------------------------------------------------
    # STEP 3: SUPABASE STORAGE DELETIONS
    # -------------------------------------------------------------------------
    print("\n--- PERFORMING SUPABASE STORAGE DELETIONS ---")
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
    storage_to_delete = [
        obj for obj in all_objects
        if any(obj.endswith(fn) or f"/{fn}" in obj for fn in TARGET_FILENAMES)
        or any(doc_id in obj for doc_id in TARGET_DOC_IDS)
    ]

    print(f"  Found {len(storage_to_delete)} storage objects to delete:")
    for obj in storage_to_delete:
        print(f"    - {obj}")

    if storage_to_delete:
        delete_url = f"{settings.supabase_url.rstrip('/')}/storage/v1/object/{bucket}"
        delete_payload = {"prefixes": storage_to_delete}
        with httpx.Client(timeout=30.0) as client:
            resp = client.request("DELETE", delete_url, headers=headers, content=json.dumps(delete_payload))
            if resp.status_code in (200, 204):
                print(f"  ✓ Successfully bulk-deleted {len(storage_to_delete)} storage objects from '{bucket}'.")
            else:
                print(f"  Bulk delete returned {resp.status_code}: {resp.text}. Retrying individual deletes...")
                indiv_headers = {"apikey": key, "Authorization": f"Bearer {key}"}
                deleted_count = 0
                for obj in storage_to_delete:
                    single_url = f"{settings.supabase_url.rstrip('/')}/storage/v1/object/{bucket}/{obj}"
                    r = client.delete(single_url, headers=indiv_headers)
                    if r.status_code in (200, 204):
                        deleted_count += 1
                        print(f"    ✓ Deleted {obj}")
                    else:
                        print(f"    ✗ Failed {obj}: {r.status_code}")
                print(f"  Individual deletes: {deleted_count}/{len(storage_to_delete)} succeeded.")

    # -------------------------------------------------------------------------
    # STEP 5 & POST-DELETION VERIFICATION OF REMOVED TARGETS
    # -------------------------------------------------------------------------
    print("\n" + "=" * 75)
    print("VERIFICATION OF TARGET REMOVAL")
    print("=" * 75)

    with get_db_connection() as conn:
        with conn.cursor() as cur:
            # 1. document_chunks remaining
            cur.execute("""
                SELECT COUNT(*) as cnt FROM document_chunks
                WHERE document_id = ANY(%s) OR filename = ANY(%s);
            """, (TARGET_DOC_IDS, TARGET_FILENAMES))
            remaining_chunks = cur.fetchone()["cnt"]

            # 2. documents remaining
            cur.execute("""
                SELECT COUNT(*) as cnt FROM documents
                WHERE id = ANY(%s) OR filename = ANY(%s);
            """, (TARGET_DOC_IDS, TARGET_FILENAMES))
            remaining_docs = cur.fetchone()["cnt"]

            # 3. chat_documents remaining
            cur.execute("""
                SELECT COUNT(*) as cnt FROM chat_documents
                WHERE document_id = ANY(%s);
            """, (TARGET_DOC_IDS,))
            remaining_chat_docs = cur.fetchone()["cnt"]

    # 4. storage objects remaining
    all_objects_after = list_prefix()
    remaining_storage = [
        obj for obj in all_objects_after
        if any(obj.endswith(fn) or f"/{fn}" in obj for fn in TARGET_FILENAMES)
        or any(doc_id in obj for doc_id in TARGET_DOC_IDS)
    ]

    print(f"  Target document_chunks remaining: {remaining_chunks} (Expected: 0)")
    print(f"  Target documents rows remaining:    {remaining_docs} (Expected: 0)")
    print(f"  Target chat_documents remaining:   {remaining_chat_docs} (Expected: 0)")
    print(f"  Target storage objects remaining:  {len(remaining_storage)} (Expected: 0)")

    assert remaining_chunks == 0, f"Expected 0 chunks, got {remaining_chunks}"
    assert remaining_docs == 0, f"Expected 0 documents rows, got {remaining_docs}"
    assert remaining_chat_docs == 0, f"Expected 0 chat links, got {remaining_chat_docs}"
    assert len(remaining_storage) == 0, f"Expected 0 storage objects, got {len(remaining_storage)}"
    print("  ✓ SUCCESS: All target items completely removed (all zero).")

    # -------------------------------------------------------------------------
    # FINAL PRODUCTION AUDIT
    # -------------------------------------------------------------------------
    print("\n" + "=" * 75)
    print("FINAL PRODUCTION AUDIT")
    print("=" * 75)

    with get_db_connection() as conn:
        with conn.cursor() as cur:
            # Total production documents
            cur.execute("SELECT id, filename, user_id, storage_path FROM documents ORDER BY filename;")
            all_db_docs = cur.fetchall()
            doc_id_set = {d["id"] for d in all_db_docs}

            # Total pgvector chunks
            cur.execute("SELECT COUNT(*) as cnt FROM document_chunks;")
            total_pgvector_chunks = cur.fetchone()["cnt"]

            # Chunks per document
            cur.execute("""
                SELECT d.id, d.filename, COUNT(c.id) as chunk_count
                FROM documents d
                LEFT JOIN document_chunks c ON d.id = c.document_id
                GROUP BY d.id, d.filename
                ORDER BY d.filename;
            """)
            docs_with_chunks = cur.fetchall()

            # Orphaned chunks (chunks referencing non-existent document_id)
            cur.execute("""
                SELECT COUNT(*) as cnt FROM document_chunks c
                WHERE NOT EXISTS (SELECT 1 FROM documents d WHERE d.id = c.document_id);
            """)
            orphaned_chunks_count = cur.fetchone()["cnt"]

            # Integrity check on users, chats, messages
            cur.execute("SELECT COUNT(*) as cnt FROM users;")
            users_count = cur.fetchone()["cnt"]
            cur.execute("SELECT COUNT(*) as cnt FROM chats;")
            chats_count = cur.fetchone()["cnt"]
            cur.execute("SELECT COUNT(*) as cnt FROM messages;")
            messages_count = cur.fetchone()["cnt"]

    # Storage objects audit & orphaned storage objects
    orphaned_storage_objects = []
    prod_storage_objects = []
    test_storage_objects = []

    for obj in all_objects_after:
        # Extract doc_id from path if patterned as .../doc_<id>/...
        parts = obj.split("/")
        doc_part = next((p for p in parts if p.startswith("doc_")), None)
        if doc_part:
            if doc_part in doc_id_set:
                prod_storage_objects.append(obj)
            else:
                orphaned_storage_objects.append(obj)
        else:
            orphaned_storage_objects.append(obj)

    print(f"Total production documents:   {len(all_db_docs)}")
    print(f"Total pgvector chunks:         {total_pgvector_chunks}")
    print(f"Total Supabase storage objects:{len(all_objects_after)}")
    print(f"Orphaned pgvector chunks:      {orphaned_chunks_count}")
    print(f"Orphaned storage objects:      {len(orphaned_storage_objects)}")
    print(f"Total users preserved:         {users_count}")
    print(f"Total chats preserved:         {chats_count}")
    print(f"Total messages preserved:      {messages_count}")

    print("\nRemaining Document Filenames in Production Catalog:")
    print(f"  {'Filename':<50} | {'Document ID':<25} | {'Chunks':<7}")
    print("  " + "-" * 88)
    for dc in docs_with_chunks:
        print(f"  {dc['filename']:<50} | {dc['id']:<25} | {dc['chunk_count']:<7}")

    if orphaned_storage_objects:
        print("\nOrphaned Storage Objects Detail:")
        for oso in orphaned_storage_objects:
            print(f"  - {oso}")

    print("\n" + "=" * 75)
    print("CLEANUP & AUDIT COMPLETE")
    print("=" * 75)

if __name__ == "__main__":
    main()
