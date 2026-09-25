# -*- coding: utf-8 -*-
"""
Clean only the 12 orphaned Supabase Storage test fixture objects:
- zeus.txt
- hermes.txt
- user_a_plan.txt
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

TARGET_ORPHAN_FILENAMES = ["zeus.txt", "hermes.txt", "user_a_plan.txt"]

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

    all_objects = list_prefix()
    orphans_to_delete = [
        obj for obj in all_objects
        if any(obj.endswith(f"/{fn}") or obj == fn for fn in TARGET_ORPHAN_FILENAMES)
    ]

    print(f"Discovered {len(orphans_to_delete)} orphaned storage objects matching criteria:")
    for o in orphans_to_delete:
        print(f"  - {o}")

    # Step 1: Verify none are referenced in documents.storage_path
    with get_db_connection() as conn:
        with conn.cursor() as cur:
            cur.execute("""
                SELECT COUNT(*) as cnt
                FROM documents
                WHERE storage_path = ANY(%s);
            """, (orphans_to_delete,))
            ref_count = cur.fetchone()["cnt"]

            cur.execute("SELECT COUNT(*) as cnt FROM documents;")
            initial_doc_count = cur.fetchone()["cnt"]

            cur.execute("SELECT COUNT(*) as cnt FROM document_chunks;")
            initial_chunk_count = cur.fetchone()["cnt"]

    print(f"\nVerification: references in documents.storage_path = {ref_count} (Must be 0)")
    assert ref_count == 0, f"Aborting: found {ref_count} references in documents table!"

    # Step 2: Delete only these orphaned objects
    if orphans_to_delete:
        delete_url = f"{settings.supabase_url.rstrip('/')}/storage/v1/object/{bucket}"
        delete_payload = {"prefixes": orphans_to_delete}
        with httpx.Client(timeout=30.0) as client:
            resp = client.request("DELETE", delete_url, headers=headers, content=json.dumps(delete_payload))
            if resp.status_code in (200, 204):
                print(f"✓ Successfully deleted {len(orphans_to_delete)} orphaned objects via bulk delete.")
            else:
                print(f"Bulk delete returned {resp.status_code}. Using individual fallback...")
                indiv_headers = {"apikey": key, "Authorization": f"Bearer {key}"}
                deleted = 0
                for o in orphans_to_delete:
                    r = client.delete(f"{settings.supabase_url.rstrip('/')}/storage/v1/object/{bucket}/{o}", headers=indiv_headers)
                    if r.status_code in (200, 204):
                        deleted += 1
                print(f"Individual fallback deleted {deleted}/{len(orphans_to_delete)}")

    # Step 3: Re-audit
    all_objects_after = list_prefix()
    with get_db_connection() as conn:
        with conn.cursor() as cur:
            cur.execute("SELECT id, storage_path FROM documents;")
            db_docs = cur.fetchall()
            doc_id_set = {d["id"] for d in db_docs}
            doc_storage_paths = {d["storage_path"] for d in db_docs if d["storage_path"]}

            cur.execute("SELECT COUNT(*) as cnt FROM documents;")
            final_doc_count = cur.fetchone()["cnt"]

            cur.execute("SELECT COUNT(*) as cnt FROM document_chunks;")
            final_chunk_count = cur.fetchone()["cnt"]

            cur.execute("""
                SELECT COUNT(*) as cnt FROM document_chunks c
                WHERE NOT EXISTS (SELECT 1 FROM documents d WHERE d.id = c.document_id);
            """)
            orphaned_chunks = cur.fetchone()["cnt"]

    orphaned_storage_after = []
    for obj in all_objects_after:
        parts = obj.split("/")
        doc_part = next((p for p in parts if p.startswith("doc_")), None)
        if doc_part:
            if doc_part not in doc_id_set and obj not in doc_storage_paths:
                orphaned_storage_after.append(obj)
        else:
            if obj not in doc_storage_paths:
                orphaned_storage_after.append(obj)

    print("\n" + "=" * 60)
    print("FINAL POST-CLEANUP COUNTS:")
    print("=" * 60)
    print(f"Orphaned Storage objects:        {len(orphaned_storage_after)}")
    print(f"Orphaned pgvector chunks:        {orphaned_chunks}")
    print(f"Production document count:       {final_doc_count} (unchanged: {final_doc_count == initial_doc_count})")
    print(f"Production pgvector chunk count: {final_chunk_count} (unchanged: {final_chunk_count == initial_chunk_count})")
    print(f"Total Supabase Storage objects:  {len(all_objects_after)}")
    print("=" * 60)

if __name__ == "__main__":
    main()
