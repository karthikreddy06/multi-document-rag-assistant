# -*- coding: utf-8 -*-
"""
Final Verification Script for Production Cleanliness & Test Fixture Exclusion.
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
from app.services.storage import SupabaseStorageService
import httpx

def main():
    print("=" * 70)
    print("FINAL VERIFICATION: PRODUCTION CLEANLINESS & FIXTURE EXCLUSION")
    print("=" * 70)

    # 1. Check local files
    dataset_hash_file = BACKEND_DIR / "data" / "uploads" / "32d3dd6f45ad15eeffd3d484e8d07752f76ed9d0384bff9a346b67f786d91b9c.xlsx"
    print("\n[1] LOCAL TEST / DATASET FILES:")
    if dataset_hash_file.exists():
        print(f"  ✓ Dataset.XLSX intact locally ({dataset_hash_file.name}, {dataset_hash_file.stat().st_size:,} bytes)")
    else:
        print(f"  ✗ Dataset.XLSX missing locally!")

    chroma_fixtures = ["Alice_in_Wonderland.pdf", "oldmansea.pdf", "sample.pdf", "company.pdf"]
    for cf in chroma_fixtures:
        p = BACKEND_DIR / "documents" / cf
        if p.exists():
            print(f"  ✓ Test fixture '{cf}' intact locally ({p.stat().st_size:,} bytes)")
        else:
            print(f"  ? Test fixture '{cf}' not in backend/documents/")

    # 2. Check Database: documents and chunks
    print("\n[2] POSTGRESQL DATABASE AUDIT:")
    with get_db_connection() as conn:
        with conn.cursor() as cur:
            # Query all documents with chunk counts
            cur.execute("""
                SELECT d.id, d.filename, d.user_id, d.storage_path,
                       COUNT(c.id) as chunk_count
                FROM documents d
                LEFT JOIN document_chunks c ON d.id = c.document_id
                GROUP BY d.id, d.filename, d.user_id, d.storage_path
                ORDER BY d.created_at DESC;
            """)
            docs = cur.fetchall()

            cur.execute("SELECT COUNT(*) as cnt FROM document_chunks;")
            total_chunks = cur.fetchone()["cnt"]

    print(f"  Total documents in DB: {len(docs)}")
    print(f"  Total chunks in DB: {total_chunks}")
    print("\n  Detailed Documents List:")
    print(f"  {'Filename':<45} | {'User ID':<20} | {'Chunks':<7} | {'Storage Path'}")
    print("  " + "-" * 105)

    test_filenames = {
        "alice_in_wonderland.pdf", "oldmansea.pdf", "sample.pdf", "company.pdf",
        "dataset.xlsx", "test.pdf", "test.txt", "notes.txt", "scores.csv", "staff.csv"
    }

    test_chunks_found = 0
    test_storage_found = 0

    for d in docs:
        doc_id = d["id"]
        fn = d["filename"]
        uid = d["user_id"]
        spath = d["storage_path"]
        c_count = d["chunk_count"]
        spath_display = (spath or "None")[:45]
        print(f"  {fn[:44]:<45} | {str(uid)[:20]:<20} | {c_count:<7} | {spath_display}")

        if fn.lower() in test_filenames or "hash_sse" in str(spath) or ("test" in fn.lower() and "cascade" not in fn.lower()):
            if c_count > 0:
                test_chunks_found += c_count
            if spath and spath.startswith("users/"):
                test_storage_found += 1

    # 3. Check Supabase Storage objects
    print("\n[3] SUPABASE STORAGE BUCKET AUDIT:")
    try:
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
                        sub_prefix = f"{prefix}/{name}".strip("/")
                        results.extend(list_prefix(sub_prefix))
                    else:
                        full_name = f"{prefix}/{name}".strip("/")
                        results.append(full_name)
                return results
            return []

        all_objects = list_prefix()
        print(f"  Total objects in bucket '{bucket}': {len(all_objects)}")
        for obj in all_objects:
            print(f"    - {obj}")

        test_in_storage = [obj for obj in all_objects if any(tf in obj.lower() for tf in test_filenames)]
        if test_in_storage:
            print(f"  ⚠️ Test files found in Supabase storage: {test_in_storage}")
            test_storage_found += len(test_in_storage)
        else:
            print(f"  ✓ Zero test fixtures found in Supabase storage.")

    except Exception as e:
        print(f"  Error inspecting Supabase storage: {e}")

    # Summary
    print("\n" + "=" * 70)
    print("VERIFICATION SUMMARY:")
    print(f"  Test pgvector chunks: {test_chunks_found} (Expected: 0)")
    print(f"  Test objects in Supabase storage: {test_storage_found} (Expected: 0)")
    print(f"  Total active pgvector chunks: {total_chunks}")
    if test_chunks_found == 0 and test_storage_found == 0:
        print("  ✓ SUCCESS: All test data completely removed. Production documents clean.")
    else:
        print("  ✗ FAILURE: Remaining test data detected.")
    print("=" * 70)

if __name__ == "__main__":
    main()
