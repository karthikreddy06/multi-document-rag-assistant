# -*- coding: utf-8 -*-
"""
Production Cleanup Script.

Removes Supabase Storage objects and pgvector chunks belonging ONLY to
confirmed test/fixture documents. Does NOT touch production documents.

Verified test documents (from audit_production_docs.py + inspect_ambiguous.py):
  - Alice_in_Wonderland.pdf  (backend/documents/ fixture, referenced in retrieval tests)
  - oldmansea.pdf            (backend/documents/ fixture, referenced in retrieval tests)
  - Dataset.XLSX             (large benchmark/dataset, 6.7 MB, 17311 chunks)
  - notes.txt                (synthetic test file, 0 bytes)
  - test.pdf                 (synthetic test file, <1 KB)
  - test.txt                 (synthetic test file, 0 bytes)
  - scores.csv               (tiny 27-byte test fixture)
  - resume.pdf (11 records)  (SSE streaming test fixtures, file_hash=hash_sse_*, storage_path=/tmp)
  - staff.csv (2 records)    (49-byte test fixture, old uploads/ path)

Production documents preserved:
  - Karthik_Reddy_DataScientist_Resume.pdf
  - KarthikReddy_Resume.pdf
  - 12thmarksheet (1).pdf
  - 10thmarksheet (1).pdf
  - Astha_Bisoi_Resume1.pdf
  - oracle java se17 - ScoreReport(M.Karthik reddy).pdf
  - AI_Driven_Collections_Strategy_Presentation.pptx
  - Run NVIDIA Nemotron 3 Ultra FREE in Your Terminal.docx
  - python_number_programs_20.pdf
  - marine_protected_area_clean.csv
  - image.jpg
  - passwd.pdf
  - Apocolypse Food Prep.xlsx
  - hermes_classified.txt
  - zeus_classified.txt
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

DRY_RUN = "--execute" not in sys.argv
if DRY_RUN:
    print("DRY RUN mode. Pass --execute to perform actual deletions.")
    print("Re-run with: .venv311\\Scripts\\python backend/scripts/cleanup_test_data.py --execute")
else:
    print("EXECUTE mode. Deleting test data from Supabase Storage and pgvector.")
print()

# ─── Confirmed test document IDs ───────────────────────────────────────────────
# doc_id -> storage_path (for docs that have Supabase Storage objects to remove)
TEST_DOCS_WITH_STORAGE = {
    "doc_d212852d1459444a": "users/legacy_user/documents/doc_d212852d1459444a/Alice_in_Wonderland.pdf",
    "doc_a32aa6fb75bf4aab": "users/legacy_user/documents/doc_a32aa6fb75bf4aab/Dataset.XLSX",
    "doc_4fa1001daddd45c3": "users/legacy_user/documents/doc_4fa1001daddd45c3/notes.txt",
    "doc_992c2b4ae15c4baf": "users/legacy_user/documents/doc_992c2b4ae15c4baf/oldmansea.pdf",
    "doc_2ee8e65b4dd6444e": "users/legacy_user/documents/doc_2ee8e65b4dd6444e/test.pdf",
    "doc_a5b5c8df0cde4ae3": "users/legacy_user/documents/doc_a5b5c8df0cde4ae3/test.txt",
    "doc_3665c7ff403f4159": "users/legacy_user/documents/doc_3665c7ff403f4159/scores.csv",
}

# SSE streaming test records — storage_path=/tmp (not real Supabase objects)
RESUME_SSE_DOC_IDS = [
    "doc_7e74afd0a01b477c", "doc_1a125b70f6ff47ec", "doc_429bdb455a904f43",
    "doc_cfd1ddb1700e4757", "doc_b30fa1160ceb487f", "doc_d5bcf5e865af4a86",
    "doc_9296a85cca8d460a", "doc_1a1d617ad8db4135", "doc_2273cab7d28d47cb",
    "doc_21418f63a11543d7", "doc_83ff5fe9189f47a7",
]

# staff.csv records — old uploads/ path (not real Supabase objects)
STAFF_CSV_DOC_IDS = [
    "doc_282bd872c00ce676",  # staff.csv legacy_user
    "doc_4dc8e805a0b176af",  # staff.csv usr_47017e7be1184d38
]

# All test doc IDs for pgvector and postgres update
ALL_TEST_DOC_IDS = (
    list(TEST_DOCS_WITH_STORAGE.keys()) +
    RESUME_SSE_DOC_IDS +
    STAFF_CSV_DOC_IDS
)


def supabase_bulk_delete(paths: list, dry_run: bool) -> bool:
    """Bulk-delete objects from Supabase Storage using the REST API."""
    key = settings.effective_supabase_secret_key
    if not key:
        print("  ERROR: SUPABASE_SECRET_KEY not configured.")
        return False
    if not paths:
        return True

    bucket = settings.supabase_storage_bucket
    # Supabase supports bulk delete: DELETE /storage/v1/object/{bucket}
    # with JSON body {"prefixes": ["path1", "path2"]}
    url = f"{settings.supabase_url.rstrip('/')}/storage/v1/object/{bucket}"
    headers = {
        "Authorization": f"Bearer {key}",
        "apikey": key,
        "Content-Type": "application/json",
    }
    payload = {"prefixes": paths}

    if dry_run:
        print(f"  [DRY RUN] Would bulk-delete {len(paths)} objects from '{bucket}':")
        for p in paths:
            print(f"    - {p!r}")
        return True

    try:
        with httpx.Client(timeout=30.0) as client:
            resp = client.request("DELETE", url, headers=headers, content=json.dumps(payload))
            if resp.status_code in (200, 204):
                print(f"  Deleted {len(paths)} objects from Supabase Storage bucket '{bucket}'")
                return True
            else:
                print(f"  Storage bulk-delete returned {resp.status_code}: {resp.text}")
                # Fallback: individual deletes
                print("  Falling back to individual deletes...")
                success = 0
                for p in paths:
                    obj_url = f"{settings.supabase_url.rstrip('/')}/storage/v1/object/{bucket}/{p}"
                    r = client.delete(obj_url, headers=headers)
                    if r.status_code in (200, 204):
                        success += 1
                        print(f"    Deleted: {p!r}")
                    else:
                        print(f"    Failed ({r.status_code}): {p!r} -> {r.text[:100]}")
                print(f"  Individual deletes: {success}/{len(paths)} succeeded")
                return success == len(paths)
    except Exception as e:
        print(f"  Storage delete error: {e}")
        return False


# ─── Step 1: Remove Supabase Storage objects ─────────────────────────────────
print("=" * 60)
print("STEP 1: Remove test files from Supabase Storage")
print("=" * 60)
storage_paths = list(TEST_DOCS_WITH_STORAGE.values())
supabase_bulk_delete(storage_paths, DRY_RUN)

# ─── Step 2: Remove pgvector chunks ──────────────────────────────────────────
print()
print("=" * 60)
print("STEP 2: Remove test pgvector chunks")
print("=" * 60)
with get_db_connection() as conn:
    with conn.cursor() as cur:
        cur.execute(
            "SELECT COUNT(*) as c FROM document_chunks WHERE document_id = ANY(%s)",
            (ALL_TEST_DOC_IDS,)
        )
        count = cur.fetchone()["c"]
        if DRY_RUN:
            print(f"  [DRY RUN] Would delete {count} pgvector chunks for {len(ALL_TEST_DOC_IDS)} test doc IDs")
        else:
            cur.execute(
                "DELETE FROM document_chunks WHERE document_id = ANY(%s) RETURNING id",
                (ALL_TEST_DOC_IDS,)
            )
            deleted = len(cur.fetchall())
            conn.commit()
            print(f"  Deleted {deleted} pgvector chunks")

# ─── Step 3: Clear invalid storage_path values in PostgreSQL ─────────────────
print()
print("=" * 60)
print("STEP 3: Clear test doc storage_path values in PostgreSQL")
print("=" * 60)
with get_db_connection() as conn:
    with conn.cursor() as cur:
        cur.execute(
            "SELECT id, filename, storage_path FROM documents WHERE id = ANY(%s)",
            (ALL_TEST_DOC_IDS,)
        )
        rows = cur.fetchall()
        if DRY_RUN:
            for r in rows:
                print(f"  [DRY RUN] Would set storage_path=NULL for {r['filename']!r} (id={r['id']})")
        else:
            cur.execute(
                "UPDATE documents SET storage_path = NULL WHERE id = ANY(%s)",
                (ALL_TEST_DOC_IDS,)
            )
            conn.commit()
            print(f"  Cleared storage_path for {len(rows)} test document records")

print()
if DRY_RUN:
    print("=" * 60)
    print("DRY RUN complete. Pass --execute to apply changes.")
    print("=" * 60)
else:
    print("=" * 60)
    print("CLEANUP COMPLETE")
    print("=" * 60)
