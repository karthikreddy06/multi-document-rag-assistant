# -*- coding: utf-8 -*-
"""Read-only production vs test document audit."""
import sys, os
sys.stdout.reconfigure(encoding='utf-8')
from pathlib import Path

BACKEND_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(BACKEND_DIR))
os.chdir(str(BACKEND_DIR))

from dotenv import load_dotenv
load_dotenv(".env")
from app.db.database import get_db_connection

# ─── Test fixture classification ────────────────────────────────────────────
# Verified from:
#  - backend/documents/ directory (primary Chroma test corpus)
#  - test source files mentioning specific filenames
FIXTURE_FILENAMES = {
    "Alice_in_Wonderland.pdf",  # referenced in test_retrieval_generic.py, test_adaptive_retrieval.py
    "oldmansea.pdf",            # referenced in test_retrieval_generic.py, test_adaptive_retrieval.py
    "sample.pdf",               # referenced in test_retrieval_generic.py (pdflatex compilation)
    "company.pdf",              # referenced in test_adaptive_retrieval.py (company employment queries)
    "test.pdf",
    "test.txt",
    "notes.txt",
    "Dataset.XLSX",
    "Dataset.xlsx",
}

FIXTURE_KEYWORDS = ["dataset", "benchmark", "_test", "test_", "fixture", "dummy", "mock"]


def classify(filename):
    if filename in FIXTURE_FILENAMES:
        return "TEST", "explicit fixture list"
    fn_lower = filename.lower()
    for kw in FIXTURE_KEYWORDS:
        if kw in fn_lower:
            return "TEST", f"keyword '{kw}'"
    return "PROD", ""


with get_db_connection() as conn:
    with conn.cursor() as cur:
        cur.execute(
            "SELECT id, filename, file_size, status, storage_path, chunk_count, user_id "
            "FROM documents ORDER BY filename"
        )
        all_docs = cur.fetchall()

        cur.execute("SELECT document_id, COUNT(*) as c FROM document_chunks GROUP BY document_id")
        chunks_by_id = {r["document_id"]: r["c"] for r in cur.fetchall()}

        cur.execute("SELECT COUNT(*) as t FROM document_chunks")
        total_chunks = cur.fetchone()["t"]

        # Distinct filenames in pgvector
        cur.execute(
            "SELECT filename, COUNT(*) as c FROM document_chunks GROUP BY filename ORDER BY c DESC"
        )
        pg_by_file = cur.fetchall()

SEP = "=" * 72
sep = "-" * 72

print(SEP)
print("  PRODUCTION vs TEST DOCUMENT AUDIT")
print(SEP)
print(f"  PostgreSQL documents: {len(all_docs)}")
print(f"  pgvector chunks:      {total_chunks}")
print()

prod_docs = []
test_docs = []

for doc in all_docs:
    label, reason = classify(doc["filename"])
    pg_chunks = chunks_by_id.get(doc["id"], 0)
    has_storage = bool(doc["storage_path"] and "users/" in doc["storage_path"])
    entry = {
        "id": doc["id"],
        "filename": doc["filename"],
        "size_kb": doc["file_size"] / 1024,
        "status": doc["status"],
        "storage_path": doc["storage_path"],
        "has_storage": has_storage,
        "pg_chunks": pg_chunks,
        "reason": reason,
    }
    if label == "TEST":
        test_docs.append(entry)
    else:
        prod_docs.append(entry)

print(sep)
print(f"  PRODUCTION DOCUMENTS ({len(prod_docs)})")
print(sep)
for d in prod_docs:
    st = "STORAGE OK" if d["has_storage"] else "NO STORAGE"
    chk = f"{d['pg_chunks']:5d} chunks" if d["pg_chunks"] > 0 else "   0 chunks (not indexed)"
    print(f"  [{st}] [{chk}] {d['filename']!r}  ({d['size_kb']:.1f} KB)")

print()
print(sep)
print(f"  TEST / DATASET DOCUMENTS ({len(test_docs)})  -- must be excluded from production")
print(sep)
for d in test_docs:
    issues = []
    if d["has_storage"]:
        issues.append(f"IN SUPABASE STORAGE: {d['storage_path']}")
    if d["pg_chunks"] > 0:
        issues.append(f"{d['pg_chunks']} pgvector chunks to DELETE")
    issue_str = " | ".join(issues) if issues else "Clean (no production data)"
    print(f"  {d['filename']!r}  ({d['size_kb']:.1f} KB)  [{issue_str}]  (reason: {d['reason']})")

print()
print(sep)
print("  PGVECTOR CHUNKS BY FILENAME")
print(sep)
for row in pg_by_file:
    fn = row["filename"] or "(null)"
    label, _ = classify(fn)
    tag = "TEST" if label == "TEST" else "PROD"
    print(f"  [{tag}]  {row['c']:6d} chunks  {fn!r}")

print()
print(SEP)
print("  SUMMARY")
print(SEP)
prod_ok_storage = sum(1 for d in prod_docs if d["has_storage"])
prod_ok_chunks = sum(1 for d in prod_docs if d["pg_chunks"] > 0)
prod_total_chunks = sum(d["pg_chunks"] for d in prod_docs)
test_bad_storage = sum(1 for d in test_docs if d["has_storage"])
test_bad_chunks = sum(d["pg_chunks"] for d in test_docs)

print(f"  Production docs in PostgreSQL:       {len(prod_docs)}")
print(f"  Production docs in Supabase Storage: {prod_ok_storage} / {len(prod_docs)}")
print(f"  Production docs with pgvector index: {prod_ok_chunks} / {len(prod_docs)}")
print(f"  Production pgvector chunk total:     {prod_total_chunks}")
print()
print(f"  Test/dataset docs in PostgreSQL:     {len(test_docs)}")
print(f"  Test docs erroneously in Storage:    {test_bad_storage}")
print(f"  Test pgvector chunks to REMOVE:      {test_bad_chunks}")
missing_prod = [d for d in prod_docs if not d["has_storage"]]
missing_indexed = [d for d in prod_docs if d["pg_chunks"] == 0]
print(f"  Production docs missing Storage:     {len(missing_prod)}")
print(f"  Production docs missing pgvector:    {len(missing_indexed)}")
if missing_prod:
    print()
    print("  MISSING FROM STORAGE:")
    for d in missing_prod:
        print(f"    {d['filename']!r}")
if missing_indexed:
    print()
    print("  NOT YET INDEXED IN PGVECTOR:")
    for d in missing_indexed:
        print(f"    {d['filename']!r}")
print(SEP)
