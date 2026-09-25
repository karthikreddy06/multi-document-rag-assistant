# -*- coding: utf-8 -*-
"""Deep-inspection of ambiguous documents."""
import sys, os
sys.stdout.reconfigure(encoding='utf-8')
from pathlib import Path

BACKEND_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(BACKEND_DIR))
os.chdir(str(BACKEND_DIR))

from dotenv import load_dotenv
load_dotenv(".env")
from app.db.database import get_db_connection

# Check resume.pdf records (11 copies), staff.csv, scores.csv, marine_protected_area_clean.csv
# Also check what Supabase objects exist for these
with get_db_connection() as conn:
    with conn.cursor() as cur:
        cur.execute("""
            SELECT id, filename, file_hash, file_size, status,
                   storage_path, chunk_count, created_at, user_id
            FROM documents
            WHERE filename IN ('resume.pdf','staff.csv','scores.csv',
                               'marine_protected_area_clean.csv','hermes_classified.txt',
                               'zeus_classified.txt','Apocolypse Food Prep.xlsx',
                               'passwd.pdf','image.jpg')
            ORDER BY filename, created_at
        """)
        rows = cur.fetchall()
        cur.execute("""
            SELECT document_id, COUNT(*) as c FROM document_chunks
            WHERE filename IN ('resume.pdf','staff.csv','scores.csv',
                               'marine_protected_area_clean.csv','hermes_classified.txt',
                               'zeus_classified.txt','Apocolypse Food Prep.xlsx',
                               'passwd.pdf','image.jpg')
            GROUP BY document_id
        """)
        chunks_by_id = {r["document_id"]: r["c"] for r in cur.fetchall()}

print("=== AMBIGUOUS DOCUMENTS DETAIL ===")
for r in rows:
    chk = chunks_by_id.get(r["id"], 0)
    sp = r["storage_path"] or "(none)"
    print(f"  filename:     {r['filename']!r}")
    print(f"  id:           {r['id']}")
    print(f"  user_id:      {r['user_id']}")
    print(f"  file_hash:    {r['file_hash'][:16]}...")
    print(f"  file_size:    {r['file_size']} bytes")
    print(f"  status:       {r['status']}")
    print(f"  storage_path: {sp}")
    print(f"  pg_chunks:    {chk}")
    print(f"  created_at:   {r['created_at']}")
    print()

# Check what local upload files exist for each
UPLOADS = Path("data/uploads")
print("=== LOCAL UPLOAD FILE SIZES ===")
for r in rows:
    fh = r["file_hash"]
    # find matching local file
    for ext in [".pdf", ".csv", ".txt", ".xlsx", ".jpg"]:
        candidate = UPLOADS / (fh + ext)
        if candidate.exists():
            print(f"  {r['filename']!r} -> LOCAL file: {candidate.name} ({candidate.stat().st_size} bytes)")
            break
    else:
        print(f"  {r['filename']!r} -> NO LOCAL FILE for hash {fh[:16]}...")
