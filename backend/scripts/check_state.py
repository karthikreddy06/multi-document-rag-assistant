import sys, os
from pathlib import Path

BACKEND_DIR = Path(__file__).resolve().parent.parent  # scripts/ -> backend/
sys.path.insert(0, str(BACKEND_DIR))
os.chdir(str(BACKEND_DIR))

from dotenv import load_dotenv
load_dotenv(".env")

from app.db.database import get_db_connection

with get_db_connection() as conn:
    with conn.cursor() as cur:
        cur.execute("SELECT COUNT(*) as total FROM document_chunks;")
        total_chunks = cur.fetchone()["total"]
        cur.execute("SELECT COUNT(DISTINCT document_id) as docs FROM document_chunks;")
        distinct_docs = cur.fetchone()["docs"]
        cur.execute("SELECT COUNT(*) as cnt FROM documents WHERE storage_path LIKE 'users/%';")
        in_storage = cur.fetchone()["cnt"]
        cur.execute("SELECT COUNT(*) as total_docs FROM documents;")
        total_docs = cur.fetchone()["total_docs"]
        cur.execute("SELECT COUNT(*) as cnt FROM documents WHERE (storage_path IS NULL OR storage_path NOT LIKE 'users/%');")
        not_migrated = cur.fetchone()["cnt"]

print("=== MIGRATION STATE ===")
print(f"pgvector chunks total:          {total_chunks}")
print(f"distinct doc_ids in pgvector:   {distinct_docs}")
print(f"docs with Supabase storage:     {in_storage} / {total_docs}")
print(f"docs NOT yet in Supabase:       {not_migrated}")
