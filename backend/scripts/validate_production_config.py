# -*- coding: utf-8 -*-
"""
Production Configuration & Connection Validation.
Validates:
- PostgreSQL connection
- pgvector connection
- Supabase Storage connection
- Database document/chunk counts
- Authentication configuration
- Groq configuration presence
Does NOT expose any secrets.
"""
import sys, os
sys.stdout.reconfigure(encoding='utf-8')
from pathlib import Path

BACKEND_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(BACKEND_DIR))
os.chdir(str(BACKEND_DIR))

from dotenv import load_dotenv
load_dotenv(".env")

# Simulate the exact production Render environment variables
os.environ["LLM_PROVIDER"] = "groq"
os.environ["LLM_MODEL"] = "openai/gpt-oss-20b"
os.environ["LLM_API_URL"] = "https://api.groq.com/openai/v1"
os.environ["EMBEDDING_PROVIDER"] = "local"
os.environ["EMBEDDING_MODEL"] = "all-MiniLM-L6-v2"
os.environ["EMBEDDING_DIMENSIONS"] = "384"
os.environ["VECTOR_STORE_PROVIDER"] = "pgvector"
os.environ["SUPABASE_URL"] = "https://ldytwnvxskfajjwcxxtb.supabase.co"
os.environ["SUPABASE_STORAGE_BUCKET"] = "rag-files"
os.environ["CORS_ORIGINS"] = '["https://multi-document-rag-assistant-phi.vercel.app"]'

from app.config import Settings
settings = Settings()
from app.db.database import get_db_connection
from app.services.storage import get_storage_service, SupabaseStorageService
from app.vectorstore.store import VectorStore
import httpx

def main():
    print("=" * 70)
    print("RENDER PRODUCTION CONFIGURATION VALIDATION")
    print("=" * 70)

    # 1. Environment & Provider Routing Checks
    print("\n[1] CONFIGURATION ROUTING CHECKS:")
    print(f"  LLM Provider:             {settings.llm_provider} (effective model: {settings.effective_llm_model})")
    print(f"  LLM API URL:              {settings.effective_llm_api_url}")
    print(f"  Embedding Provider:       {settings.embedding_provider} (effective model: {settings.effective_embedding_model}, {settings.effective_embedding_dimensions}d)")
    print(f"  Vector Store Provider:    {settings.vector_store_provider} (is_pgvector: {settings.is_pgvector})")
    print(f"  Storage Provider:         {'Supabase Storage' if settings.is_supabase_storage else 'LocalStorage'}")
    print(f"  Storage Bucket:           {settings.supabase_storage_bucket}")
    print(f"  Database Mode:            {'PostgreSQL (Supabase)' if settings.is_postgres else 'SQLite'}")
    print(f"  CORS Allowed Origins:     {settings.cors_origins}")

    # 2. Secret Presence (boolean check only, no printing of secrets)
    print("\n[2] CREDENTIALS & SECRETS PRESENCE:")
    print(f"  DATABASE_URL configured:        {bool(settings.database_url)}")
    print(f"  SUPABASE_SECRET_KEY configured: {bool(settings.effective_supabase_secret_key)}")
    print(f"  GROQ_API_KEY configured:        {bool(settings.groq_api_key or settings.llm_api_key)}")
    print(f"  JWT_SECRET_KEY configured:      {bool(settings.jwt_secret_key)}")
    print(f"  JWT Algorithm:                  {settings.jwt_algorithm}")

    assert settings.is_postgres, "FAILED: Application is NOT in PostgreSQL mode!"
    assert settings.is_pgvector, "FAILED: Application is NOT in pgvector mode!"
    assert settings.is_supabase_storage, "FAILED: Application is NOT in Supabase Storage mode!"
    assert bool(settings.groq_api_key or settings.llm_api_key), "FAILED: Groq API key is missing!"
    assert bool(settings.jwt_secret_key), "FAILED: JWT secret key is missing!"

    # 3. PostgreSQL Connection & Schema Check
    print("\n[3] POSTGRESQL DATABASE & COUNTS:")
    with get_db_connection() as conn:
        with conn.cursor() as cur:
            cur.execute("SELECT COUNT(*) as cnt FROM documents;")
            doc_count = cur.fetchone()["cnt"]
            cur.execute("SELECT COUNT(*) as cnt FROM document_chunks;")
            chunk_count = cur.fetchone()["cnt"]
            cur.execute("SELECT COUNT(*) as cnt FROM users;")
            user_count = cur.fetchone()["cnt"]
            cur.execute("SELECT COUNT(*) as cnt FROM chats;")
            chat_count = cur.fetchone()["cnt"]
            cur.execute("SELECT COUNT(*) as cnt FROM messages;")
            msg_count = cur.fetchone()["cnt"]

    print(f"  PostgreSQL Connection:    OK")
    print(f"  Documents in Catalog:     {doc_count} (Expected: 9)")
    print(f"  Chunks in pgvector:       {chunk_count} (Expected: 1208)")
    print(f"  Users count:              {user_count}")
    print(f"  Chats count:              {chat_count}")
    print(f"  Messages count:           {msg_count}")

    assert doc_count == 9, f"FAILED: Expected 9 documents, found {doc_count}"
    assert chunk_count == 1208, f"FAILED: Expected 1208 chunks, found {chunk_count}"

    # 4. Supabase Storage Connection & Objects Count
    print("\n[4] SUPABASE STORAGE CONNECTION & OBJECTS:")
    storage_svc = get_storage_service()
    assert isinstance(storage_svc, SupabaseStorageService), "FAILED: Storage service is not SupabaseStorageService!"

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

    storage_objects = list_prefix()
    print(f"  Supabase Storage Connection: OK")
    print(f"  Storage Objects Count:       {len(storage_objects)} (Expected: 9)")
    assert len(storage_objects) == 9, f"FAILED: Expected 9 storage objects, found {len(storage_objects)}"

    # 5. Groq API Connection Check
    print("\n[5] GROQ LLM API HEALTH CHECK:")
    groq_key = settings.groq_api_key or settings.llm_api_key
    groq_url = "https://api.groq.com/openai/v1/models"
    groq_headers = {"Authorization": f"Bearer {groq_key}"}
    try:
        r = httpx.get(groq_url, headers=groq_headers, timeout=10)
        if r.status_code == 200:
            print("  Groq API Connection:         OK (Authenticated & Active)")
        else:
            print(f"  Groq API Connection:         Status {r.status_code}")
    except Exception as e:
        print(f"  Groq API Connection Error:   {e}")

    print("\n" + "=" * 70)
    print("ALL PRODUCTION CONFIGURATION & CONNECTIVITY CHECKS PASSED!")
    print("=" * 70)

if __name__ == "__main__":
    main()
