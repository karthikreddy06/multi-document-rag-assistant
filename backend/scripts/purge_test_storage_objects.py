# -*- coding: utf-8 -*-
"""
Delete confirmed test fixture objects from Supabase Storage bucket.
Safeguards:
- STRICT whitelist of real production documents that must NEVER be touched.
- Only objects belonging to test/benchmark files are deleted.
"""
import sys, os, json
sys.stdout.reconfigure(encoding='utf-8')
from pathlib import Path

BACKEND_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(BACKEND_DIR))
os.chdir(str(BACKEND_DIR))

from dotenv import load_dotenv
load_dotenv(".env")

from app.config import settings
import httpx

# Real production documents - MUST NEVER BE TOUCHED
PROTECTED_PROD_DOC_PATHS = {
    "users/legacy_user/documents/doc_0a3fb82e08424130/Karthik_Reddy_DataScientist_Resume.pdf",
    "users/legacy_user/documents/doc_2371a21aa1034434/passwd.pdf",
    "users/legacy_user/documents/doc_29d5957f740a4ecf/oracle java se17 - ScoreReport(M.Karthik reddy).pdf",
    "users/legacy_user/documents/doc_3db6004113ce41a9/python_number_programs_20.pdf",
    "users/legacy_user/documents/doc_3e6e0177abc44b7e/Astha_Bisoi_Resume1.pdf",
    "users/legacy_user/documents/doc_7434cd3761f241f5/marine_protected_area_clean.csv",
    "users/legacy_user/documents/doc_762a3344ec8a41b0/Run NVIDIA Nemotron 3 Ultra FREE in Your Terminal.docx",
    "users/legacy_user/documents/doc_adbbb71c1346476c/KarthikReddy_Resume.pdf",
    "users/legacy_user/documents/doc_b20f7a93133f424d/10thmarksheet (1).pdf",
    "users/legacy_user/documents/doc_b55d07906c484096/AI_Driven_Collections_Strategy_Presentation.pptx",
    "users/legacy_user/documents/doc_d180e03b45a34ad5/image.jpg",
    "users/legacy_user/documents/doc_d1928cdb40734c24/12thmarksheet (1).pdf",
}

# Explicit list of test storage objects to purge
TEST_STORAGE_OBJECTS = [
    "users/legacy_user/documents/doc_810f2b3f1c74de81/scores.csv",
    "users/legacy_user/documents/doc_c81699bf1bf468cf/notes.txt",
    "users/legacy_user/documents/doc_cbd17cddca2d17f6/test.pdf",
    "users/usr_38aa4dabc9974db7/documents/converted_doc_001cfcb95f87db8e/staff.xlsx",
    "users/usr_38aa4dabc9974db7/documents/doc_001cfcb95f87db8e/staff.csv",
    # test_security / test_upload_api mock files
    "users/legacy_user/documents/doc_485e39f99dc1bdae/calc.pdf",
    "users/legacy_user/documents/doc_485e39f99dc1bdae/chat_b_doc.pdf",
    "users/legacy_user/documents/doc_485e39f99dc1bdae/cmd.pdf",
    "users/legacy_user/documents/doc_485e39f99dc1bdae/first_upload.pdf",
    "users/legacy_user/documents/doc_485e39f99dc1bdae/first_upload_renamed.pdf",
    "users/legacy_user/documents/doc_485e39f99dc1bdae/nested.pdf",
    "users/legacy_user/documents/doc_485e39f99dc1bdae/passwd.pdf",
    "users/legacy_user/documents/doc_485e39f99dc1bdae/shadow.pdf",
    "users/legacy_user/documents/doc_485e39f99dc1bdae/shared.pdf",
]

def purge_storage():
    bucket = settings.supabase_storage_bucket
    key = settings.effective_supabase_secret_key
    headers = {
        "apikey": key,
        "Authorization": f"Bearer {key}",
        "Content-Type": "application/json",
    }

    # Filter out protected
    to_delete = [p for p in TEST_STORAGE_OBJECTS if p not in PROTECTED_PROD_DOC_PATHS]
    print(f"Purging {len(to_delete)} test objects from bucket '{bucket}' via bulk delete...")

    url = f"{settings.supabase_url.rstrip('/')}/storage/v1/object/{bucket}"
    payload = {"prefixes": to_delete}

    with httpx.Client(timeout=30.0) as client:
        resp = client.request("DELETE", url, headers=headers, content=json.dumps(payload))
        if resp.status_code in (200, 204):
            print(f"  ✓ Successfully bulk-deleted {len(to_delete)} objects.")
            for p in to_delete:
                print(f"    - {p}")
        else:
            print(f"  Bulk delete returned {resp.status_code}: {resp.text}")
            print("  Retrying individual DELETE without Content-Type header...")
            indiv_headers = {
                "apikey": key,
                "Authorization": f"Bearer {key}",
            }
            deleted = 0
            for p in to_delete:
                single_url = f"{settings.supabase_url.rstrip('/')}/storage/v1/object/{bucket}/{p}"
                r = client.delete(single_url, headers=indiv_headers)
                if r.status_code in (200, 204):
                    print(f"    ✓ Deleted: {p}")
                    deleted += 1
                else:
                    print(f"    ✗ Failed ({r.status_code}): {p} -> {r.text[:100]}")
            print(f"  Individual fallback deleted: {deleted}/{len(to_delete)}")

if __name__ == "__main__":
    purge_storage()
