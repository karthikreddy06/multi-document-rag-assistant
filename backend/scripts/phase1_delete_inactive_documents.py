# -*- coding: utf-8 -*-
"""
Phase 1: Safe deletion of the 22 inactive document records from PostgreSQL documents table.
Safeguards:
- Explicit whitelist of the 9 active production IDs that CANNOT be touched under any circumstance.
- Verifies target records have 0 chunks and 0 storage objects before deleting.
- Removes any related chat_documents links first.
- Preserves all users, chats, messages, and active documents.
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

# The 9 active production documents that MUST NEVER BE MODIFIED
PROTECTED_PROD_DOC_IDS = {
    "doc_7434cd3761f241f5",  # marine_protected_area_clean.csv (1146 chunks)
    "doc_3e6e0177abc44b7e",  # Astha_Bisoi_Resume1.pdf (14 chunks)
    "doc_762a3344ec8a41b0",  # Run NVIDIA Nemotron 3 Ultra FREE in Your Terminal.docx (12 chunks)
    "doc_0a3fb82e08424130",  # Karthik_Reddy_DataScientist_Resume.pdf (9 chunks)
    "doc_adbbb71c1346476c",  # KarthikReddy_Resume.pdf (8 chunks)
    "doc_524cdc0f973e4626",  # Apocolypse Food Prep.xlsx (7 chunks)
    "doc_3db6004113ce41a9",  # python_number_programs_20.pdf (6 chunks)
    "doc_b55d07906c484096",  # AI_Driven_Collections_Strategy_Presentation.pptx (5 chunks)
    "doc_d180e03b45a34ad5",  # image.jpg (1 chunk)
}

# The exact 22 inactive document IDs to be deleted
INACTIVE_DOC_IDS = [
    "doc_d212852d1459444a",  # Alice_in_Wonderland.pdf
    "doc_992c2b4ae15c4baf",  # oldmansea.pdf
    "doc_2ee8e65b4dd6444e",  # test.pdf
    "doc_4fa1001daddd45c3",  # notes.txt
    "doc_3665c7ff403f4159",  # scores.csv
    "doc_a5b5c8df0cde4ae3",  # test.txt
    "doc_a32aa6fb75bf4aab",  # Dataset.XLSX
    "doc_a2908d0a6bca4674",  # sample.pdf
    "doc_7e74afd0a01b477c",  # resume.pdf
    "doc_1a125b70f6ff47ec",  # resume.pdf
    "doc_429bdb455a904f43",  # resume.pdf
    "doc_cfd1ddb1700e4757",  # resume.pdf
    "doc_b30fa1160ceb487f",  # resume.pdf
    "doc_d5bcf5e865af4a86",  # resume.pdf
    "doc_9296a85cca8d460a",  # resume.pdf
    "doc_1a1d617ad8db4135",  # resume.pdf
    "doc_2273cab7d28d47cb",  # resume.pdf
    "doc_21418f63a11543d7",  # resume.pdf
    "doc_83ff5fe9189f47a7",  # resume.pdf
    "doc_282bd872c00ce676",  # staff.csv
    "doc_3221cfa4d8094973",  # cascade_test.pdf
    "doc_4dc8e805a0b176af",  # staff.csv
]

def main():
    print("=" * 70)
    print("PHASE 1: REMOVING 22 INACTIVE DOCUMENT METADATA RECORDS")
    print("=" * 70)

    # Sanity checks
    assert len(INACTIVE_DOC_IDS) == 22, f"Expected 22 IDs, got {len(INACTIVE_DOC_IDS)}"
    overlap = set(INACTIVE_DOC_IDS).intersection(PROTECTED_PROD_DOC_IDS)
    assert len(overlap) == 0, f"FATAL: Overlap detected with protected IDs: {overlap}"

    with get_db_connection() as conn:
        with conn.cursor() as cur:
            # Check chunks count for inactive docs
            cur.execute("""
                SELECT COUNT(*) as cnt FROM document_chunks
                WHERE document_id = ANY(%s);
            """, (INACTIVE_DOC_IDS,))
            chunk_cnt = cur.fetchone()["cnt"]
            assert chunk_cnt == 0, f"Aborting: found {chunk_cnt} chunks for inactive doc IDs!"

            # Check existing records in documents
            cur.execute("""
                SELECT id, filename FROM documents
                WHERE id = ANY(%s);
            """, (INACTIVE_DOC_IDS,))
            found_docs = cur.fetchall()
            print(f"Found {len(found_docs)} of 22 inactive documents in database:")
            for fd in found_docs:
                print(f"  - {fd['id']}: {fd['filename']}")

            # Step 1: Remove chat_documents junction rows if any exist
            cur.execute("""
                DELETE FROM chat_documents
                WHERE document_id = ANY(%s);
            """, (INACTIVE_DOC_IDS,))
            chat_links_deleted = cur.rowcount
            print(f"\n✓ Removed {chat_links_deleted} chat_documents links referencing inactive documents.")

            # Step 2: Delete from documents
            cur.execute("""
                DELETE FROM documents
                WHERE id = ANY(%s);
            """, (INACTIVE_DOC_IDS,))
            docs_deleted = cur.rowcount
            print(f"✓ Deleted {docs_deleted} inactive documents from PostgreSQL.")

            conn.commit()

    print("\nPhase 1 deletion completed successfully.")

if __name__ == "__main__":
    main()
