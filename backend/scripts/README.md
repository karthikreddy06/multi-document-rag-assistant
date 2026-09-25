# Backend Operational & Migration Scripts

This directory contains database migration, operational maintenance, and audit scripts for the Multi-Document RAG Assistant.

## Reusable Operational & Validation Scripts

| Script | Purpose | When to Run |
| :--- | :--- | :--- |
| `phase2_verify_clean_state.py` | Full production integrity audit (documents, pgvector chunks, storage objects, active RAG query test). | Pre-deployment & post-deployment health checks. |
| `verify_final_production.py` | Quick audit of PostgreSQL documents and chunks vs Supabase Storage. | Routine production verification. |
| `migrate_documents_to_supabase.py` | Document storage & vector migration pipeline with explicit test-fixture exclusion list. | When migrating new local documents to Supabase. |

## Completed Migration & Cleanup Scripts (Archived)

These scripts were executed during the Supabase Storage and pgvector production migration and cleanup. They have completed their purpose and should not be re-run against production:

* `phase1_delete_inactive_documents.py` — Removed the 22 inactive/historical metadata records from PostgreSQL `documents`.
* `clean_orphaned_storage.py` — Purged ephemeral test fixture objects (`zeus.txt`, `hermes.txt`, `user_a_plan.txt`) from Supabase Storage.
* `execute_targeted_cleanup.py` — Removed sensitive/personal documents and test files from production corpus.
* `purge_test_storage_objects.py` — Bulk deleted historical test fixture files from Supabase Storage.
* `cleanup_test_data.py` — Initial test dataset cleanup script.
* `index_xlsx_batched.py` — Batched indexer for Dataset.XLSX (Permanently blocked with `SystemExit` guard to protect production).
* `migrate_sqlite_to_supabase.py` — One-time SQLite to Supabase relational schema migration.
* `audit_31_documents.py`, `audit_targeted_cleanup.py`, `audit_production_docs.py`, `inspect_ambiguous.py`, `check_state.py` — Diagnostic audit scripts.
