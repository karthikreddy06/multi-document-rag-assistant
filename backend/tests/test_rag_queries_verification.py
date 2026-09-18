"""
Verification of 10 Critical RAG Queries for Phase 4.12 Regression Testing.
"""

import time
from app.main import RAGApplication

REGRESSION_QUERIES = [
    ("Alice in Wonderland main character", "What is the main character in Alice in Wonderland?"),
    ("Old Man and the Sea main character", "Who is the main character in The Old Man and the Sea?"),
    ("Santiago / 84 days query", "How many days did Santiago go without catching a fish in The Old Man and the Sea?"),
    ("Sample PDF specific question", "What is the date on the Sample PDF Document?"),
    ("Sample PDF broad question", "What is the Sample PDF document about?"),
    ("TravelTrack authentication/database question", "What authentication and database does TravelTrack use?"),
    ("SkillMatch project question", "What are the core features of the SkillMatch project?"),
    ("Cross-project database question", "What databases are used across the projects?"),
    ("Comparative/cross-document query", "Compare the main characters in Alice in Wonderland and The Old Man and the Sea."),
    ("Unsupported/favorite movie query", "What is the author's favorite movie in The Old Man and the Sea?"),
]

def run_regression_checks():
    app = RAGApplication(auto_ingest=False)
    results = []

    print("\n--- RUNNING 10 CRITICAL RAG REGRESSION CHECKS ---")
    for title, q in REGRESSION_QUERIES:
        t0 = time.time()
        chunks = app.retriever.retrieve(q, top_k=5)
        retrieval_ms = (time.time() - t0) * 1000
        filenames = sorted(list({str(c.metadata.get('filename')) for c in chunks}))

        # Quick check for retrieval coverage
        has_chunks = len(chunks) > 0
        print(f"[{title}] Retrieval: {retrieval_ms:.1f}ms | Chunks: {len(chunks)} | Files: {filenames}")
        results.append({
            "title": title,
            "query": q,
            "retrieval_ms": retrieval_ms,
            "chunk_count": len(chunks),
            "files": filenames,
            "status": "PASS" if has_chunks else "FAIL",
        })

    return results

if __name__ == "__main__":
    run_regression_checks()
