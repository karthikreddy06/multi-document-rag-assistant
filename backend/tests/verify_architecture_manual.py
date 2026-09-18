"""
Manual Architecture Verification Script for General-Purpose RAG Engine.
Executes diverse real queries across document types (exhaustive, comparison, ordinal,
multi-aspect, factual, corpus search, follow-up) and prints the full telemetry suite:
QUERY, QUERY TYPE, TARGET DOCUMENTS, RETRIEVAL PLAN, ASPECTS, CANDIDATE COUNT,
FINAL CHUNK COUNT, PAGE COVERAGE, DOCUMENT COVERAGE, CONTEXT SIZE, ANSWER, SOURCES, TOTAL TIME.
"""

import sys
import time
from pathlib import Path

# Add backend to python path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from app.main import RAGApplication
from app.models import RetrievedChunk


def run_query_benchmark(rag: RAGApplication, query: str, where=None, history=None):
    start = time.time()
    chunks, plan, coverage = rag.retriever.retrieve_adaptive(
        query=query,
        where=where,
        recent_messages=history,
    )

    answer = rag.generator.generate_answer(
        question=query,
        chunks=chunks,
        num_predict=plan.generation_budget.num_predict,
        num_ctx=plan.generation_budget.num_ctx,
    )
    elapsed = time.time() - start

    context_size = sum(len(c.text) for c in chunks)
    sources = [
        f"{c.metadata.get('filename')} (p.{c.metadata.get('page_number')}, s.{c.metadata.get('section')})"
        for c in chunks
    ]

    print("\n" + "=" * 90)
    print(f"QUERY: {query}")
    print(f"QUERY TYPE: {plan.strategy.value}")
    print(f"TARGET DOCUMENTS: {[d.filename for d in plan.target_documents]}")
    print(f"RETRIEVAL PLAN: {plan.strategy.name} (cand_k={plan.candidate_k}, final_k={plan.final_top_k})")
    print(f"ASPECTS: {plan.aspects}")
    print(f"CANDIDATE COUNT: {plan.candidate_k}")
    print(f"FINAL CHUNK COUNT: {len(chunks)}")
    print(f"PAGE COVERAGE: {coverage.pages_covered}")
    print(f"DOCUMENT COVERAGE: {coverage.documents_covered}")
    print(f"CONTEXT SIZE: {context_size} chars")
    print("-" * 90)
    print(f"ANSWER:\n{answer}")
    print("-" * 90)
    print(f"SOURCES ({len(sources)}): {sources[:8]}")
    print(f"TOTAL TIME: {elapsed:.2f}s")
    print("=" * 90)

    return answer, chunks, plan


def main():
    print("\nInitializing RAG Engine for Comprehensive Manual Architecture Testing...")
    rag = RAGApplication(auto_ingest=False)

    test_queries = [
        # 1. Exhaustive Document-Wide Query
        ("give me all programs", {"filename": "python_number_programs_20.pdf"}, None),

        # 2. Ordinal with Exclusions
        ("tell me all programs except the first and second", {"filename": "python_number_programs_20.pdf"}, None),

        # 3. Follow-up resolution
        (
            "explain the third one",
            {"filename": "python_number_programs_20.pdf"},
            [
                {"role": "user", "content": "What are the first three programs?"},
                {
                    "role": "assistant",
                    "content": (
                        "1. Prime Number Check\n"
                        "2. Factorial of a Number\n"
                        "3. Fibonacci Series up to N terms"
                    )
                }
            ]
        ),

        # 4. Cross-Document Comparison
        (
            "Compare the main characters and setting in Alice in Wonderland and The Old Man and the Sea.",
            None,
            None
        ),

        # 5. Multi-Aspect Query
        (
            "What is the background, methodology, and conclusion?",
            {"filename": "sample.pdf"},
            None
        ),

        # 6. Specific Factual Query
        (
            "Who is Santiago?",
            {"filename": "oldmansea.pdf"},
            None
        ),
    ]

    for q, where, hist in test_queries:
        try:
            run_query_benchmark(rag, q, where=where, history=hist)
        except Exception as e:
            print(f"[ERROR] Failed query '{q}': {e}")


if __name__ == "__main__":
    main()
