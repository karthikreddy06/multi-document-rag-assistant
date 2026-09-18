"""
Proof of concept test for Document-Wide Retrieval with python_number_programs_20.pdf.
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from app.main import RAGApplication
from app.models import RetrievedChunk
from app.generation.prompts import build_rag_prompt

rag = RAGApplication(auto_ingest=False)

# 1. Fetch all chunks of python_number_programs_20.pdf
res = rag.vector_store.collection.get(
    where={"filename": "python_number_programs_20.pdf"},
    include=["metadatas", "documents"]
)

raw_chunks = []
for doc_text, meta in zip(res["documents"], res["metadatas"]):
    raw_chunks.append(RetrievedChunk(
        text=doc_text,
        metadata=meta,
        score=1.0
    ))

# Sort sequentially by page_number, then chunk_index
raw_chunks.sort(key=lambda c: (c.metadata.get("page_number", 1), c.metadata.get("chunk_index", 0)))
print(f"Loaded {len(raw_chunks)} sequential chunks across pages: {[c.metadata.get('page_number') for c in raw_chunks]}")

queries = [
    "What are all the programs in this document?",
    "Tell me all programs except the first and second",
    "Explain all programs in python_number_programs_20.pdf"
]

for q in queries:
    print("\n" + "=" * 80)
    print(f"QUERY: {q}")
    print("=" * 80)
    
    prompt = build_rag_prompt(q, raw_chunks)
    ans = rag.generator.generate_answer(q, raw_chunks)
    print("ANSWER:")
    print(ans)
