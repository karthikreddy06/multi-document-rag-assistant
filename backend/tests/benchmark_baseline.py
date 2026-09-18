import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import time
import requests
from app.main import RAGApplication
from app.retrieval.retriever import QueryDecomposer
from app.generation.prompts import build_rag_prompt

rag = RAGApplication(auto_ingest=False)

questions = [
    "Tell me about Karthik.",
    "What are Karthik's technical skills?",
    "What projects has Karthik worked on?"
]

print("=" * 70)
print("BASELINE PERFORMANCE MEASUREMENT")
print("=" * 70)

for q in questions:
    print(f"\n--- QUESTION: '{q}' ---")
    t0 = time.perf_counter()
    
    # 1. Decomposition
    t_decomp_start = time.perf_counter()
    subqueries = QueryDecomposer.decompose(q)
    t_decomp = (time.perf_counter() - t_decomp_start) * 1000
    
    # 2. Embedding time alone
    t_emb_start = time.perf_counter()
    q_emb = rag.retriever.embedding_service.embed_text(q)
    t_emb = (time.perf_counter() - t_emb_start) * 1000
    
    # 3. Dense search time
    t_dense_start = time.perf_counter()
    dense_res = rag.retriever.vector_store.query(q_emb, top_k=5)
    t_dense = (time.perf_counter() - t_dense_start) * 1000
    
    # 4. Full retrieval (dense + lexical + decomposition + fusion)
    t_ret_start = time.perf_counter()
    chunks = rag.retriever.retrieve(q)
    t_ret = (time.perf_counter() - t_ret_start) * 1000
    
    # 5. Prompt construction
    t_prompt_start = time.perf_counter()
    prompt = build_rag_prompt(q, chunks)
    t_prompt = (time.perf_counter() - t_prompt_start) * 1000
    
    # 6. Ollama Generation with exact token metrics from Ollama API
    t_gen_start = time.perf_counter()
    res = rag.generator.client.generate(
        model=rag.generator.model,
        prompt=prompt,
        options={"temperature": 0.0}
    )
    t_gen = time.perf_counter() - t_gen_start
    t_total = time.perf_counter() - t0
    
    ans = res.get("response", "").strip()
    eval_count = res.get("eval_count", 0)
    eval_duration = res.get("eval_duration", 0) / 1e9
    prompt_eval_count = res.get("prompt_eval_count", 0)
    prompt_eval_duration = res.get("prompt_eval_duration", 0) / 1e9
    tok_per_sec = eval_count / eval_duration if eval_duration > 0 else 0
    
    print(f"1. Query Decomposition:   {t_decomp:6.2f} ms (subqueries: {subqueries})")
    print(f"2. Query Embedding:       {t_emb:6.2f} ms")
    print(f"3. Dense Vector Search:   {t_dense:6.2f} ms")
    print(f"4. Total Retrieval (RRF): {t_ret:6.2f} ms ({len(chunks)} chunks)")
    print(f"5. Prompt Construction:   {t_prompt:6.2f} ms ({len(prompt)} chars, ~{prompt_eval_count} prompt tokens)")
    print(f"6. Ollama Generation:     {t_gen:6.2f} s")
    print(f"   -> Tokens Generated:   {eval_count} tokens")
    print(f"   -> Prompt Eval Time:   {prompt_eval_duration:6.2f} s")
    print(f"   -> Eval Generation:    {eval_duration:6.2f} s ({tok_per_sec:4.1f} tokens/sec)")
    print(f"7. Total Response Time:   {t_total:6.2f} s")
    print(f"Answer ({len(ans.split())} words):\n{ans[:250]}...")
