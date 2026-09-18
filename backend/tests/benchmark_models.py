import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import time
import ollama
from app.main import RAGApplication
from app.generation.prompts import build_rag_prompt

rag = RAGApplication(auto_ingest=False)

test_questions = [
    "Tell me about Karthik.",
    "What are Karthik's technical skills?",
    "What projects has Karthik worked on?"
]

models = ["llama3.2:latest", "llama3.2:1b"]

print("=" * 80)
print("MODEL & PROMPT BENCHMARK: llama3.2 (3B) vs llama3.2:1b (1B)")
print("=" * 80)

for model in models:
    print(f"\n==================== MODEL: {model} ====================")
    for q in test_questions:
        print(f"\n>>> QUESTION: '{q}'")
        
        t_ret0 = time.perf_counter()
        chunks = rag.retriever.retrieve(q, top_k=4)
        t_ret = (time.perf_counter() - t_ret0) * 1000
        
        prompt = build_rag_prompt(q, chunks)
        
        t0 = time.perf_counter()
        res = ollama.generate(
            model=model,
            prompt=prompt,
            options={
                "temperature": 0.0,
                "num_predict": 150,
                "num_ctx": 2048
            }
        )
        t_gen = time.perf_counter() - t0
        
        eval_count = res.get("eval_count", 0)
        eval_duration = res.get("eval_duration", 0) / 1e9
        prompt_eval_count = res.get("prompt_eval_count", 0)
        prompt_eval_duration = res.get("prompt_eval_duration", 0) / 1e9
        tok_s = eval_count / eval_duration if eval_duration > 0 else 0
        
        ans = res.get("response", "").strip()
        
        print(f"Retrieval ({len(chunks)} chunks): {t_ret:.1f} ms")
        print(f"Prompt Tokens: {prompt_eval_count} in {prompt_eval_duration:.2f}s")
        print(f"LLM Gen Time:  {t_gen:.2f} s ({eval_count} tokens @ {tok_s:.1f} tok/s)")
        print(f"Total Time:    {t_ret/1000 + t_gen:.2f} s")
        print(f"Answer Preview:\n{ans[:200]}...")
