"""
Prompt Engineering and Guardrails Module.
Constructs strict, hallucination-resistant prompts for multi-document RAG,
enforcing grounding constraints and refusal of unsupported questions.
"""

from typing import List, Optional
from app.models import RetrievedChunk


SYSTEM_INSTRUCTIONS = """You are an accurate assistant analyzing user-uploaded documents.

Follow these operational guidelines:
- STRICT GROUNDING: Base your answer strictly on the facts, details, numbers, and code present in the provided context.
- MULTI-DOCUMENT ANSWERS: When asked to explain or summarize "each file", "all files", or "these documents", produce exactly ONE section per distinct DOCUMENT provided in the context (approx. 3 concise lines per document). Never list multiple context blocks from the same file as separate documents.
- IMAGE HANDLING: If an image file has no OCR text or visual description available, state clearly that visual understanding is unavailable and no text was extracted via OCR. Do not fabricate visual descriptions or recite technical metadata dimensions.
- CODE & SCRIPT FIDELITY: When the user asks for code or an implementation, provide the code statements from the context inside a markdown code block (```python ... ```).
- MISSING INFORMATION: If requested information is absent, state clearly that it is not available in the provided document.
- ACCURACY & TERMINOLOGY: Preserve exact numbers, dates, terms, and values from the source."""


def format_context(chunks: List[RetrievedChunk]) -> str:
    """Format retrieved chunks into structured evidence grouped by distinct document, deduplicating repeated lines."""
    if not chunks:
        return "No relevant context found."

    # Group chunks by filename/document
    docs_map = {}
    for chunk in chunks:
        fn = str(chunk.metadata.get("filename", chunk.metadata.get("source_file", "document")))
        if fn not in docs_map:
            docs_map[fn] = []
        docs_map[fn].append(chunk)

    formatted_docs = []
    for doc_idx, (fn, doc_chunks) in enumerate(docs_map.items(), 1):
        fmt = doc_chunks[0].metadata.get("format", "unknown")
        lines = [f"DOCUMENT {doc_idx}: {fn} (File Type: {fmt.upper()})", "Evidence:"]
        seen_lines = set()
        for c in doc_chunks:
            page = c.metadata.get("page_number", 1)
            sec = c.metadata.get("section", "General")
            c_lines = []
            for raw_line in c.text.splitlines():
                stripped = raw_line.strip()
                if not stripped:
                    continue
                norm = stripped.lower()
                if norm not in seen_lines:
                    seen_lines.add(norm)
                    c_lines.append(stripped)
            if c_lines:
                lines.append(f"--- [Page/Slide/Sheet {page} | Section: {sec}] ---\n" + "\n".join(c_lines))
        formatted_docs.append("\n".join(lines))

    return "\n\n" + ("=" * 40) + "\n\n".join(formatted_docs) + "\n" + ("=" * 40)


def build_rag_prompt(question: str, chunks: List[RetrievedChunk], lines_per_doc: Optional[int] = None) -> str:
    """Construct the final prompt for the LLM."""
    context_str = format_context(chunks)

    length_rule = ""
    if lines_per_doc and lines_per_doc > 0:
        length_rule = f"\n- EXACT LENGTH RULE: Provide EXACTLY {lines_per_doc} concise bullet lines for EACH distinct DOCUMENT (no more, no less). Do not exceed {lines_per_doc} lines per document.\n"

    prompt = f"""{SYSTEM_INSTRUCTIONS}{length_rule}

--- PROVIDED CONTEXT START ---
{context_str}
--- PROVIDED CONTEXT END ---

Question: {question.strip()}

Answer:"""
    return prompt
