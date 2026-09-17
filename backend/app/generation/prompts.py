"""
Prompt Engineering and Guardrails Module.
Constructs strict, hallucination-resistant prompts for multi-document RAG,
enforcing grounding constraints and refusal of unsupported questions.
"""

from typing import List
from app.models import RetrievedChunk


SYSTEM_INSTRUCTIONS = """You are an accurate, benign multi-document RAG assistant analyzing user-uploaded documents (such as project reports, technical documentation, and literary works).

Follow these operational guidelines:
1. STRICT GROUNDING: Base your answer strictly on the provided context. If information is not in the context, state that it is not available.
2. COMPLETENESS & DETAIL: When answering what was used, implemented, built, or described, carefully read all bullet points and sentences. If multiple items, tools, libraries, or methods are mentioned (especially paired or conjoined items like 'A and B'), list ALL of them. Do not omit any item mentioned in the text.
3. MULTI-DOCUMENT COMPARISONS: When comparing entities across documents, use the provided context from each document to present a clear, grounded comparison highlighting similarities and differences.
4. PARTIAL CONTEXT: If some parts of a question are supported by the context but other parts are missing, answer the supported parts completely and explicitly note which specific parts are not available.
5. ACCURACY & TERMINOLOGY: Preserve exact names, numbers, dates, and terminology as written in the source documents.
6. CLARITY: Present your answer clearly and concisely without mentioning internal retrieval mechanics or chunk IDs."""


def format_context(chunks: List[RetrievedChunk]) -> str:
    """Format retrieved chunks into a clear, labeled context block with document, page, and section."""
    if not chunks:
        return "No relevant context found."

    formatted_blocks = []
    for i, chunk in enumerate(chunks, 1):
        section_label = chunk.section
        source_label = chunk.metadata.get("filename", chunk.metadata.get("source_file", "document"))
        page_label = chunk.metadata.get("page_number", chunk.metadata.get("page", 1))
        formatted_blocks.append(
            f"--- Context Block {i} [Document: {source_label} | Page: {page_label} | Section: {section_label}] ---\n"
            f"{chunk.text.strip()}"
        )

    return "\n\n".join(formatted_blocks)


def build_rag_prompt(question: str, chunks: List[RetrievedChunk]) -> str:
    """Construct the final prompt for the LLM."""
    context_str = format_context(chunks)

    prompt = f"""{SYSTEM_INSTRUCTIONS}

--- PROVIDED CONTEXT START ---
{context_str}
--- PROVIDED CONTEXT END ---

Question: {question.strip()}

Answer:"""
    return prompt
