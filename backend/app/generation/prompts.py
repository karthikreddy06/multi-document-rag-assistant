"""
Prompt Engineering and Guardrails Module.
Constructs strict, hallucination-resistant prompts for multi-document RAG,
enforcing grounding constraints and refusal of unsupported questions.
"""

import re
from typing import List, Optional
from app.models import RetrievedChunk


SYSTEM_INSTRUCTIONS = """You are an accurate assistant analyzing user-uploaded documents.

Follow these operational guidelines:
- STRICT GROUNDING: Base your answer strictly on the facts, details, numbers, and code present in the provided context. Use ONLY information contained in the retrieved document context. Do not invent missing information, do not infer facts that are not explicitly supported, and do not add general external knowledge about any person, company, project, or topic. Never expose or assume information from documents outside the provided context.
- MISSING INFORMATION: If a requested category, topic, or fact is not present in the retrieved context, state clearly: "Not found in the provided document." Do not fabricate or hallucinate plausible details.
- ACCURACY & TERMINOLOGY: Preserve exact terminology, names, dates, numbers, and values from the source document. Do not silently correct or normalize discrepancies in the source document.
- STRUCTURED MARKDOWN FORMATTING:
  * Present information using clean, readable Markdown structure.
  * For broad questions or document summaries, organize the response into clear sections using Markdown headings (e.g. ## Section Name and ### Sub-item where appropriate).
  * Use bullet points for lists of attributes, items, components, contact fields, and duties.
  * Use bold labels for important fields and categories (e.g. - **Field**: Value).
  * Use short, focused paragraphs where descriptive text is required.
  * Never compress the response into a single continuous wall of text or one large paragraph.
  * Avoid unnecessary repetition of the same information across sections.
  * For complete-summary requests, systematically cover each distinct topic or category present in the document or requested by the user, keeping the response concise yet complete.
- MULTI-DOCUMENT ANSWERS: When asked to explain or summarize "each file", "all files", or "these documents", produce exactly ONE section per distinct DOCUMENT provided in the context (approx. 3 concise lines per document). Never list multiple context blocks from the same file as separate documents.
- IMAGE HANDLING: If an image file has no OCR text or visual description available, state clearly that visual understanding is unavailable and no text was extracted via OCR. Do not fabricate visual descriptions or recite technical metadata dimensions.
- CODE & SCRIPT FIDELITY: When the user asks for code or an implementation, provide the code statements from the context inside a markdown code block (```python ... ```)."""


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


def build_rag_prompt(
    question: str,
    chunks: List[RetrievedChunk],
    lines_per_doc: Optional[int] = None,
    is_summary: Optional[bool] = None,
) -> str:
    """Construct the final prompt for the LLM."""
    context_str = format_context(chunks)

    # Detect broad or document-summary requests to reinforce section formatting
    q_lower = question.lower()
    summary_query = (
        is_summary or
        bool(re.search(
            r"\b(?:summarize|summary|overview|complete\s+(?:summary|overview|details|profile|breakdown)|full\s+(?:summary|overview|details|profile|breakdown)|entire|all\s+sections|comprehensive)\b",
            q_lower
        ))
    )

    summary_directive = ""
    if summary_query and not (lines_per_doc and lines_per_doc > 0):
        summary_directive = (
            "\n- DOCUMENT SUMMARY FORMATTING DIRECTIVE: Organize the response into clean Markdown sections with "
            "descriptive headings (## Heading) matching the key categories in the document. "
            "Use bullet points with bold field names (- **Field**: Value) for structured details, "
            "attributes, items, and contact data. Use subheadings (### Item) for distinct projects, roles, or modules. "
            "Keep each entry readable and distinct. If a requested category is absent, say \"Not found in the provided document.\". "
            "Do NOT output a single compressed paragraph.\n"
        )

    length_rule = ""
    if lines_per_doc and lines_per_doc > 0:
        length_rule = f"\n- EXACT LENGTH RULE: Provide EXACTLY {lines_per_doc} concise bullet lines for EACH distinct DOCUMENT (no more, no less). Do not exceed {lines_per_doc} lines per document.\n"

    prompt = f"""{SYSTEM_INSTRUCTIONS}{summary_directive}{length_rule}

--- PROVIDED CONTEXT START ---
{context_str}
--- PROVIDED CONTEXT END ---

Question: {question.strip()}

Answer:"""
    return prompt
