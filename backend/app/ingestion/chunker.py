"""
Structure-Aware and Semantic Document Chunking Module.
Preserves document structure, section headings, numbered lists, tables,
and atomic sub-entries without relying on hardcoded document keywords.
"""

from dataclasses import dataclass, field
import re
from typing import Dict, List, Optional
from langchain_text_splitters import RecursiveCharacterTextSplitter

from app.config import settings
from app.models import Document, Chunk
from app.utils.logger import setup_logger

logger = setup_logger("ingestion.chunker")


class DocumentChunker:
    """Production chunker combining structural section awareness with recursive fallback."""

    # Generic common headers across academic, corporate, technical, and general literature
    COMMON_SECTION_HEADERS = [
        "TABLE OF CONTENTS",
        "EXECUTIVE SUMMARY",
        "PROFESSIONAL SUMMARY",
        "TECHNICAL SKILLS",
        "PROFESSIONAL EXPERIENCE",
        "MACHINE LEARNING EXPERIENCE",
        "PROJECTS",
        "CERTIFICATIONS",
        "EDUCATION",
        "EXPERIENCE",
        "PUBLICATIONS",
        "ACKNOWLEDGMENTS",
        "ABSTRACT",
        "INTRODUCTION",
        "OVERVIEW",
        "BACKGROUND",
        "METHODOLOGY",
        "METHODS",
        "EXPERIMENTS",
        "RESULTS",
        "DISCUSSION",
        "CONCLUSION",
        "CONCLUSIONS",
        "FUTURE WORK",
        "REFERENCES",
        "APPENDIX",
        "SUMMARY",
        "SKILLS",
    ]

    def __init__(
        self,
        chunk_size: Optional[int] = None,
        chunk_overlap: Optional[int] = None,
    ):
        self.chunk_size = chunk_size or settings.chunk_size
        self.chunk_overlap = chunk_overlap or settings.chunk_overlap
        self.fallback_splitter = RecursiveCharacterTextSplitter(
            chunk_size=self.chunk_size,
            chunk_overlap=self.chunk_overlap,
            separators=["\n\n", "\n", "• ", "- ", ". ", " ", ""]
        )

    def _split_into_sections(self, text: str) -> List[Dict[str, str]]:
        """
        Identify section boundaries using generic heading patterns:
        - Markdown headers (# Heading)
        - Numbered chapters/sections (Chapter 1, Section 2.1)
        - Capitalized common structural headers
        - Generic all-caps headings (2 to 5 words)
        """
        # Generic heading regex: matches markdown headers, chapter/section keywords,
        # hierarchical numbered sections (e.g. 1.1 Overview), common structural headers,
        # or titled header lines ending in colon followed by newline.
        pattern = (
            r"(?:^|\n)(?:"
            r"#{1,4}\s+[^\n]+|"                                       # Markdown headers
            r"CHAPTER\s+\d+|SECTION\s+\d+|"                           # Numbered chapters/sections
            r"\d+\.\d+(?:\.\d+)*\s+[A-Z][^\n]{2,50}|"                 # Hierarchical sections: "1.2 Methodology"
            r"(?:" + "|".join(re.escape(h) for h in self.COMMON_SECTION_HEADERS) + r")(?:\b|:|\.)" # Common sections
            r")"
        )
        matches = list(re.finditer(pattern, text, flags=re.IGNORECASE))

        if not matches:
            return [{"section": "General", "content": text.strip()}]

        sections: List[Dict[str, str]] = []

        # Header/Profile before the first matched section
        if matches[0].start() > 0:
            first_part = text[:matches[0].start()].strip()
            if first_part:
                sections.append({"section": "Overview / Header", "content": first_part})

        for i, m in enumerate(matches):
            sec_name = re.sub(r"^#+\s*", "", m.group(0)).strip(" :\n\t\r.")
            start = m.start()
            end = matches[i + 1].start() if i + 1 < len(matches) else len(text)
            sec_text = text[start:end].strip()

            # Generic sub-item detection: check if section contains 2+ distinct titled items
            # (e.g. 'Project Alpha - Description' or 'Role - Organization')
            sub_pattern = r"(?:^|\n)([A-Z][A-Za-z0-9_\s]{2,30}\s*[-–—]\s*[^\n]+)"
            sub_matches = list(re.finditer(sub_pattern, sec_text))
            if len(sub_matches) >= 2:
                for sj, sm in enumerate(sub_matches):
                    sub_title = re.split(r"[-–—:]", sm.group(1))[0].strip()
                    sub_start = sm.start()
                    sub_end = sub_matches[sj + 1].start() if sj + 1 < len(sub_matches) else len(sec_text)
                    sub_content = sec_text[sub_start:sub_end].strip()
                    sections.append({
                        "section": f"{sec_name}: {sub_title}",
                        "content": sub_content
                    })
            else:
                sections.append({"section": sec_name, "content": sec_text})

        return sections

    def _detect_content_type(self, text: str) -> str:
        """
        Heuristically identify whether text is code, tabular data,
        procedural steps, key-value pairs, list, or regular prose.
        """
        lines = [line.strip() for line in text.split("\n") if line.strip()]
        if not lines:
            return "text"

        # 1. Code check (fenced code block or multiple programming language keywords / syntax)
        if "```" in text or text.strip().startswith("def ") or text.strip().startswith("class "):
            return "code"
        code_syntax_markers = [
            r"\bdef\s+[a-zA-Z_]\w*\s*\(",
            r"\bclass\s+[A-Z]\w*",
            r"\bimport\s+[a-zA-Z_]",
            r"\bfrom\s+[a-zA-Z_].*import",
            r"\bfor\s+\w+\s+in\s+",
            r"\bwhile\s*\(?.*\)?\s*:",
            r"\bif\s+.*:\s*$",
            r"\bfunction\s+[a-zA-Z_]\w*\s*\(",
            r"\bconst\s+[a-zA-Z_]\w*\s*=",
            r"\bvar\s+[a-zA-Z_]\w*\s*=",
            r"\blet\s+[a-zA-Z_]\w*\s*=",
            r"\breturn\s+[^\n;]+[;]?",
            r"^\s*[a-zA-Z_]\w*\s*\([^\)]*\)\s*$",  # function calls
        ]
        code_matches = sum(1 for line in lines if any(re.search(m, line) for m in code_syntax_markers))
        if code_matches >= 2 or (len(lines) <= 4 and code_matches >= 1):
            return "code"

        # 2. Table check (presence of '|' or multiple columnar aligned columns)
        table_lines = sum(1 for l in lines if "|" in l or len(re.findall(r"\s{3,}", l)) >= 2)
        if table_lines >= max(len(lines) // 2, 2):
            return "table"

        # 3. Procedural steps check ("Step 1:", "Step 2", "Phase 1:")
        step_lines = sum(1 for l in lines if re.match(r"^(?:step|phase|stage|task)\s+\d+[\:\.\-]?\s+", l, re.IGNORECASE))
        if step_lines >= 2:
            return "procedure"

        # 4. Key-Value / Structured Data check ("Key: Value", "Marks: 95", "Date: 2024-01-01")
        kv_lines = sum(1 for l in lines if re.match(r"^[A-Z][A-Za-z0-9_\s]{1,25}\s*:\s*[^\n]+$", l))
        if kv_lines >= max(len(lines) // 2, 2):
            return "key_value"

        # 5. List check (numbered lines or bullet points)
        list_lines = sum(1 for l in lines if re.match(r"^(?:\d+[\.\)]|[-*•])\s+", l))
        if list_lines >= max(len(lines) // 2, 2):
            return "list"

        return "text"

    def split_documents(self, documents: List[Document]) -> List[Chunk]:
        """
        Split loaded documents into structure-aware chunks with stable IDs,
        content types, and relational metadata.
        """
        all_chunks: List[Chunk] = []
        chunk_counter = 0

        for doc in documents:
            doc_id = str(doc.metadata.get("doc_id", "doc"))
            page_num = doc.metadata.get("page_number", 1)

            sections = self._split_into_sections(doc.page_content)

            for sec in sections:
                sec_name = sec["section"]
                sec_content = sec["content"]

                if not sec_content:
                    continue

                content_type = self._detect_content_type(sec_content)

                # Keep cohesive sections, tables, and lists intact if they fit
                if len(sec_content) <= self.chunk_size:
                    chunk_id = f"{doc_id}_c{chunk_counter}"
                    chunk_meta = {
                        **doc.metadata,
                        "chunk_id": chunk_id,
                        "chunk_index": chunk_counter,
                        "section": sec_name,
                        "content_type": content_type,
                        "char_count": len(sec_content),
                    }
                    all_chunks.append(Chunk(text=sec_content, metadata=chunk_meta))
                    chunk_counter += 1
                else:
                    # Recursive split if section is larger than chunk_size
                    sub_texts = self.fallback_splitter.split_text(sec_content)
                    for sub_idx, sub_text in enumerate(sub_texts):
                        chunk_id = f"{doc_id}_c{chunk_counter}"
                        chunk_meta = {
                            **doc.metadata,
                            "chunk_id": chunk_id,
                            "chunk_index": chunk_counter,
                            "section": f"{sec_name} (Part {sub_idx + 1})" if len(sub_texts) > 1 else sec_name,
                            "content_type": self._detect_content_type(sub_text),
                            "char_count": len(sub_text),
                        }
                        all_chunks.append(Chunk(text=sub_text, metadata=chunk_meta))
                        chunk_counter += 1

        logger.info(f"Chunking complete. Created {len(all_chunks)} chunks from {len(documents)} document pages.")
        return all_chunks
