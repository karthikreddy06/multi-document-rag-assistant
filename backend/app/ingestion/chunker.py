"""
Section-Aware and Semantic Chunking Module.
Splits documents intelligently, preserving section boundaries, preventing fragmentation
of atomic sections (e.g. Technical Skills, Projects), and retaining rich metadata.
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

    # Common section headers in structured documents, reports, manuals, and profiles
    SECTION_HEADERS = [
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
        Identify section boundaries and return list of dicts with 'section' and 'content'.
        Uses generic heading and itemized sub-entry detection without hardcoding document terms.
        """
        # Match standard section headers or numbered chapters/sections
        pattern = (
            r"(?:^|\n)(?:CHAPTER\s+\d+|SECTION\s+\d+|"
            + "|".join(re.escape(h) for h in self.SECTION_HEADERS)
            + r")(?:\b|:|\.)"
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
            sec_name = m.group(0).strip(" :\n\t\r.")
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

    def split_documents(self, documents: List[Document]) -> List[Chunk]:
        """
        Split a list of loaded documents into chunks with stable IDs and section metadata.
        """
        all_chunks: List[Chunk] = []
        chunk_counter = 0

        for doc in documents:
            doc_id = str(doc.metadata.get("doc_id", "doc"))
            page_num = doc.metadata.get("page_number", 1)
            source = str(doc.metadata.get("source", ""))
            filename = str(doc.metadata.get("filename", ""))

            sections = self._split_into_sections(doc.page_content)

            for sec in sections:
                sec_name = sec["section"]
                sec_content = sec["content"]

                if not sec_content:
                    continue

                # If the section comfortably fits in chunk_size, keep it intact
                if len(sec_content) <= self.chunk_size:
                    chunk_id = f"{doc_id}_c{chunk_counter}"
                    chunk_meta = {
                        **doc.metadata,
                        "chunk_id": chunk_id,
                        "chunk_index": chunk_counter,
                        "section": sec_name,
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
                            "char_count": len(sub_text),
                        }
                        all_chunks.append(Chunk(text=sub_text, metadata=chunk_meta))
                        chunk_counter += 1

        logger.info(f"Chunking complete. Created {len(all_chunks)} chunks from {len(documents)} document pages.")
        return all_chunks
