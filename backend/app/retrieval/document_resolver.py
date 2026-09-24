"""
Document and Subject Resolution Module.
Resolves which documents and subjects are targeted by a query based on
chat-scoped attachments, explicit filename mentions, token similarity,
entity extraction, and conversational cues without hardcoded filenames.
"""

from dataclasses import dataclass, field
from pathlib import Path
import re
from typing import Any, Dict, List, Optional, Set, Tuple

from app.utils.logger import setup_logger

logger = setup_logger("retrieval.document_resolver")


@dataclass
class ResolvedDocument:
    """Represents a resolved document entity."""
    filename: str
    doc_id: Optional[str] = None
    file_hash: Optional[str] = None
    title: Optional[str] = None
    score: float = 1.0


@dataclass
class DocumentResolutionResult:
    """Outcome of resolving target documents for a query."""
    resolved_documents: List[ResolvedDocument] = field(default_factory=list)
    is_strictly_targeted: bool = False  # True if user explicitly named document or singular document context
    chroma_where_filter: Optional[Dict[str, Any]] = None
    warning: Optional[str] = None


class DocumentResolver:
    """
    General-purpose document resolver enforcing chat-scoped boundaries
    and detecting targeted documents from natural language queries.
    """

    STOPWORDS = {
        "a", "an", "the", "in", "of", "for", "to", "and", "or", "on", "at",
        "by", "with", "from", "about", "pdf", "document", "file", "this",
        "that", "these", "those", "according"
    }

    @classmethod
    def resolve(
        cls,
        query: str,
        available_documents: List[Dict[str, Any]],
        base_where_filter: Optional[Dict[str, Any]] = None,
        is_comparison: bool = False,
    ) -> DocumentResolutionResult:
        """
        Resolve relevant documents from the available (chat-scoped) documents.
        
        available_documents: list of dicts with keys: 'filename', 'id' (doc_id), 'file_hash'
        base_where_filter: existing chat-scoped filter
        """
        if not available_documents:
            return DocumentResolutionResult(
                resolved_documents=[],
                is_strictly_targeted=False,
                chroma_where_filter=base_where_filter,
            )

        # Restrict available_documents to match base_where_filter if an explicit filename or id was scoped
        if base_where_filter:
            if "filename" in base_where_filter and isinstance(base_where_filter["filename"], str):
                scoped_docs = [d for d in available_documents if d.get("filename") == base_where_filter["filename"]]
                if scoped_docs:
                    available_documents = scoped_docs
            elif "document_id" in base_where_filter and isinstance(base_where_filter["document_id"], str):
                scoped_docs = [d for d in available_documents if d.get("id") == base_where_filter["document_id"]]
                if scoped_docs:
                    available_documents = scoped_docs

        q_lower = query.lower()
        q_clean = re.sub(r"[^\w\s\.-]", " ", q_lower)
        q_tokens = [w for w in q_clean.split() if w not in cls.STOPWORDS]

        # 1. Check singular document context: "this document", "the pdf", "this file", "the uploaded resume", etc.
        has_singular_doc_ref = bool(
            re.search(
                r"\b(?:this|the|that|my)\s+(?:uploaded\s+|attached\s+)?(?:document|pdf|file|resume|cv|report|paper|guide|notes?|doc)\b"
                r"|\b(?:the\s+uploaded|the\s+attached)\b",
                q_lower,
            )
        )

        if len(available_documents) == 1 and (has_singular_doc_ref or not is_comparison):
            doc = available_documents[0]
            resolved = [ResolvedDocument(
                filename=doc.get("filename", ""),
                doc_id=doc.get("id"),
                file_hash=doc.get("file_hash"),
            )]
            where = cls._build_where_filter([doc], base_where_filter)
            return DocumentResolutionResult(
                resolved_documents=resolved,
                is_strictly_targeted=has_singular_doc_ref,
                chroma_where_filter=where,
            )

        # 2. Check "compare these two documents / resumes / files" when exactly 2 documents are in scope
        has_two_doc_ref = bool(
            re.search(r"\b(?:these\s+two|both\s+documents|both\s+files|both\s+resumes|the\s+two)\b", q_lower)
        )
        if len(available_documents) == 2 and (has_two_doc_ref or is_comparison):
            resolved = [
                ResolvedDocument(
                    filename=d.get("filename", ""),
                    doc_id=d.get("id"),
                    file_hash=d.get("file_hash"),
                )
                for d in available_documents
            ]
            return DocumentResolutionResult(
                resolved_documents=resolved,
                is_strictly_targeted=True,
                chroma_where_filter=base_where_filter,
            )

        # 3. Match explicit filename tokens or title tokens
        matched_docs: List[Tuple[Dict[str, Any], float]] = []

        for doc in available_documents:
            fn = doc.get("filename", "")
            base = re.sub(r"\.[a-zA-Z0-9]+$", "", fn).lower()
            base_clean = re.sub(r"[^a-z0-9]", " ", base)
            doc_tokens = set([w for w in base_clean.split() if w not in cls.STOPWORDS and len(w) >= 2])

            if not doc_tokens:
                continue

            # Exact full filename or base name in query (including concatenated titles e.g. oldmansea)
            q_condensed = re.sub(r"[^a-z0-9]", "", q_lower)
            q_token_joined = "".join(q_tokens)
            base_condensed = re.sub(r"[^a-z0-9]", "", base)

            if fn.lower() in q_lower or base in q_lower or (base_condensed and (base_condensed in q_condensed or base_condensed in q_token_joined)):
                matched_docs.append((doc, 100.0))
                continue

            # Check token overlap
            common = set(q_tokens) & doc_tokens
            if common:
                overlap_ratio = len(common) / len(doc_tokens)
                score = len(common) * 10.0 + overlap_ratio * 20.0
                if score >= 15.0 or overlap_ratio >= 0.5:
                    matched_docs.append((doc, score))

        if matched_docs:
            matched_docs.sort(key=lambda x: x[1], reverse=True)
            top_score = matched_docs[0][1]
            selected = [d for d, s in matched_docs if s >= top_score * 0.75]
            resolved = [
                ResolvedDocument(
                    filename=d.get("filename", ""),
                    doc_id=d.get("id"),
                    file_hash=d.get("file_hash"),
                    score=s,
                )
                for d, s in matched_docs if d in selected
            ]
            where = cls._build_where_filter(selected, base_where_filter)
            return DocumentResolutionResult(
                resolved_documents=resolved,
                is_strictly_targeted=True,
                chroma_where_filter=where,
            )

        # 4. Match format category keywords (spreadsheet, presentation, docx, image, pdf)
        FORMAT_CATEGORY_MAP = {
            "spreadsheet": {".xlsx", ".xls", ".csv"},
            "excel": {".xlsx", ".xls"},
            "presentation": {".pptx"},
            "slides": {".pptx"},
            "powerpoint": {".pptx"},
            "docx": {".docx"},
            "word": {".docx"},
            "image": {".jpg", ".jpeg", ".png", ".webp"},
            "photo": {".jpg", ".jpeg", ".png", ".webp"},
            "picture": {".jpg", ".jpeg", ".png", ".webp"},
            "pdf": {".pdf"},
        }
        for category_kw, valid_exts in FORMAT_CATEGORY_MAP.items():
            if re.search(r"\b" + re.escape(category_kw) + r"\b", q_lower):
                fmt_matches = [
                    d for d in available_documents
                    if Path(d.get("filename", "")).suffix.lower() in valid_exts
                ]
                if fmt_matches:
                    resolved = [
                        ResolvedDocument(
                            filename=d.get("filename", ""),
                            doc_id=d.get("id"),
                            file_hash=d.get("file_hash"),
                            score=90.0,
                        )
                        for d in fmt_matches
                    ]
                    where = cls._build_where_filter(fmt_matches, base_where_filter)
                    return DocumentResolutionResult(
                        resolved_documents=resolved,
                        is_strictly_targeted=True,
                        chroma_where_filter=where,
                    )

        # 4. Fallback: all available documents are candidates
        all_resolved = [
            ResolvedDocument(
                filename=d.get("filename", ""),
                doc_id=d.get("id"),
                file_hash=d.get("file_hash"),
            )
            for d in available_documents
        ]
        return DocumentResolutionResult(
            resolved_documents=all_resolved,
            is_strictly_targeted=False,
            chroma_where_filter=base_where_filter,
        )

    @classmethod
    def _build_where_filter(
        cls,
        docs: List[Dict[str, Any]],
        base_filter: Optional[Dict[str, Any]] = None,
    ) -> Optional[Dict[str, Any]]:
        """Construct ChromaDB where filter isolating to selected documents."""
        if not docs:
            return base_filter

        doc_filters = []
        for d in docs:
            if d.get("filename"):
                doc_filters.append({"filename": d["filename"]})
            elif d.get("file_hash"):
                doc_filters.append({"file_hash": d["file_hash"]})
            elif d.get("id"):
                doc_filters.append({"document_id": d["id"]})

        if not doc_filters:
            return base_filter

        if len(doc_filters) == 1:
            target_filter = doc_filters[0]
        else:
            target_filter = {"$or": doc_filters}

        if base_filter:
            if "$and" in base_filter and isinstance(base_filter["$and"], list):
                return {"$and": [*base_filter["$and"], target_filter]}
            return {"$and": [base_filter, target_filter]}
        return target_filter
