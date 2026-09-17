"""
Core Data Models for RAG Pipeline.
Defines Document, Chunk, and RetrievedChunk to avoid circular dependencies.
"""

from dataclasses import dataclass, field
from typing import Any, Dict


@dataclass
class Document:
    """Represents a loaded document page or entire document."""
    page_content: str
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class Chunk:
    """Represents a text chunk with associated metadata for vector storage."""
    text: str
    metadata: Dict[str, Any] = field(default_factory=dict)

    @property
    def chunk_id(self) -> str:
        return str(self.metadata.get("chunk_id", ""))

    @property
    def section(self) -> str:
        return str(self.metadata.get("section", "General"))


@dataclass
class RetrievedChunk:
    """Represents a retrieved context chunk with relevance score and metadata."""
    text: str
    metadata: Dict[str, Any] = field(default_factory=dict)
    score: float = 0.0

    @property
    def chunk_id(self) -> str:
        return str(self.metadata.get("chunk_id", ""))

    @property
    def section(self) -> str:
        return str(self.metadata.get("section", "General"))
