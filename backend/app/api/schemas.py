"""
Pydantic Schemas for FastAPI API Layer.
Provides request and response validation for chat, documents, and ingestion endpoints.
"""

from typing import Any, Dict, List, Optional
from pydantic import BaseModel, Field


class HealthResponse(BaseModel):
    """Health check response."""
    status: str = Field(default="ok", description="Service health status")


class SourceChunk(BaseModel):
    """Retrieved source context chunk for response provenance."""
    filename: str = Field(..., description="Source PDF filename")
    page: Optional[int] = Field(None, description="Page number of the chunk")
    section: Optional[str] = Field(None, description="Section or chapter header")
    chunk_id: Optional[str] = Field(None, description="Unique chunk identifier")
    score: Optional[float] = Field(None, description="Retrieval similarity/relevance score")
    text: Optional[str] = Field(None, description="Excerpt content of the chunk")


class ChatRequest(BaseModel):
    """Chat question request payload."""
    query: str = Field(..., min_length=1, description="Question to answer using the RAG pipeline")
    show_context: bool = Field(default=False, description="Whether to include detailed chunk excerpts")


class ChatResponse(BaseModel):
    """Grounded RAG answer and source provenance."""
    answer: str = Field(..., description="Grounded answer synthesized from retrieved documents")
    sources: List[SourceChunk] = Field(default_factory=list, description="Source chunks used to generate the answer")


class DocumentInfo(BaseModel):
    """Summary metadata for an indexed document."""
    filename: str = Field(..., description="Filename of the document")
    file_hash: Optional[str] = Field(None, description="SHA-256 content hash")
    chunk_count: int = Field(default=0, description="Number of indexed chunks")
    page_count: Optional[int] = Field(None, description="Total pages in document if available")
    sections: List[str] = Field(default_factory=list, description="Extracted section titles")


class DocumentsResponse(BaseModel):
    """List of indexed documents and overall statistics."""
    total_documents: int = Field(..., description="Total unique documents indexed")
    total_chunks: int = Field(..., description="Total chunks in vector database")
    documents: List[DocumentInfo] = Field(default_factory=list, description="Metadata for each indexed document")


class IngestResponse(BaseModel):
    """Document ingestion response."""
    status: str = Field(default="success", description="Status of ingestion operation")
    documents_processed: int = Field(..., description="Number of documents processed")
    chunks_stored: int = Field(..., description="Number of chunks added or updated")
    message: str = Field(..., description="Human-readable result summary")


class ReindexResponse(BaseModel):
    """Document reindexing response."""
    status: str = Field(default="success", description="Status of reindexing operation")
    documents_processed: int = Field(..., description="Number of documents reindexed")
    chunks_stored: int = Field(..., description="Total chunks indexed in fresh collection")
    message: str = Field(..., description="Human-readable result summary")
