"""
Pydantic Schemas for FastAPI API Layer.
Provides request and response validation for chat, documents, and ingestion endpoints.
"""

from typing import Any, Dict, List, Optional
from pydantic import BaseModel, Field, field_validator


class HealthResponse(BaseModel):
    """Health check response."""
    status: str = Field(default="ok", description="Service health status")


class ChatSessionResponse(BaseModel):
    """Metadata representation for a chat session."""
    id: str = Field(..., description="Unique chat identifier")
    title: str = Field(..., description="Chat session title")
    created_at: str = Field(..., description="Creation timestamp in UTC ISO-8601")
    updated_at: str = Field(..., description="Last updated timestamp in UTC ISO-8601")


class ChatCreateRequest(BaseModel):
    """Request payload to create a new chat."""
    title: Optional[str] = Field(default=None, max_length=200, description="Optional chat title")

    @field_validator("title")
    @classmethod
    def sanitize_title(cls, v: Optional[str]) -> Optional[str]:
        if v is not None:
            v = v.strip()
            if not v:
                return None
        return v


class ChatRenameRequest(BaseModel):
    """Request payload to rename an existing chat."""
    title: str = Field(..., min_length=1, max_length=200, description="New chat title")

    @field_validator("title")
    @classmethod
    def validate_title(cls, v: str) -> str:
        v = v.strip()
        if not v:
            raise ValueError("Chat title cannot be blank or whitespace-only.")
        return v


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
    """Summary metadata for an indexed document in the file library."""
    id: Optional[str] = Field(None, description="Unique document ID")
    filename: str = Field(..., description="Filename of the document")
    file_type: Optional[str] = Field(None, description="Derived file type or extension")
    file_size: Optional[int] = Field(None, description="File size in bytes")
    file_hash: Optional[str] = Field(None, description="SHA-256 content hash")
    chunk_count: int = Field(default=0, description="Number of indexed chunks")
    page_count: Optional[int] = Field(None, description="Total pages in document if available")
    sections: List[str] = Field(default_factory=list, description="Extracted section titles")
    status: Optional[str] = Field(default="ready", description="Document status")
    created_at: Optional[str] = Field(None, description="Upload timestamp in UTC ISO-8601")
    storage_path: Optional[str] = Field(None, description="Safe storage path")


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


class DocumentRecordResponse(BaseModel):
    """Metadata record for a registered or attached document in SQLite."""
    id: str = Field(..., description="Unique document ID")
    filename: str = Field(..., description="Original PDF filename")
    file_hash: str = Field(..., description="SHA-256 content hash")
    file_size: int = Field(..., description="Size in bytes")
    page_count: Optional[int] = Field(None, description="Number of pages")
    chunk_count: Optional[int] = Field(None, description="Number of chunks")
    storage_path: Optional[str] = Field(None, description="Relative or safe storage path")
    status: str = Field(..., description="Processing status: pending, processing, ready, failed")
    error_message: Optional[str] = Field(None, description="Safe error message on failure")
    created_at: str = Field(..., description="Creation timestamp in UTC ISO-8601")
    attached_at: Optional[str] = Field(None, description="Timestamp when attached to chat")


class DocumentUploadResponse(BaseModel):
    """Response payload after uploading a document to a chat."""
    message: str = Field(..., description="Status summary message")
    document: DocumentRecordResponse = Field(..., description="Attached document details")


class ChatMessageResponse(BaseModel):
    """Persisted chat message with sources."""
    id: str = Field(..., description="Unique message ID")
    chat_id: str = Field(..., description="Parent chat ID")
    role: str = Field(..., description="Message role: user or assistant")
    content: str = Field(..., description="Message text content")
    sources: List[SourceChunk] = Field(default_factory=list, description="Source provenance chunks for assistant responses")
    created_at: str = Field(..., description="Message creation timestamp in UTC ISO-8601")


class ConversionRequest(BaseModel):
    """Payload to request format conversion of a stored document."""
    target_format: str = Field(..., description="Desired target extension (e.g. xlsx, csv, docx, pdf)")


class ConversionTargetsResponse(BaseModel):
    """Supported conversion targets for a document."""
    source_format: str
    supported_targets: List[str]


