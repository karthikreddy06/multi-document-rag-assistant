"""
FastAPI HTTP API Routes for Multi-Document RAG Assistant.
Reuses existing RAGApplication orchestrator, retriever, vector store, and generation services.
"""

import hashlib
import json
from pathlib import Path
import re
from typing import Any, Dict, List, Optional
from fastapi import FastAPI, HTTPException, Depends, status, Response, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse

from app.config import settings
from app.ingestion.loader import FileTypeDetector
from app.db import repository, init_db
from app.main import RAGApplication
from app.utils.logger import setup_logger
from app.api.schemas import (
    HealthResponse,
    DocumentsResponse,
    DocumentInfo,
    ChatRequest,
    ChatResponse,
    SourceChunk,
    IngestResponse,
    ReindexResponse,
    ChatSessionResponse,
    ChatCreateRequest,
    ChatRenameRequest,
    DocumentRecordResponse,
    DocumentUploadResponse,
    ChatMessageResponse,
)

logger = setup_logger("api.routes")

# Initialize database schema idempotently
init_db()

# Initialize FastAPI application
app = FastAPI(
    title="Multi-Document RAG Assistant API",
    description="REST API layer for multi-document retrieval-augmented generation and ingestion.",
    version="1.0.0",
)

# Configure CORS for local development and frontend integrations
app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.cors_origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Singleton container for RAGApplication
_rag_app_instance: Optional[RAGApplication] = None


def get_rag_app() -> RAGApplication:
    """
    Dependency provider for RAGApplication.
    Instantiates lazily on first request to prevent blocking module import.
    """
    global _rag_app_instance
    if _rag_app_instance is None:
        logger.info("Initializing shared RAGApplication instance for API layer...")
        _rag_app_instance = RAGApplication(auto_ingest=False)
    return _rag_app_instance


# ------------------------------------------------------------------------------
# 1. Health Endpoint
# ------------------------------------------------------------------------------
@app.get("/api/health", response_model=HealthResponse, tags=["System"])
def health_check() -> HealthResponse:
    """Return API health status."""
    return HealthResponse(status="ok")


# ------------------------------------------------------------------------------
# 2. Documents Metadata Endpoint
# ------------------------------------------------------------------------------
@app.get("/api/documents", response_model=DocumentsResponse, tags=["Documents"])
def get_documents(rag_app: RAGApplication = Depends(get_rag_app)) -> DocumentsResponse:
    """
    Return currently indexed documents and existing metadata from ChromaDB.
    """
    try:
        indexed_map = rag_app.vector_store.get_indexed_files()
        total_chunks = rag_app.vector_store.count()

        if total_chunks == 0:
            return DocumentsResponse(
                total_documents=0,
                total_chunks=0,
                documents=[],
            )

        # Retrieve all chunk metadata from the active collection
        raw = rag_app.vector_store.collection.get(include=["metadatas"])
        metadatas = raw.get("metadatas", []) or []

        # Aggregate per-document statistics
        doc_stats: Dict[str, Dict[str, Any]] = {}
        for meta in metadatas:
            if not meta:
                continue
            fn = meta.get("filename", "unknown")
            if fn not in doc_stats:
                doc_stats[fn] = {
                    "filename": fn,
                    "file_hash": meta.get("file_hash") or indexed_map.get(fn),
                    "chunk_count": 0,
                    "pages": set(),
                    "sections": set(),
                }
            doc_stats[fn]["chunk_count"] += 1
            if meta.get("page_number"):
                doc_stats[fn]["pages"].add(meta.get("page_number"))
            sec = meta.get("section")
            if sec and sec != "General":
                doc_stats[fn]["sections"].add(sec)

        doc_list: List[DocumentInfo] = []
        for fn, info in sorted(doc_stats.items(), key=lambda x: x[0]):
            doc_list.append(
                DocumentInfo(
                    filename=info["filename"],
                    file_hash=info["file_hash"],
                    chunk_count=info["chunk_count"],
                    page_count=len(info["pages"]) if info["pages"] else None,
                    sections=sorted(list(info["sections"]))[:10],
                )
            )

        return DocumentsResponse(
            total_documents=len(doc_list),
            total_chunks=total_chunks,
            documents=doc_list,
        )
    except Exception as e:
        logger.error(f"Error retrieving documents metadata: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to retrieve document metadata from vector store."
        )


# ------------------------------------------------------------------------------
# 3. Chat Endpoint
# ------------------------------------------------------------------------------
@app.post("/api/chat", response_model=ChatResponse, tags=["Chat"])
def chat(
    request: ChatRequest,
    rag_app: RAGApplication = Depends(get_rag_app),
) -> ChatResponse:
    """
    Execute grounded RAG query through existing retriever and LLM generator.
    """
    query = request.query.strip()
    if not query:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Query string must not be empty.",
        )

    try:
        # 1. Retrieve relevant chunks using Adaptive RAG
        chunks: List[RetrievedChunk] = []
        plan = None
        num_pred, num_ctx = None, None

        if hasattr(rag_app.retriever, "retrieve_adaptive"):
            try:
                res = rag_app.retriever.retrieve_adaptive(query=query)
                if isinstance(res, tuple) and len(res) == 3:
                    chunks, plan, coverage = res
                    budget = getattr(plan, "generation_budget", None)
                    num_pred = getattr(budget, "num_predict", None)
                    num_ctx = getattr(budget, "num_ctx", None)
                elif isinstance(res, list):
                    chunks = res
                else:
                    chunks = rag_app.retriever.retrieve(query)
            except Exception:
                chunks = rag_app.retriever.retrieve(query)
        else:
            chunks = rag_app.retriever.retrieve(query)

        # 2. Generate grounded answer using dynamic generation budget
        answer = rag_app.generator.generate_answer(
            question=query,
            chunks=chunks,
            plan=plan,
            num_predict=num_pred,
            num_ctx=num_ctx,
        )

        # 3. Format source chunk provenance
        sources: List[SourceChunk] = []
        for chunk in chunks:
            sources.append(
                SourceChunk(
                    filename=str(chunk.metadata.get("filename", "unknown")),
                    page=chunk.metadata.get("page_number"),
                    section=chunk.metadata.get("section"),
                    chunk_id=chunk.metadata.get("chunk_id"),
                    score=round(chunk.score, 4) if chunk.score is not None else None,
                    text=chunk.text[:300] if request.show_context else None,
                )
            )

        return ChatResponse(answer=answer, sources=sources)
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error processing chat query '{query}': {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to generate answer from documents.",
        )


# ------------------------------------------------------------------------------
# 4. Ingest Endpoint
# ------------------------------------------------------------------------------
@app.post("/api/ingest", response_model=IngestResponse, tags=["Ingestion"])
def trigger_ingestion(rag_app: RAGApplication = Depends(get_rag_app)) -> IngestResponse:
    """
    Trigger incremental document ingestion on the documents directory.
    """
    try:
        res = rag_app.pipeline.ingest_directory(force=False)
        docs_count = res.get("documents", 0)
        chunks_count = res.get("chunks", 0)
        return IngestResponse(
            status="success",
            documents_processed=docs_count,
            chunks_stored=chunks_count,
            message=f"Ingestion completed. Processed {docs_count} documents, stored {chunks_count} chunks.",
        )
    except Exception as e:
        logger.error(f"Ingestion failed: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Document ingestion failed.",
        )


# ------------------------------------------------------------------------------
# 5. Reindex Endpoint
# ------------------------------------------------------------------------------
@app.post("/api/reindex", response_model=ReindexResponse, tags=["Ingestion"])
def trigger_reindex(rag_app: RAGApplication = Depends(get_rag_app)) -> ReindexResponse:
    """
    Clear vector store and re-index all documents from scratch.
    """
    try:
        rag_app.vector_store.clear()
        res = rag_app.pipeline.ingest_directory(force=True)
        docs_count = res.get("documents", 0)
        chunks_count = res.get("chunks", 0)
        return ReindexResponse(
            status="success",
            documents_processed=docs_count,
            chunks_stored=chunks_count,
            message=f"Reindexing completed. Cleared database and indexed {chunks_count} chunks from {docs_count} documents.",
        )
    except Exception as e:
        logger.error(f"Reindexing failed: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Document reindexing failed.",
        )


# ------------------------------------------------------------------------------
# 6. Chat Session CRUD Endpoints
# ------------------------------------------------------------------------------
@app.get("/api/chats", response_model=List[ChatSessionResponse], tags=["Chats"])
def list_chats() -> List[ChatSessionResponse]:
    """
    Retrieve all chat sessions ordered by updated_at DESC.
    """
    try:
        chats = repository.list_chats()
        return [ChatSessionResponse(**c) for c in chats]
    except Exception as e:
        logger.error(f"Error listing chats: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to retrieve chats.",
        )


@app.post(
    "/api/chats",
    response_model=ChatSessionResponse,
    status_code=status.HTTP_201_CREATED,
    tags=["Chats"],
)
def create_chat(request: Optional[ChatCreateRequest] = None) -> ChatSessionResponse:
    """
    Create a new chat session. Defaults title to 'New Chat' if omitted or empty.
    """
    try:
        title = (request.title if (request and request.title) else "New Chat")
        chat_data = repository.create_chat(title=title)
        return ChatSessionResponse(**chat_data)
    except Exception as e:
        logger.error(f"Error creating chat: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to create chat.",
        )


@app.get("/api/chats/{chat_id}", response_model=ChatSessionResponse, tags=["Chats"])
def get_chat(chat_id: str) -> ChatSessionResponse:
    """
    Retrieve a single chat session by ID.
    """
    chat_data = repository.get_chat(chat_id)
    if not chat_data:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Chat '{chat_id}' not found.",
        )
    return ChatSessionResponse(**chat_data)


@app.patch("/api/chats/{chat_id}", response_model=ChatSessionResponse, tags=["Chats"])
def rename_chat(chat_id: str, request: ChatRenameRequest) -> ChatSessionResponse:
    """
    Rename an existing chat session.
    """
    try:
        updated = repository.rename_chat(chat_id, request.title)
        if not updated:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"Chat '{chat_id}' not found.",
            )
        return ChatSessionResponse(**updated)
    except HTTPException:
        raise
    except ValueError as ve:
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail=str(ve),
        )
    except Exception as e:
        logger.error(f"Error renaming chat '{chat_id}': {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to rename chat.",
        )


@app.delete("/api/chats/{chat_id}", status_code=status.HTTP_204_NO_CONTENT, tags=["Chats"])
def delete_chat(chat_id: str) -> Response:
    """
    Delete a chat session, its messages, and chat-document relationships.
    """
    try:
        deleted = repository.delete_chat(chat_id)
        if not deleted:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"Chat '{chat_id}' not found.",
            )
        return Response(status_code=status.HTTP_204_NO_CONTENT)
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error deleting chat '{chat_id}': {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to delete chat.",
        )


# ------------------------------------------------------------------------------
# 7. Chat Document Management Endpoints (Phases 4.4, 4.5, 4.9)
# ------------------------------------------------------------------------------
MAX_UPLOAD_SIZE = 50 * 1024 * 1024  # 50 MB limit


@app.post(
    "/api/chats/{chat_id}/documents",
    response_model=DocumentUploadResponse,
    status_code=status.HTTP_201_CREATED,
    tags=["Chat Documents"],
)
def upload_document_to_chat(
    chat_id: str,
    file: UploadFile = File(...),
    rag_app: RAGApplication = Depends(get_rag_app),
) -> DocumentUploadResponse:
    """
    Upload and attach a document to a chat session.
    Supports PDF, DOCX, PPTX, XLSX, CSV, TXT, MD, and image files.
    Validates extension against supported types, enforces 50MB size limit,
    sanitizes filename, computes SHA-256 hash, stores in backend/data/uploads/,
    and indexes into ChromaDB via the ParserRegistry.
    Reuses existing document records if SHA-256 already exists and is ready.
    """
    # 1. Verify chat exists
    chat = repository.get_chat(chat_id)
    if not chat:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Chat '{chat_id}' not found.",
        )

    # 2. Filename validation and sanitization against path traversal
    raw_filename = file.filename or ""
    # Extract only the base name (prevents directory traversal e.g. ../../)
    safe_name = Path(raw_filename).name
    # Strip dangerous characters and null bytes
    safe_name = re.sub(r"[\x00/\\:*?\"<>|]", "_", safe_name).strip()

    if not safe_name:
        safe_name = "document"

    # 2a. Validate file extension against all supported types
    file_ext = Path(safe_name).suffix.lower()
    if not file_ext or not FileTypeDetector.is_supported(file_ext):
        supported_exts = ", ".join(sorted(FileTypeDetector.supported_extensions()))
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=(
                f"Unsupported file type '{file_ext}'. "
                f"Permitted file types: {supported_exts}"
            ),
        )

    # 3. Read content and enforce size limits
    try:
        content = file.file.read()
    except Exception as e:
        logger.error(f"Failed to read uploaded file payload: {e}")
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Failed to read uploaded file content.",
        )

    file_size = len(content)
    if file_size == 0:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Empty file uploaded. File content must not be 0 bytes.",
        )

    if file_size > MAX_UPLOAD_SIZE:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="File size exceeds maximum permitted limit of 50 MB.",
        )

    # 4. Magic-byte validation (format-agnostic; returns True for formats with no magic bytes)
    detected_format = FileTypeDetector.detect_format(Path(safe_name))
    if not FileTypeDetector.validate_magic_bytes(content, detected_format):
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=(
                f"File content does not match the expected format for '{file_ext}'. "
                "Ensure the file is not corrupt or disguised with a wrong extension."
            ),
        )

    # 5. Compute SHA-256 hash
    file_hash = hashlib.sha256(content).hexdigest()

    # 6. Save file under backend/data/uploads/ using hash + original extension for correct parser dispatch
    upload_dir = settings.upload_abs_path
    upload_dir.mkdir(parents=True, exist_ok=True)
    stored_path = upload_dir / f"{file_hash}{file_ext}"
    if not stored_path.exists():
        with open(stored_path, "wb") as f:
            f.write(content)

    safe_rel_path = f"uploads/{file_hash}{file_ext}"

    # 7. Check if document record already exists in catalog
    existing_doc = repository.get_document_by_hash(file_hash)
    if existing_doc is not None:
        doc_id = existing_doc["id"]
        # Idempotently attach to chat
        repository.attach_document_to_chat(chat_id, doc_id)

        # If already ready, reuse without any reprocessing
        if existing_doc["status"] == "ready":
            attached_docs = repository.list_chat_documents(chat_id)
            attached_rec = next((d for d in attached_docs if d["id"] == doc_id), existing_doc)
            return DocumentUploadResponse(
                message="Existing document attached to chat without reprocessing.",
                document=DocumentRecordResponse(**attached_rec),
            )
    else:
        # Check if already present in vector store from pre-existing documents
        indexed_files = rag_app.vector_store.get_indexed_files()
        if file_hash in indexed_files.values():
            # Count existing chunks
            res = rag_app.vector_store.collection.get(where={"file_hash": file_hash})
            cnt = len(res.get("ids", []))
            new_doc = repository.create_document(
                filename=safe_name,
                file_hash=file_hash,
                file_size=file_size,
                chunk_count=cnt,
                storage_path=safe_rel_path,
                status="ready",
            )
            doc_id = new_doc["id"]
            repository.attach_document_to_chat(chat_id, doc_id)
            attached_docs = repository.list_chat_documents(chat_id)
            attached_rec = next((d for d in attached_docs if d["id"] == doc_id), new_doc)
            return DocumentUploadResponse(
                message="Document registered from existing vector store and attached to chat.",
                document=DocumentRecordResponse(**attached_rec),
            )
        else:
            # Register new pending document
            new_doc = repository.create_document(
                filename=safe_name,
                file_hash=file_hash,
                file_size=file_size,
                storage_path=safe_rel_path,
                status="pending",
            )
            doc_id = new_doc["id"]
            repository.attach_document_to_chat(chat_id, doc_id)

    # 8. Document Processing / Indexing
    repository.update_document_status(doc_id, status="processing")
    try:
        ingest_res = rag_app.pipeline.ingest_uploaded_document(
            file_path=stored_path,
            original_filename=safe_name,
            document_id=doc_id,
            file_hash=file_hash,
        )
        ready_doc = repository.update_document_status(
            doc_id,
            status="ready",
            page_count=ingest_res.get("page_count", 0),
            chunk_count=ingest_res.get("chunk_count", 0),
            storage_path=safe_rel_path,
        )
        attached_docs = repository.list_chat_documents(chat_id)
        attached_rec = next((d for d in attached_docs if d["id"] == doc_id), ready_doc)
        return DocumentUploadResponse(
            message="Document uploaded, parsed, and indexed successfully.",
            document=DocumentRecordResponse(**attached_rec),
        )
    except Exception as e:
        logger.error(f"Error processing uploaded document {doc_id}: {e}")
        failed_doc = repository.update_document_status(
            doc_id,
            status="failed",
            error_message=f"Failed to parse and index document: {type(e).__name__}",
        )
        attached_docs = repository.list_chat_documents(chat_id)
        attached_rec = next((d for d in attached_docs if d["id"] == doc_id), failed_doc)
        return DocumentUploadResponse(
            message="Document upload recorded, but processing failed.",
            document=DocumentRecordResponse(**attached_rec),
        )


@app.get(
    "/api/chats/{chat_id}/documents",
    response_model=List[DocumentRecordResponse],
    tags=["Chat Documents"],
)
def list_chat_documents(chat_id: str) -> List[DocumentRecordResponse]:
    """
    List all documents attached to a specific chat session.
    """
    chat = repository.get_chat(chat_id)
    if not chat:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Chat '{chat_id}' not found.",
        )
    docs = repository.list_chat_documents(chat_id)
    return [DocumentRecordResponse(**d) for d in docs]


@app.delete(
    "/api/chats/{chat_id}/documents/{document_id}",
    status_code=status.HTTP_204_NO_CONTENT,
    tags=["Chat Documents"],
)
def remove_document_from_chat(chat_id: str, document_id: str) -> Response:
    """
    Detach a document from a chat session.
    Does NOT delete the document from the global catalog or ChromaDB.
    """
    chat = repository.get_chat(chat_id)
    if not chat:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Chat '{chat_id}' not found.",
        )

    detached = repository.detach_document_from_chat(chat_id, document_id)
    if not detached:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Document '{document_id}' is not attached to chat '{chat_id}'.",
        )
    return Response(status_code=status.HTTP_204_NO_CONTENT)


# ------------------------------------------------------------------------------
# 8. Chat-Scoped Query and Message Persistence (Phases 4.6, 4.7)
# ------------------------------------------------------------------------------
@app.post(
    "/api/chats/{chat_id}/chat",
    response_model=ChatResponse,
    tags=["Chat"],
)
def chat_in_session(
    chat_id: str,
    request: ChatRequest,
    rag_app: RAGApplication = Depends(get_rag_app),
) -> ChatResponse:
    """
    Execute a chat question scoped strictly to documents attached to this chat session.
    Persists user message and assistant answer with sources into SQLite.
    """
    chat = repository.get_chat(chat_id)
    if not chat:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Chat '{chat_id}' not found.",
        )

    query = request.query.strip()
    if not query:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Query string must not be empty.",
        )

    # Retrieve ready documents attached to this chat
    attached_docs = repository.list_chat_documents(chat_id)
    ready_docs = [d for d in attached_docs if d["status"] == "ready"]

    # If no ready documents are attached, return clean response and persist
    if not ready_docs:
        notice = "No ready documents are attached to this chat. Please upload and attach a PDF document before asking questions."
        repository.create_message(chat_id=chat_id, role="user", content=query)
        repository.create_message(chat_id=chat_id, role="assistant", content=notice, sources_json=[])
        return ChatResponse(answer=notice, sources=[])

    # Construct strict Chroma filter restricting retrieval to attached ready documents
    if len(ready_docs) == 1:
        doc = ready_docs[0]
        if doc.get("file_hash"):
            where_filter = {
                "$or": [
                    {"document_id": doc["id"]},
                    {"file_hash": doc["file_hash"]},
                ]
            }
        else:
            where_filter = {"document_id": doc["id"]}
    else:
        doc_ids = [d["id"] for d in ready_docs]
        doc_hashes = [d["file_hash"] for d in ready_docs if d.get("file_hash")]
        where_filter = {
            "$or": [
                {"document_id": {"$in": doc_ids}},
                {"file_hash": {"$in": doc_hashes}},
            ]
        }

    recent_messages = repository.list_messages(chat_id)

    try:
        import time
        start_time = time.time()

        # Retrieve candidate chunks scoped to chat documents using Adaptive RAG
        chunks, plan, coverage = rag_app.retriever.retrieve_adaptive(
            query=query,
            where=where_filter,
            recent_messages=recent_messages,
            available_documents=ready_docs,
        )

        # Generate answer synthesized from retrieved chunks using dynamic generation budget
        budget = getattr(plan, "generation_budget", None)
        num_pred = getattr(budget, "num_predict", None)
        num_ctx = getattr(budget, "num_ctx", None)
        effective_q = plan.query if (plan and getattr(plan, "is_follow_up", False) and getattr(plan, "query", None)) else query
        answer = rag_app.generator.generate_answer(
            question=effective_q,
            chunks=chunks,
            plan=plan,
            num_predict=num_pred,
            num_ctx=num_ctx,
        )

        elapsed = time.time() - start_time

        # Build provenance sources
        sources: List[SourceChunk] = []
        for chunk in chunks:
            sources.append(
                SourceChunk(
                    filename=str(chunk.metadata.get("filename", "unknown")),
                    page=chunk.metadata.get("page_number"),
                    section=chunk.metadata.get("section"),
                    chunk_id=chunk.metadata.get("chunk_id"),
                    score=round(chunk.score, 4) if chunk.score is not None else None,
                    text=chunk.text[:300] if request.show_context else None,
                )
            )

        # Structured debug telemetry
        logger.info(
            f"\n--- RAG EXECUTION TELEMETRY ---\n"
            f"QUERY: {query}\n"
            f"QUERY TYPE: {plan.strategy.value}\n"
            f"TARGET DOCUMENTS: {[d.filename for d in plan.target_documents]}\n"
            f"RETRIEVAL PLAN: {plan.strategy.name} (k={plan.candidate_k})\n"
            f"ASPECTS: {plan.aspects}\n"
            f"CANDIDATE COUNT: {plan.candidate_k}\n"
            f"FINAL CHUNK COUNT: {len(chunks)}\n"
            f"PAGE COVERAGE: {coverage.pages_covered}\n"
            f"DOCUMENT COVERAGE: {coverage.documents_covered}\n"
            f"CONTEXT SIZE: {sum(len(c.text) for c in chunks)} chars\n"
            f"SOURCES: {len(sources)}\n"
            f"TOTAL TIME: {elapsed:.2f}s\n"
            f"-------------------------------"
        )

        # Persist messages in database
        repository.create_message(chat_id=chat_id, role="user", content=query)
        sources_payload = [s.model_dump() for s in sources]
        repository.create_message(
            chat_id=chat_id,
            role="assistant",
            content=answer,
            sources_json=sources_payload,
        )

        return ChatResponse(answer=answer, sources=sources)
    except Exception as e:
        logger.error(f"Error during chat in session '{chat_id}': {e}")
        # Clean safe error message
        safe_msg = "An error occurred while generating an answer from the attached documents."
        repository.create_message(chat_id=chat_id, role="user", content=query)
        repository.create_message(chat_id=chat_id, role="assistant", content=safe_msg, sources_json=[])
        return ChatResponse(answer=safe_msg, sources=[])


@app.post(
    "/api/chats/{chat_id}/stream",
    tags=["Chat"],
)
def chat_in_session_stream(
    chat_id: str,
    request: ChatRequest,
    rag_app: RAGApplication = Depends(get_rag_app),
) -> StreamingResponse:
    """
    Stream token chunks via Server-Sent Events (SSE) for low perceived latency.
    Persists user message and completed assistant response with sources upon completion.
    """
    chat = repository.get_chat(chat_id)
    if not chat:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Chat '{chat_id}' not found.",
        )

    query = request.query.strip()
    if not query:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Query string must not be empty.",
        )

    attached_docs = repository.list_chat_documents(chat_id)
    ready_docs = [d for d in attached_docs if d["status"] == "ready"]

    if not ready_docs:
        notice = "No ready documents are attached to this chat. Please upload and attach a PDF document before asking questions."
        repository.create_message(chat_id=chat_id, role="user", content=query)
        repository.create_message(chat_id=chat_id, role="assistant", content=notice, sources_json=[])

        def empty_generator():
            yield f"data: {json.dumps({'type': 'token', 'token': notice})}\n\n"
            yield f"data: {json.dumps({'type': 'sources', 'sources': []})}\n\n"
            yield "data: [DONE]\n\n"

        return StreamingResponse(empty_generator(), media_type="text/event-stream")

    if len(ready_docs) == 1:
        doc = ready_docs[0]
        if doc.get("file_hash"):
            where_filter = {
                "$or": [
                    {"document_id": doc["id"]},
                    {"file_hash": doc["file_hash"]},
                ]
            }
        else:
            where_filter = {"document_id": doc["id"]}
    else:
        doc_ids = [d["id"] for d in ready_docs]
        doc_hashes = [d["file_hash"] for d in ready_docs if d.get("file_hash")]
        where_filter = {
            "$or": [
                {"document_id": {"$in": doc_ids}},
                {"file_hash": {"$in": doc_hashes}},
            ]
        }

    recent_messages = repository.list_messages(chat_id)

    chunks, plan, coverage = rag_app.retriever.retrieve_adaptive(
        query=query,
        where=where_filter,
        recent_messages=recent_messages,
        available_documents=ready_docs,
    )

    sources: List[SourceChunk] = []
    for chunk in chunks:
        sources.append(
            SourceChunk(
                filename=str(chunk.metadata.get("filename", "unknown")),
                page=chunk.metadata.get("page_number"),
                section=chunk.metadata.get("section"),
                chunk_id=chunk.metadata.get("chunk_id"),
                score=round(chunk.score, 4) if chunk.score is not None else None,
                text=chunk.text[:300] if request.show_context else None,
            )
        )

    def event_stream():
        accumulated_tokens = []
        try:
            effective_q = plan.query if (plan and getattr(plan, "is_follow_up", False) and getattr(plan, "query", None)) else query
            for token in rag_app.generator.generate_answer_stream(
                question=effective_q,
                chunks=chunks,
                plan=plan,
                num_predict=plan.generation_budget.num_predict,
                num_ctx=plan.generation_budget.num_ctx,
            ):
                accumulated_tokens.append(token)
                yield f"data: {json.dumps({'type': 'token', 'token': token})}\n\n"

            full_answer = "".join(accumulated_tokens).strip()
            repository.create_message(chat_id=chat_id, role="user", content=query)
            sources_payload = [s.model_dump() for s in sources]
            repository.create_message(
                chat_id=chat_id,
                role="assistant",
                content=full_answer,
                sources_json=sources_payload,
            )
            yield f"data: {json.dumps({'type': 'sources', 'sources': sources_payload})}\n\n"
            yield "data: [DONE]\n\n"
        except Exception as err:
            logger.error(f"Error during streaming in chat '{chat_id}': {err}")
            yield f"data: {json.dumps({'type': 'error', 'error': 'Failed to complete streaming response.'})}\n\n"
            yield f"data: {json.dumps({'type': 'error', 'error': str(err)})}\n\n"

    return StreamingResponse(event_stream(), media_type="text/event-stream")


@app.get(
    "/api/chats/{chat_id}/messages",
    response_model=List[ChatMessageResponse],
    tags=["Chat Messages"],
)
def get_chat_messages(chat_id: str) -> List[ChatMessageResponse]:
    """
    Retrieve chronological message history for a specific chat session.
    """
    chat = repository.get_chat(chat_id)
    if not chat:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Chat '{chat_id}' not found.",
        )

    raw_messages = repository.list_messages(chat_id)
    result: List[ChatMessageResponse] = []
    for m in raw_messages:
        sources_list: List[SourceChunk] = []
        if m.get("sources_json"):
            try:
                parsed = json.loads(m["sources_json"])
                if isinstance(parsed, list):
                    sources_list = [SourceChunk(**item) for item in parsed]
            except Exception:
                pass

        result.append(
            ChatMessageResponse(
                id=m["id"],
                chat_id=m["chat_id"],
                role=m["role"],
                content=m["content"],
                sources=sources_list,
                created_at=m["created_at"],
            )
        )
    return result


