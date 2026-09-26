"""
FastAPI HTTP API Routes for Multi-Document RAG Assistant.
Reuses existing RAGApplication orchestrator, retriever, vector store, and generation services.
Enforces strict multi-user scoping and authentication on all user-facing resources.
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
from app.db import repository, init_db, close_db
from app.main import RAGApplication
from app.models import RetrievedChunk
from app.utils.logger import setup_logger
from app.services.storage import get_storage_service
from app.services.converter import (
    DocumentConverter,
    is_conversion_supported,
    get_supported_targets,
    UnsupportedConversionError,
    MIME_TYPES,
)
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
    ConversionRequest,
    ConversionTargetsResponse,
)
from app.api.auth import (
    router as auth_router,
    get_current_user,
    get_current_user_id,
    get_current_user_from_header_or_query,
)
from app.auth.security import create_access_token
from app.retrieval.query_understanding import parse_conversion_request

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

# Mount authentication router (Phase 2)
app.include_router(auth_router)


@app.on_event("shutdown")
def on_shutdown() -> None:
    """Safely release database pool connections on application shutdown."""
    close_db()

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
# 1. Health Endpoint (Public)
# ------------------------------------------------------------------------------
@app.get("/api/health", response_model=HealthResponse, tags=["System"])
def health_check() -> HealthResponse:
    """Return API health status."""
    return HealthResponse(status="ok")


# ------------------------------------------------------------------------------
# Helpers: Chat Auto-Titling & Document Storage Resolution
# ------------------------------------------------------------------------------
def generate_chat_title(query: str, max_words: int = 6) -> str:
    """Generate a clean, deterministic short title from the user's first prompt."""
    cleaned = re.sub(r"[^\w\s-]", "", query).strip()
    words = cleaned.split()
    if not words:
        return "New Chat"
    title = " ".join(words[:max_words]).strip()
    return title[0].upper() + title[1:] if title else "New Chat"


def _fetch_document_bytes(doc: Dict[str, Any], user_id: str) -> bytes:
    """Safely retrieve original document content from persistent or local storage."""
    storage = get_storage_service()
    storage_path = doc.get("storage_path")
    if storage_path:
        try:
            return storage.download(storage_path)
        except Exception as e:
            logger.warning(f"Could not download from storage_path '{storage_path}': {e}")

    # Fallback to local uploads directory
    fn = doc.get("filename", "")
    fhash = doc.get("file_hash", "")
    ext = Path(fn).suffix.lower()
    upload_dir = settings.upload_abs_path
    candidate_paths = [
        upload_dir / f"{fhash}{ext}",
        upload_dir / fn,
        settings.upload_abs_path / "uploads" / f"{fhash}{ext}",
        Path(settings.upload_dir) / f"{fhash}{ext}",
        Path(settings.upload_dir) / fn,
        Path(settings.documents_dir) / fn,
    ]
    for p in candidate_paths:
        if p.exists() and p.is_file():
            try:
                with open(p, "rb") as f:
                    return f.read()
            except Exception:
                pass

    raise HTTPException(
        status_code=status.HTTP_404_NOT_FOUND,
        detail=f"Source file for document '{doc.get('id')}' is not available in storage.",
    )


# ------------------------------------------------------------------------------
# 2. Documents Metadata & File Library Endpoints (User Scoped)
# ------------------------------------------------------------------------------
@app.get("/api/documents", response_model=DocumentsResponse, tags=["Documents"])
def get_documents(
    current_user: Dict[str, Any] = Depends(get_current_user),
    rag_app: RAGApplication = Depends(get_rag_app),
) -> DocumentsResponse:
    """
    Return all documents in the user's file library and catalog,
    with filename, file type, size, upload date, page count, and status.
    """
    user_id = str(current_user["id"])
    try:
        db_docs = repository.list_documents(user_id=user_id)
        if db_docs:
            doc_list: List[DocumentInfo] = []
            total_chunks = 0
            for doc in db_docs:
                chunks = doc.get("chunk_count") or 0
                total_chunks += chunks
                fn = doc.get("filename", "unknown")
                ext = Path(fn).suffix.lower().lstrip(".")
                doc_list.append(
                    DocumentInfo(
                        id=doc["id"],
                        filename=fn,
                        file_type=ext.upper() if ext else "FILE",
                        file_size=doc.get("file_size"),
                        file_hash=doc.get("file_hash"),
                        chunk_count=chunks,
                        page_count=doc.get("page_count"),
                        sections=[],
                        status=doc.get("status", "ready"),
                        created_at=doc.get("created_at"),
                        storage_path=doc.get("storage_path"),
                    )
                )
            return DocumentsResponse(
                total_documents=len(doc_list),
                total_chunks=total_chunks,
                documents=doc_list,
            )

        # Fallback to vector store for legacy documents
        indexed_map = rag_app.vector_store.get_indexed_files(user_id=user_id)
        raw = rag_app.vector_store.collection.get(
            where={"user_id": user_id},
            include=["metadatas"],
        )
        metadatas = raw.get("metadatas", []) or []
        total_chunks = len(metadatas)

        if total_chunks == 0:
            return DocumentsResponse(
                total_documents=0,
                total_chunks=0,
                documents=[],
            )

        # Aggregate per-document statistics
        doc_stats: Dict[str, Dict[str, Any]] = {}
        for meta in metadatas:
            if not meta:
                continue
            fn = meta.get("filename", "unknown")
            if fn not in doc_stats:
                doc_stats[fn] = {
                    "id": meta.get("document_id"),
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
            ext = Path(fn).suffix.lower().lstrip(".")
            doc_list.append(
                DocumentInfo(
                    id=info.get("id"),
                    filename=info["filename"],
                    file_type=ext.upper() if ext else "FILE",
                    file_hash=info["file_hash"],
                    chunk_count=info["chunk_count"],
                    page_count=len(info["pages"]) if info["pages"] else None,
                    sections=sorted(list(info["sections"]))[:10],
                    status="ready",
                )
            )

        return DocumentsResponse(
            total_documents=len(doc_list),
            total_chunks=total_chunks,
            documents=doc_list,
        )
    except Exception as e:
        logger.error(f"Error retrieving documents metadata for user {user_id}: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to retrieve document metadata from vector store."
        )


@app.get(
    "/api/documents/{document_id}",
    response_model=DocumentRecordResponse,
    tags=["Documents"],
)
def get_document(
    document_id: str,
    current_user: Dict[str, Any] = Depends(get_current_user),
) -> DocumentRecordResponse:
    """
    Retrieve a single document from the catalog by ID, verifying ownership.
    """
    user_id = str(current_user["id"])
    doc = repository.get_document_by_id(document_id, user_id=user_id)
    if not doc:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Document '{document_id}' not found.",
        )
    return DocumentRecordResponse(**doc)


@app.get("/api/documents/{document_id}/download", tags=["Documents"])
def download_document(
    document_id: str,
    current_user: Dict[str, Any] = Depends(get_current_user_from_header_or_query),
) -> Response:
    """
    Download original file payload verifying user ownership.
    """
    user_id = str(current_user["id"])
    doc = repository.get_document_by_id(document_id, user_id=user_id)
    if not doc:
        raise HTTPException(status_code=404, detail=f"Document '{document_id}' not found.")
    data = _fetch_document_bytes(doc, user_id)
    filename = doc.get("filename", "document")
    return Response(
        content=data,
        media_type="application/octet-stream",
        headers={"Content-Disposition": f'attachment; filename="{filename}"'},
    )


@app.get("/api/documents/{document_id}/view", tags=["Documents"])
def view_document(
    document_id: str,
    current_user: Dict[str, Any] = Depends(get_current_user_from_header_or_query),
) -> Response:
    """
    View file content inline where browser supported (PDF, text, images).
    Office documents (DOCX, XLSX, PPTX) return an attachment download.
    """
    user_id = str(current_user["id"])
    doc = repository.get_document_by_id(document_id, user_id=user_id)
    if not doc:
        raise HTTPException(status_code=404, detail=f"Document '{document_id}' not found.")
    data = _fetch_document_bytes(doc, user_id)
    filename = doc.get("filename", "document")
    ext = Path(filename).suffix.lower()

    viewable_mimes = {
        ".pdf": "application/pdf",
        ".txt": "text/plain; charset=utf-8",
        ".md": "text/plain; charset=utf-8",
        ".markdown": "text/plain; charset=utf-8",
        ".png": "image/png",
        ".jpg": "image/jpeg",
        ".jpeg": "image/jpeg",
        ".webp": "image/webp",
        ".svg": "image/svg+xml",
        ".csv": "text/plain; charset=utf-8",
    }
    if ext in viewable_mimes:
        mime = viewable_mimes[ext]
        disp = "inline"
    else:
        mime = "application/octet-stream"
        disp = "attachment"

    return Response(
        content=data,
        media_type=mime,
        headers={"Content-Disposition": f'{disp}; filename="{filename}"'},
    )


@app.delete(
    "/api/documents/{document_id}",
    status_code=status.HTTP_204_NO_CONTENT,
    tags=["Documents"],
)
def delete_document(
    document_id: str,
    current_user: Dict[str, Any] = Depends(get_current_user),
    rag_app: RAGApplication = Depends(get_rag_app),
) -> Response:
    """
    Delete a document record, physical/cloud storage file, and associated vector chunks.
    """
    user_id = str(current_user["id"])
    doc = repository.get_document_by_id(document_id, user_id=user_id)
    if not doc:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Document '{document_id}' not found.",
        )

    # 1. Clean up storage object
    storage_path = doc.get("storage_path")
    if storage_path:
        try:
            get_storage_service().delete(storage_path)
        except Exception as e:
            logger.warning(f"Failed to delete storage file '{storage_path}': {e}")

    # 2. Delete database row (cascades to junction tables)
    deleted = repository.delete_document(document_id, user_id=user_id)
    if not deleted:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Document '{document_id}' not found.",
        )

    # 3. Clean up associated vector chunks
    rag_app.vector_store.delete_by_document_id(document_id, user_id=user_id)
    return Response(status_code=status.HTTP_204_NO_CONTENT)


# ------------------------------------------------------------------------------
# 2b. File Conversion Endpoints (User Scoped)
# ------------------------------------------------------------------------------
@app.get("/api/documents/{document_id}/convert/targets", response_model=ConversionTargetsResponse, tags=["Conversion"])
def get_conversion_targets(
    document_id: str,
    current_user: Dict[str, Any] = Depends(get_current_user),
) -> ConversionTargetsResponse:
    """Get list of supported conversion target formats for a specific document."""
    user_id = str(current_user["id"])
    doc = repository.get_document_by_id(document_id, user_id=user_id)
    if not doc:
        raise HTTPException(status_code=404, detail=f"Document '{document_id}' not found.")
    ext = Path(doc["filename"]).suffix.lower()
    targets = get_supported_targets(ext)
    return ConversionTargetsResponse(source_format=ext, supported_targets=targets)


@app.post("/api/documents/{document_id}/convert", tags=["Conversion"])
@app.get("/api/documents/{document_id}/convert", tags=["Conversion"])
def convert_document(
    document_id: str,
    target_format: Optional[str] = None,
    payload: Optional[ConversionRequest] = None,
    current_user: Dict[str, Any] = Depends(get_current_user_from_header_or_query),
    rag_app: RAGApplication = Depends(get_rag_app),
) -> Response:
    """Convert an existing document to a supported target format and return converted bytes."""
    user_id = str(current_user["id"])
    doc = repository.get_document_by_id(document_id, user_id=user_id)
    if not doc:
        raise HTTPException(status_code=404, detail=f"Document '{document_id}' not found.")

    tgt = target_format or (payload.target_format if payload else None)
    if not tgt or not tgt.strip():
        raise HTTPException(status_code=400, detail="Missing required 'target_format' parameter.")

    tgt = tgt.strip().lower()
    if not tgt.startswith("."):
        tgt = f".{tgt}"

    source_bytes = _fetch_document_bytes(doc, user_id)

    try:
        converted_bytes, out_filename, mime_type = DocumentConverter.convert(
            source_bytes=source_bytes,
            source_filename=doc["filename"],
            target_format=tgt,
            llm_generator=rag_app.generator if tgt == ".pptx" else None,
        )
    except UnsupportedConversionError as ue:
        raise HTTPException(status_code=400, detail=str(ue))
    except Exception as e:
        logger.error(f"Conversion failed for document {document_id}: {e}")
        raise HTTPException(status_code=500, detail=f"Conversion error: {e}")

    # Persist in storage if Supabase/persistent storage configured
    try:
        storage = get_storage_service()
        storage.upload_file(
            file_bytes=converted_bytes,
            filename=out_filename,
            user_id=user_id,
            document_id=f"converted_{document_id}",
            content_type=mime_type,
        )
    except Exception as e:
        logger.warning(f"Could not persist converted file in storage: {e}")

    return Response(
        content=converted_bytes,
        media_type=mime_type,
        headers={"Content-Disposition": f'attachment; filename="{out_filename}"'},
    )



# ------------------------------------------------------------------------------
# 3. Global Chat Endpoint (User Scoped)
# ------------------------------------------------------------------------------
@app.post("/api/chat", response_model=ChatResponse, tags=["Chat"])
def chat(
    request: ChatRequest,
    current_user: Dict[str, Any] = Depends(get_current_user),
    rag_app: RAGApplication = Depends(get_rag_app),
) -> ChatResponse:
    """
    Execute grounded RAG query through existing retriever and LLM generator,
    strictly scoped to authenticated user's documents.
    """
    user_id = str(current_user["id"])
    query = request.query.strip()
    if not query:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Query string must not be empty.",
        )

    # Check for natural language document conversion request
    user_docs = repository.list_documents(user_id=user_id)
    conv_info = parse_conversion_request(query, user_docs)
    if conv_info:
        if conv_info.clarification_needed:
            return ChatResponse(answer=conv_info.clarification_needed, sources=[])
        doc = conv_info.matched_document
        source_bytes = _fetch_document_bytes(doc, user_id)
        converted_bytes, out_filename, mime_type = DocumentConverter.convert(
            source_bytes=source_bytes,
            source_filename=doc["filename"],
            target_format=conv_info.target_format,
            llm_generator=rag_app.generator if conv_info.target_format == ".pptx" else None,
        )
        try:
            storage = get_storage_service()
            storage.upload_file(
                file_bytes=converted_bytes,
                filename=out_filename,
                user_id=user_id,
                document_id=f"converted_{doc['id']}",
                content_type=mime_type,
            )
        except Exception as se:
            logger.warning(f"Could not persist converted file in storage: {se}")

        token = create_access_token({"sub": user_id})
        clean_target = conv_info.target_format.lstrip(".")
        download_url = f"/api/documents/{doc['id']}/convert?target_format={clean_target}&token={token}"
        size_kb = round(len(converted_bytes) / 1024, 1)
        ans = (
            f"Successfully converted **{doc['filename']}** to **{out_filename}** ({size_kb} KB).\n\n"
            f"[📥 Click here to download {out_filename}]({download_url})"
        )
        return ChatResponse(answer=ans, sources=[])

    try:
        user_where = {"user_id": user_id}
        chunks: List[RetrievedChunk] = []
        plan = None
        num_pred, num_ctx = None, None

        if hasattr(rag_app.retriever, "retrieve_adaptive"):
            try:
                res = rag_app.retriever.retrieve_adaptive(query=query, where=user_where)
                if isinstance(res, tuple) and len(res) == 3:
                    chunks, plan, coverage = res
                    budget = getattr(plan, "generation_budget", None)
                    num_pred = getattr(budget, "num_predict", None)
                    num_ctx = getattr(budget, "num_ctx", None)
                elif isinstance(res, list):
                    chunks = res
                else:
                    chunks = rag_app.retriever.retrieve(query, where=user_where)
            except Exception:
                chunks = rag_app.retriever.retrieve(query, where=user_where)
        else:
            chunks = rag_app.retriever.retrieve(query, where=user_where)

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
        logger.error(f"Error processing chat query '{query}' for user {user_id}: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to generate answer from documents.",
        )


# ------------------------------------------------------------------------------
# 4. Ingest Endpoint (User Scoped)
# ------------------------------------------------------------------------------
@app.post("/api/ingest", response_model=IngestResponse, tags=["Ingestion"])
def trigger_ingestion(
    current_user: Dict[str, Any] = Depends(get_current_user),
    rag_app: RAGApplication = Depends(get_rag_app),
) -> IngestResponse:
    """
    Trigger incremental document ingestion on the documents directory,
    tagging chunks with the authenticated user ID.
    """
    user_id = str(current_user["id"])
    try:
        res = rag_app.pipeline.ingest_directory(force=False, user_id=user_id)
        docs_count = res.get("documents", 0)
        chunks_count = res.get("chunks", 0)
        return IngestResponse(
            status="success",
            documents_processed=docs_count,
            chunks_stored=chunks_count,
            message=f"Ingestion completed. Processed {docs_count} documents, stored {chunks_count} chunks.",
        )
    except Exception as e:
        logger.error(f"Ingestion failed for user {user_id}: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Document ingestion failed.",
        )


# ------------------------------------------------------------------------------
# 5. Reindex Endpoint (User Scoped)
# ------------------------------------------------------------------------------
@app.post("/api/reindex", response_model=ReindexResponse, tags=["Ingestion"])
def trigger_reindex(
    current_user: Dict[str, Any] = Depends(get_current_user),
    rag_app: RAGApplication = Depends(get_rag_app),
) -> ReindexResponse:
    """
    Clear vector store and re-index all documents from scratch for the authenticated user.
    """
    user_id = str(current_user["id"])
    try:
        rag_app.vector_store.clear()
        res = rag_app.pipeline.ingest_directory(force=True, user_id=user_id)
        docs_count = res.get("documents", 0)
        chunks_count = res.get("chunks", 0)
        return ReindexResponse(
            status="success",
            documents_processed=docs_count,
            chunks_stored=chunks_count,
            message=f"Reindexing completed. Cleared database and indexed {chunks_count} chunks from {docs_count} documents.",
        )
    except Exception as e:
        logger.error(f"Reindexing failed for user {user_id}: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Document reindexing failed.",
        )


# ------------------------------------------------------------------------------
# 6. Chat Session CRUD Endpoints (User Scoped)
# ------------------------------------------------------------------------------
@app.get("/api/chats", response_model=List[ChatSessionResponse], tags=["Chats"])
def list_chats(
    current_user: Dict[str, Any] = Depends(get_current_user),
) -> List[ChatSessionResponse]:
    """
    Retrieve all chat sessions owned by the authenticated user ordered by updated_at DESC.
    """
    user_id = str(current_user["id"])
    try:
        chats = repository.list_chats(user_id=user_id)
        return [ChatSessionResponse(**c) for c in chats]
    except Exception as e:
        logger.error(f"Error listing chats for user {user_id}: {e}")
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
def create_chat(
    request: Optional[ChatCreateRequest] = None,
    current_user: Dict[str, Any] = Depends(get_current_user),
) -> ChatSessionResponse:
    """
    Create a new chat session owned by the authenticated user.
    """
    user_id = str(current_user["id"])
    try:
        title = (request.title if (request and request.title) else "New Chat")
        chat_data = repository.create_chat(title=title, user_id=user_id)
        return ChatSessionResponse(**chat_data)
    except Exception as e:
        logger.error(f"Error creating chat for user {user_id}: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to create chat.",
        )


@app.get("/api/chats/{chat_id}", response_model=ChatSessionResponse, tags=["Chats"])
def get_chat(
    chat_id: str,
    current_user: Dict[str, Any] = Depends(get_current_user),
) -> ChatSessionResponse:
    """
    Retrieve a single chat session by ID, verifying user ownership.
    """
    user_id = str(current_user["id"])
    chat_data = repository.get_chat(chat_id, user_id=user_id)
    if not chat_data:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Chat '{chat_id}' not found.",
        )
    return ChatSessionResponse(**chat_data)


@app.patch("/api/chats/{chat_id}", response_model=ChatSessionResponse, tags=["Chats"])
def rename_chat(
    chat_id: str,
    request: ChatRenameRequest,
    current_user: Dict[str, Any] = Depends(get_current_user),
) -> ChatSessionResponse:
    """
    Rename an existing chat session, verifying user ownership.
    """
    user_id = str(current_user["id"])
    try:
        updated = repository.rename_chat(chat_id, request.title, user_id=user_id)
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
        logger.error(f"Error renaming chat '{chat_id}' for user {user_id}: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to rename chat.",
        )


@app.delete("/api/chats/{chat_id}", status_code=status.HTTP_204_NO_CONTENT, tags=["Chats"])
def delete_chat(
    chat_id: str,
    current_user: Dict[str, Any] = Depends(get_current_user),
) -> Response:
    """
    Delete a chat session and all its cascade records, verifying user ownership.
    """
    user_id = str(current_user["id"])
    try:
        deleted = repository.delete_chat(chat_id, user_id=user_id)
        if not deleted:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"Chat '{chat_id}' not found.",
            )
        return Response(status_code=status.HTTP_204_NO_CONTENT)
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error deleting chat '{chat_id}' for user {user_id}: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to delete chat.",
        )


# ------------------------------------------------------------------------------
# 7. Chat Document Management Endpoints (User Scoped)
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
    current_user: Dict[str, Any] = Depends(get_current_user),
    rag_app: RAGApplication = Depends(get_rag_app),
) -> DocumentUploadResponse:
    """
    Upload and attach a document to a chat session, verifying ownership.
    Supports PDF, DOCX, PPTX, XLSX, CSV, TXT, MD, and image files.
    Enforces 50MB size limit, sanitizes filename, computes SHA-256,
    stores under backend/data/uploads/, and indexes into ChromaDB with user_id.
    """
    user_id = str(current_user["id"])

    # 1. Verify chat exists and is owned by authenticated user
    chat = repository.get_chat(chat_id, user_id=user_id)
    if not chat:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Chat '{chat_id}' not found.",
        )

    # 2. Filename validation and sanitization against path traversal
    raw_filename = file.filename or ""
    safe_name = Path(raw_filename).name
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

    # 4. Magic-byte validation
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

    # 6. Save file under backend/data/uploads/ using hash + original extension
    upload_dir = settings.upload_abs_path
    upload_dir.mkdir(parents=True, exist_ok=True)
    stored_path = upload_dir / f"{file_hash}{file_ext}"
    if not stored_path.exists():
        with open(stored_path, "wb") as f:
            f.write(content)

    safe_rel_path = f"uploads/{file_hash}{file_ext}"

    # Persistent storage upload (Supabase Storage in production, Local in dev)
    storage = get_storage_service()
    deterministic_doc_id = f"doc_{hashlib.sha256(f'{user_id}:{file_hash}'.encode()).hexdigest()[:16]}"
    try:
        storage_path = storage.upload_file(
            file_bytes=content,
            filename=safe_name,
            user_id=user_id,
            document_id=deterministic_doc_id,
            content_type=file.content_type,
        )
    except Exception as e:
        logger.warning(f"Could not upload file to primary storage: {e}")
        storage_path = safe_rel_path

    # 7. Check if document record already exists in catalog for THIS authenticated user
    existing_doc = repository.get_document_by_hash(file_hash, user_id=user_id)
    if existing_doc is not None:
        doc_id = existing_doc["id"]
        repository.attach_document_to_chat(chat_id, doc_id, user_id=user_id)

        # If already ready, reuse without reprocessing
        if existing_doc["status"] == "ready":
            attached_docs = repository.list_chat_documents(chat_id, user_id=user_id)
            attached_rec = next((d for d in attached_docs if d["id"] == doc_id), existing_doc)
            return DocumentUploadResponse(
                message="Existing document attached to chat without reprocessing.",
                document=DocumentRecordResponse(**attached_rec),
            )
    else:
        # Check if already present in vector store for THIS user
        indexed_files = rag_app.vector_store.get_indexed_files(user_id=user_id)
        if file_hash in indexed_files.values():
            res = rag_app.vector_store.collection.get(
                where={"$and": [{"file_hash": file_hash}, {"user_id": user_id}]}
            )
            cnt = len(res.get("ids", []))
            new_doc = repository.create_document(
                document_id=deterministic_doc_id,
                filename=safe_name,
                file_hash=file_hash,
                file_size=file_size,
                chunk_count=cnt,
                storage_path=storage_path,
                status="ready",
                user_id=user_id,
            )
            doc_id = new_doc["id"]
            repository.attach_document_to_chat(chat_id, doc_id, user_id=user_id)
            attached_docs = repository.list_chat_documents(chat_id, user_id=user_id)
            attached_rec = next((d for d in attached_docs if d["id"] == doc_id), new_doc)
            return DocumentUploadResponse(
                message="Document registered from existing vector store and attached to chat.",
                document=DocumentRecordResponse(**attached_rec),
            )
        else:
            # Register new pending document for authenticated user
            new_doc = repository.create_document(
                document_id=deterministic_doc_id,
                filename=safe_name,
                file_hash=file_hash,
                file_size=file_size,
                storage_path=storage_path,
                status="pending",
                user_id=user_id,
            )
            doc_id = new_doc["id"]
            repository.attach_document_to_chat(chat_id, doc_id, user_id=user_id)

    # 8. Document Processing / Indexing with user_id
    repository.update_document_status(doc_id, status="processing", user_id=user_id)
    try:
        ingest_res = rag_app.pipeline.ingest_uploaded_document(
            file_path=stored_path,
            original_filename=safe_name,
            document_id=doc_id,
            file_hash=file_hash,
            user_id=user_id,
        )
        ready_doc = repository.update_document_status(
            doc_id,
            status="ready",
            page_count=ingest_res.get("page_count", 0),
            chunk_count=ingest_res.get("chunk_count", 0),
            storage_path=safe_rel_path,
            user_id=user_id,
        )
        attached_docs = repository.list_chat_documents(chat_id, user_id=user_id)
        attached_rec = next((d for d in attached_docs if d["id"] == doc_id), ready_doc)
        return DocumentUploadResponse(
            message="Document uploaded, parsed, and indexed successfully.",
            document=DocumentRecordResponse(**attached_rec),
        )
    except Exception as e:
        logger.error(f"Error processing uploaded document {doc_id} for user {user_id}: {e}")
        failed_doc = repository.update_document_status(
            doc_id,
            status="failed",
            error_message=f"Failed to parse and index document: {type(e).__name__}",
            user_id=user_id,
        )
        attached_docs = repository.list_chat_documents(chat_id, user_id=user_id)
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
def list_chat_documents(
    chat_id: str,
    current_user: Dict[str, Any] = Depends(get_current_user),
) -> List[DocumentRecordResponse]:
    """
    List all documents attached to a specific chat session, verifying ownership.
    """
    user_id = str(current_user["id"])
    chat = repository.get_chat(chat_id, user_id=user_id)
    if not chat:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Chat '{chat_id}' not found.",
        )
    docs = repository.list_chat_documents(chat_id, user_id=user_id)
    return [DocumentRecordResponse(**d) for d in docs]


@app.delete(
    "/api/chats/{chat_id}/documents/{document_id}",
    status_code=status.HTTP_204_NO_CONTENT,
    tags=["Chat Documents"],
)
def remove_document_from_chat(
    chat_id: str,
    document_id: str,
    current_user: Dict[str, Any] = Depends(get_current_user),
) -> Response:
    """
    Detach a document from a chat session, verifying ownership.
    Does NOT delete the document from the catalog or ChromaDB.
    """
    user_id = str(current_user["id"])
    chat = repository.get_chat(chat_id, user_id=user_id)
    if not chat:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Chat '{chat_id}' not found.",
        )

    detached = repository.detach_document_from_chat(chat_id, document_id, user_id=user_id)
    if not detached:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Document '{document_id}' is not attached to chat '{chat_id}'.",
        )
    return Response(status_code=status.HTTP_204_NO_CONTENT)


@app.post(
    "/api/chats/{chat_id}/documents/{document_id}",
    response_model=DocumentRecordResponse,
    tags=["Chat Documents"],
)
def attach_existing_document_to_chat(
    chat_id: str,
    document_id: str,
    current_user: Dict[str, Any] = Depends(get_current_user),
) -> DocumentRecordResponse:
    """Attach an existing catalog document to a chat session, verifying user ownership."""
    user_id = str(current_user["id"])
    chat = repository.get_chat(chat_id, user_id=user_id)
    if not chat:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Chat '{chat_id}' not found.",
        )
    doc = repository.get_document_by_id(document_id, user_id=user_id)
    if not doc:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Document '{document_id}' not found.",
        )
    repository.attach_document_to_chat(chat_id, document_id, user_id=user_id)
    attached_docs = repository.list_chat_documents(chat_id, user_id=user_id)
    rec = next((d for d in attached_docs if d["id"] == document_id), doc)
    return DocumentRecordResponse(**rec)


# ------------------------------------------------------------------------------
# 8. Chat-Scoped Query and Message Persistence (User Scoped)
# ------------------------------------------------------------------------------
@app.post(
    "/api/chats/{chat_id}/chat",
    response_model=ChatResponse,
    tags=["Chat"],
)
def chat_in_session(
    chat_id: str,
    request: ChatRequest,
    current_user: Dict[str, Any] = Depends(get_current_user),
    rag_app: RAGApplication = Depends(get_rag_app),
) -> ChatResponse:
    """
    Execute a chat question scoped strictly to documents attached to this chat session
    and owned by the authenticated user.
    Persists user message and assistant answer with sources into SQLite.
    """
    user_id = str(current_user["id"])
    chat = repository.get_chat(chat_id, user_id=user_id)
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

    # Auto-title chat from first query
    if chat.get("title") in ("New Chat", "", None):
        auto_title = generate_chat_title(query)
        repository.rename_chat(chat_id, auto_title, user_id=user_id)

    # Retrieve ready documents attached to this chat
    attached_docs = repository.list_chat_documents(chat_id, user_id=user_id)
    ready_docs = [d for d in attached_docs if d["status"] == "ready"]

    # Check for natural language document conversion request
    conv_info = parse_conversion_request(query, ready_docs)
    if conv_info:
        if conv_info.clarification_needed:
            ans = conv_info.clarification_needed
            repository.create_message(chat_id=chat_id, role="user", content=query, user_id=user_id)
            repository.create_message(chat_id=chat_id, role="assistant", content=ans, sources_json=[], user_id=user_id)
            return ChatResponse(answer=ans, sources=[])
        doc = conv_info.matched_document
        source_bytes = _fetch_document_bytes(doc, user_id)
        converted_bytes, out_filename, mime_type = DocumentConverter.convert(
            source_bytes=source_bytes,
            source_filename=doc["filename"],
            target_format=conv_info.target_format,
            llm_generator=rag_app.generator if conv_info.target_format == ".pptx" else None,
        )
        try:
            storage = get_storage_service()
            storage.upload_file(
                file_bytes=converted_bytes,
                filename=out_filename,
                user_id=user_id,
                document_id=f"converted_{doc['id']}",
                content_type=mime_type,
            )
        except Exception as se:
            logger.warning(f"Could not persist converted file in storage: {se}")

        token = create_access_token({"sub": user_id})
        clean_target = conv_info.target_format.lstrip(".")
        download_url = f"/api/documents/{doc['id']}/convert?target_format={clean_target}&token={token}"
        size_kb = round(len(converted_bytes) / 1024, 1)
        ans = (
            f"Successfully converted **{doc['filename']}** to **{out_filename}** ({size_kb} KB).\n\n"
            f"[📥 Click here to download {out_filename}]({download_url})"
        )
        repository.create_message(chat_id=chat_id, role="user", content=query, user_id=user_id)
        repository.create_message(chat_id=chat_id, role="assistant", content=ans, sources_json=[], user_id=user_id)
        return ChatResponse(answer=ans, sources=[])

    # If no ready documents are attached, return clean response and persist
    if not ready_docs:
        notice = "No ready documents are attached to this chat. Please upload and attach a PDF document before asking questions."
        repository.create_message(chat_id=chat_id, role="user", content=query, user_id=user_id)
        repository.create_message(chat_id=chat_id, role="assistant", content=notice, sources_json=[], user_id=user_id)
        return ChatResponse(answer=notice, sources=[])

    # Construct strict Chroma filter restricting retrieval to authenticated user + attached ready docs
    user_cond = {"user_id": user_id}
    if len(ready_docs) == 1:
        doc = ready_docs[0]
        if doc.get("file_hash"):
            doc_cond = {
                "$or": [
                    {"document_id": doc["id"]},
                    {"file_hash": doc["file_hash"]},
                ]
            }
        else:
            doc_cond = {"document_id": doc["id"]}
    else:
        doc_ids = [d["id"] for d in ready_docs]
        doc_hashes = [d["file_hash"] for d in ready_docs if d.get("file_hash")]
        doc_cond = {
            "$or": [
                {"document_id": {"$in": doc_ids}},
                {"file_hash": {"$in": doc_hashes}},
            ]
        }
    where_filter = {"$and": [user_cond, doc_cond]}

    recent_messages = repository.list_messages(chat_id, user_id=user_id)

    try:
        import time
        start_time = time.time()

        # Retrieve candidate chunks scoped strictly to user and chat documents using Adaptive RAG
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
            f"USER: {user_id}\n"
            f"QUERY: {query}\n"
            f"QUERY TYPE: {plan.strategy.value}\n"
            f"TARGET DOCUMENTS: {[d.filename for d in plan.target_documents]}\n"
            f"RETRIEVAL PLAN: {plan.strategy.name} (k={plan.candidate_k})\n"
            f"FINAL CHUNK COUNT: {len(chunks)}\n"
            f"PAGE COVERAGE: {coverage.pages_covered}\n"
            f"DOCUMENT COVERAGE: {coverage.documents_covered}\n"
            f"CONTEXT SIZE: {sum(len(c.text) for c in chunks)} chars\n"
            f"SOURCES: {len(sources)}\n"
            f"TOTAL TIME: {elapsed:.2f}s\n"
            f"-------------------------------"
        )

        # Persist messages in database scoped to user
        repository.create_message(chat_id=chat_id, role="user", content=query, user_id=user_id)
        sources_payload = [s.model_dump() for s in sources]
        repository.create_message(
            chat_id=chat_id,
            role="assistant",
            content=answer,
            sources_json=sources_payload,
            user_id=user_id,
        )

        return ChatResponse(answer=answer, sources=sources)
    except Exception as e:
        logger.error(f"Error during chat in session '{chat_id}' for user {user_id}: {e}")
        safe_msg = "An error occurred while generating an answer from the attached documents."
        repository.create_message(chat_id=chat_id, role="user", content=query, user_id=user_id)
        repository.create_message(chat_id=chat_id, role="assistant", content=safe_msg, sources_json=[], user_id=user_id)
        return ChatResponse(answer=safe_msg, sources=[])


@app.post(
    "/api/chats/{chat_id}/stream",
    tags=["Chat"],
)
def chat_in_session_stream(
    chat_id: str,
    request: ChatRequest,
    current_user: Dict[str, Any] = Depends(get_current_user),
    rag_app: RAGApplication = Depends(get_rag_app),
) -> StreamingResponse:
    """
    Stream token chunks via Server-Sent Events (SSE) for low perceived latency,
    strictly scoped to authenticated user documents.
    Persists user message and completed assistant response with sources upon completion.
    """
    user_id = str(current_user["id"])
    chat = repository.get_chat(chat_id, user_id=user_id)
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

    # Auto-title chat from first query
    if chat.get("title") in ("New Chat", "", None):
        auto_title = generate_chat_title(query)
        repository.rename_chat(chat_id, auto_title, user_id=user_id)

    attached_docs = repository.list_chat_documents(chat_id, user_id=user_id)
    ready_docs = [d for d in attached_docs if d["status"] == "ready"]

    # Check for natural language document conversion request
    conv_info = parse_conversion_request(query, ready_docs)
    if conv_info:
        def conversion_stream():
            if conv_info.clarification_needed:
                ans = conv_info.clarification_needed
                repository.create_message(chat_id=chat_id, role="user", content=query, user_id=user_id)
                repository.create_message(chat_id=chat_id, role="assistant", content=ans, sources_json=[], user_id=user_id)
                yield f"data: {json.dumps({'type': 'token', 'token': ans})}\n\n"
                yield f"data: {json.dumps({'type': 'sources', 'sources': []})}\n\n"
                yield "data: [DONE]\n\n"
                return

            doc = conv_info.matched_document
            doc_fn = doc.get("filename", "document")
            intro_token = f"Converting **{doc_fn}** to {conv_info.target_label}..."
            yield f"data: {json.dumps({'type': 'token', 'token': intro_token})}\n\n"

            try:
                source_bytes = _fetch_document_bytes(doc, user_id)
                converted_bytes, out_filename, mime_type = DocumentConverter.convert(
                    source_bytes=source_bytes,
                    source_filename=doc["filename"],
                    target_format=conv_info.target_format,
                    llm_generator=rag_app.generator if conv_info.target_format == ".pptx" else None,
                )
                try:
                    storage = get_storage_service()
                    storage.upload_file(
                        file_bytes=converted_bytes,
                        filename=out_filename,
                        user_id=user_id,
                        document_id=f"converted_{doc['id']}",
                        content_type=mime_type,
                    )
                except Exception as se:
                    logger.warning(f"Could not persist converted file in storage: {se}")

                token = create_access_token({"sub": user_id})
                clean_target = conv_info.target_format.lstrip(".")
                download_url = f"/api/documents/{doc['id']}/convert?target_format={clean_target}&token={token}"
                size_kb = round(len(converted_bytes) / 1024, 1)

                msg = (
                    f"\n\nSuccessfully converted **{doc['filename']}** to **{out_filename}** ({size_kb} KB).\n\n"
                    f"[📥 Click here to download {out_filename}]({download_url})"
                )
                yield f"data: {json.dumps({'type': 'token', 'token': msg})}\n\n"
                repository.create_message(chat_id=chat_id, role="user", content=query, user_id=user_id)
                repository.create_message(
                    chat_id=chat_id,
                    role="assistant",
                    content=f"Successfully converted **{doc['filename']}** to **{out_filename}** ({size_kb} KB).\n\n[📥 Click here to download {out_filename}]({download_url})",
                    sources_json=[],
                    user_id=user_id,
                )
                yield f"data: {json.dumps({'type': 'sources', 'sources': []})}\n\n"
                yield "data: [DONE]\n\n"
            except Exception as e:
                err_msg = f"\n\nConversion failed: {e}"
                yield f"data: {json.dumps({'type': 'token', 'token': err_msg})}\n\n"
                yield f"data: {json.dumps({'type': 'sources', 'sources': []})}\n\n"
                yield "data: [DONE]\n\n"

        return StreamingResponse(conversion_stream(), media_type="text/event-stream")

    if not ready_docs:
        notice = "No ready documents are attached to this chat. Please upload and attach a PDF document before asking questions."
        repository.create_message(chat_id=chat_id, role="user", content=query, user_id=user_id)
        repository.create_message(chat_id=chat_id, role="assistant", content=notice, sources_json=[], user_id=user_id)

        def empty_generator():
            yield f"data: {json.dumps({'type': 'token', 'token': notice})}\n\n"
            yield f"data: {json.dumps({'type': 'sources', 'sources': []})}\n\n"
            yield "data: [DONE]\n\n"

        return StreamingResponse(empty_generator(), media_type="text/event-stream")

    user_cond = {"user_id": user_id}
    if len(ready_docs) == 1:
        doc = ready_docs[0]
        if doc.get("file_hash"):
            doc_cond = {
                "$or": [
                    {"document_id": doc["id"]},
                    {"file_hash": doc["file_hash"]},
                ]
            }
        else:
            doc_cond = {"document_id": doc["id"]}
    else:
        doc_ids = [d["id"] for d in ready_docs]
        doc_hashes = [d["file_hash"] for d in ready_docs if d.get("file_hash")]
        doc_cond = {
            "$or": [
                {"document_id": {"$in": doc_ids}},
                {"file_hash": {"$in": doc_hashes}},
            ]
        }
    where_filter = {"$and": [user_cond, doc_cond]}

    recent_messages = repository.list_messages(chat_id, user_id=user_id)

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

            if not accumulated_tokens:
                logger.warning(f"generate_answer_stream yielded 0 tokens in chat '{chat_id}'. Attempting fallback generation.")
                try:
                    fallback_answer = rag_app.generator.generate_answer(
                        question=effective_q,
                        chunks=chunks,
                        plan=plan,
                        num_predict=plan.generation_budget.num_predict,
                        num_ctx=plan.generation_budget.num_ctx,
                    )
                    if fallback_answer:
                        accumulated_tokens.append(fallback_answer)
                        yield f"data: {json.dumps({'type': 'token', 'token': fallback_answer})}\n\n"
                except Exception as fb_err:
                    logger.error(f"Fallback generation failed in chat '{chat_id}': {fb_err}")

            if not accumulated_tokens:
                default_msg = "I was unable to retrieve a response from the model. Please verify your query or document context."
                accumulated_tokens.append(default_msg)
                yield f"data: {json.dumps({'type': 'token', 'token': default_msg})}\n\n"

            full_answer = "".join(accumulated_tokens).strip()
            repository.create_message(chat_id=chat_id, role="user", content=query, user_id=user_id)
            sources_payload = [s.model_dump() for s in sources]
            repository.create_message(
                chat_id=chat_id,
                role="assistant",
                content=full_answer,
                sources_json=sources_payload,
                user_id=user_id,
            )
            yield f"data: {json.dumps({'type': 'sources', 'sources': sources_payload})}\n\n"
            yield "data: [DONE]\n\n"
        except Exception as err:
            logger.error(f"Error during streaming in chat '{chat_id}' for user {user_id}: {err}")
            yield f"data: {json.dumps({'type': 'error', 'error': 'Failed to complete streaming response.'})}\n\n"
            yield f"data: {json.dumps({'type': 'error', 'error': str(err)})}\n\n"

    return StreamingResponse(event_stream(), media_type="text/event-stream")


# ------------------------------------------------------------------------------
# 9. Message History Endpoint (User Scoped)
# ------------------------------------------------------------------------------
@app.get(
    "/api/chats/{chat_id}/messages",
    response_model=List[ChatMessageResponse],
    tags=["Chat Messages"],
)
def get_chat_messages(
    chat_id: str,
    current_user: Dict[str, Any] = Depends(get_current_user),
) -> List[ChatMessageResponse]:
    """
    Retrieve chronological message history for a specific chat session, verifying user ownership.
    """
    user_id = str(current_user["id"])
    chat = repository.get_chat(chat_id, user_id=user_id)
    if not chat:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Chat '{chat_id}' not found.",
        )

    raw_messages = repository.list_messages(chat_id, user_id=user_id)
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
