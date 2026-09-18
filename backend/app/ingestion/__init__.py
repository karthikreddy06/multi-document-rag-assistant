"""
Document ingestion, loading, chunking, and pipeline execution module.

Supports multi-format document ingestion (PDF, DOCX, PPTX, XLSX, CSV, TXT, MD, Images)
via the ParserRegistry abstraction.
"""

from app.models import Document, Chunk
from .loader import PDFLoader, FileTypeDetector
from .chunker import DocumentChunker
from .pipeline import IngestionPipeline
from .parsers import ParserRegistry, UnsupportedFileTypeError

__all__ = [
    "PDFLoader",
    "FileTypeDetector",
    "Document",
    "DocumentChunker",
    "Chunk",
    "IngestionPipeline",
    "ParserRegistry",
    "UnsupportedFileTypeError",
]
